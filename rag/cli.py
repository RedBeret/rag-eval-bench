"""CLI entry point for rag-eval-bench."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console

from rag.corpus.chunker import chunk_corpus
from rag.corpus.loader import load_corpus
from rag.eval.generation import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    _is_ollama_available,
    compute_generation_metrics,
    evaluate_generation,
)
from rag.eval.judge import (
    compute_agreement,
    compute_judge_metrics,
    export_grade_sheet,
    judge_results,
)
from rag.eval.retrieval import compute_retrieval_metrics, evaluate_retrieval
from rag.index.embedder import MockEmbedder, SentenceTransformerEmbedder
from rag.index.store import VectorStore
from rag.qa.generator import (
    deduplicate_questions,
    generate_qa_dataset,
    generate_unanswerable_items,
    select_generation_items,
)
from rag.qa.types import GenerationResult, QAItem
from rag.report import builder as report_builder

console = Console()


def _load_qa(path: Path) -> list[QAItem]:
    items = json.loads(path.read_text(encoding="utf-8"))
    return [QAItem.from_dict(i) for i in items]


def _save_qa(items: list[QAItem], path: Path) -> None:
    path.write_text(json.dumps([i.to_dict() for i in items], indent=2), encoding="utf-8")


def _make_embedder(model_name: str):
    if model_name == "mock":
        return MockEmbedder(dim=16)
    return SentenceTransformerEmbedder(model_name=model_name)


@click.group()
@click.version_option("1.1.0", prog_name="rag-eval")
def cli() -> None:
    """rag-eval-bench — benchmark RAG pipelines against synthetic ground truth.

    \b
    All data processed by this tool is synthetic.
    Meridian Fabrication Co. is a fictional company.
    """


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("corpus", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--persist-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="ChromaDB persist directory (default: in-memory).",
)
@click.option(
    "--model",
    default="all-MiniLM-L6-v2",
    show_default=True,
    help="Sentence-transformers embedding model.",
)
@click.option("--chunk-size", default=512, show_default=True, help="Max characters per chunk.")
@click.option("--reset", is_flag=True, help="Delete existing index before adding documents.")
def index(corpus: Path, persist_dir: Path | None, model: str, chunk_size: int, reset: bool) -> None:
    """Load and index a corpus file into the vector store.

    \b
    CORPUS is a CSV (parts or suppliers) or JSONL file.

    \b
    Example:
      rag-eval index parts_v1.csv --persist-dir .rag-index
    """
    console.print(f"[cyan]Loading corpus from {corpus}...[/cyan]")
    docs = load_corpus(corpus)
    console.print(f"  Loaded {len(docs):,} documents")

    chunks = chunk_corpus(docs, max_chars=chunk_size)
    console.print(f"  Chunked into {len(chunks):,} passages")

    embedder = _make_embedder(model)
    store = VectorStore(persist_dir=persist_dir, embedder=embedder)

    if reset:
        store.reset()
        console.print("  [yellow]Index reset.[/yellow]")

    console.print("[cyan]Embedding and indexing...[/cyan]")
    store.add_documents(chunks)
    console.print(f"[green]✓ Indexed {store.count():,} passages[/green]")


# ---------------------------------------------------------------------------
# qa-gen
# ---------------------------------------------------------------------------


@cli.command("qa-gen")
@click.argument("corpus", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--out", type=click.Path(path_type=Path), required=True, help="Output path for QA pairs (JSON)."
)
@click.option(
    "--max-items", default=200, show_default=True, help="Maximum number of QA pairs to generate."
)
@click.option(
    "--seed", default=42, show_default=True, help="Random seed for reproducible sampling."
)
@click.option(
    "--unanswerable",
    default=0,
    show_default=True,
    help="Also generate N questions the corpus cannot answer (scores correct-refusal rate).",
)
def qa_gen(corpus: Path, out: Path, max_items: int, seed: int, unanswerable: int) -> None:
    """Generate ground-truth QA pairs from a corpus file.

    \b
    Example:
      rag-eval qa-gen parts_v1.csv --out qa.json --max-items 100 --unanswerable 10
    """
    docs = load_corpus(corpus)
    console.print(f"[cyan]Loaded {len(docs):,} documents[/cyan]")

    items = generate_qa_dataset(docs, max_items=max_items, seed=seed)
    items = deduplicate_questions(items)
    console.print(f"[cyan]Generated {len(items):,} QA pairs (deduplicated)[/cyan]")

    if unanswerable > 0:
        items += generate_unanswerable_items(docs, n=unanswerable, seed=seed)
        console.print(f"[cyan]Added {unanswerable} unanswerable questions[/cyan]")

    _save_qa(items, out)
    console.print(f"[green]✓ Saved QA pairs → {out}[/green]")


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("qa", type=click.Path(exists=True, path_type=Path))
@click.argument("corpus", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--k", default=5, show_default=True, help="Number of documents to retrieve per query."
)
@click.option("--model", default="all-MiniLM-L6-v2", show_default=True, help="Embedding model.")
@click.option(
    "--out", type=click.Path(path_type=Path), default=None, help="Write JSON report to file."
)
def retrieval(qa: Path, corpus: Path, k: int, model: str, out: Path | None) -> None:
    """Benchmark retrieval quality on a QA dataset.

    \b
    QA is the ground-truth QA pairs JSON produced by qa-gen.
    CORPUS is the source CSV/JSONL file used to build the index.

    \b
    Example:
      rag-eval retrieval qa.json parts_v1.csv --k 5
    """
    qa_items = _load_qa(qa)
    console.print(f"[cyan]Loaded {len(qa_items):,} QA pairs[/cyan]")

    docs = load_corpus(corpus)
    chunks = chunk_corpus(docs)
    embedder = _make_embedder(model)
    store = VectorStore(embedder=embedder)
    store.add_documents(chunks)
    console.print(f"[cyan]Indexed {store.count():,} passages[/cyan]")

    results = evaluate_retrieval(qa_items, store, k=k)
    metrics = compute_retrieval_metrics(results)
    report_builder.print_retrieval_metrics(metrics)
    report_builder.print_summary(metrics)

    if out:
        report_builder.save_report(
            {
                "type": "retrieval",
                "k": k,
                "embedding_model": model,
                "metrics": metrics,
                "results": [r.to_dict() for r in results],
            },
            out,
        )


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("qa", type=click.Path(exists=True, path_type=Path))
@click.argument("corpus", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    show_default=True,
    help="Ollama model name (e.g. gemma3:4b, mistral).",
)
@click.option(
    "--ollama-url", default=DEFAULT_OLLAMA_URL, show_default=True, help="Ollama server URL."
)
@click.option("--k", default=3, show_default=True, help="Number of context documents to retrieve.")
@click.option("--max-items", default=50, show_default=True, help="Max QA items to evaluate.")
@click.option(
    "--out", type=click.Path(path_type=Path), default=None, help="Write JSON report to file."
)
def generation(
    qa: Path, corpus: Path, model: str, ollama_url: str, k: int, max_items: int, out: Path | None
) -> None:
    """Benchmark generation quality using a local Ollama model.

    \b
    Requires a running Ollama server with the target model pulled:
      ollama pull gemma3:4b
      ollama serve

    \b
    Example:
      rag-eval generation qa.json parts_v1.csv --model gemma3:4b
    """
    if not _is_ollama_available(ollama_url):
        console.print(f"[red]Ollama not available at {ollama_url}[/red]")
        console.print("  Start Ollama: [dim]ollama serve[/dim]")
        console.print(f"  Pull model:   [dim]ollama pull {model}[/dim]")
        sys.exit(1)

    console.print(f"[green]✓ Ollama available — model: {model}[/green]")

    qa_items = select_generation_items(_load_qa(qa), max_items=max_items)
    docs = load_corpus(corpus)
    chunks = chunk_corpus(docs)
    embedder = _make_embedder("all-MiniLM-L6-v2")
    store = VectorStore(embedder=embedder)
    store.add_documents(chunks)

    console.print(f"[cyan]Generating answers for {len(qa_items)} questions...[/cyan]")
    results = evaluate_generation(qa_items, store, model=model, base_url=ollama_url, k=k)
    metrics = compute_generation_metrics(results)
    report_builder.print_generation_metrics(metrics)
    report_builder.print_summary({}, metrics)

    if out:
        report_builder.save_report(
            {
                "type": "generation",
                "model": model,
                "k": k,
                "metrics": metrics,
                "results": [r.to_dict() for r in results],
            },
            out,
        )


# ---------------------------------------------------------------------------
# bench (full pipeline)
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("corpus", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    show_default=True,
    help="Ollama model (skipped if unavailable).",
)
@click.option(
    "--embed-model", default="all-MiniLM-L6-v2", show_default=True, help="Embedding model."
)
@click.option("--k", default=5, show_default=True, help="Retrieval top-k.")
@click.option(
    "--max-qa", default=100, show_default=True, help="Max QA pairs to generate and evaluate."
)
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option("--skip-generation", is_flag=True, help="Skip Ollama generation even if available.")
@click.option(
    "--unanswerable",
    default=5,
    show_default=True,
    help="Number of unanswerable questions to include (0 to disable).",
)
@click.option(
    "--out", type=click.Path(path_type=Path), default=None, help="Write full JSON report to file."
)
def bench(
    corpus: Path,
    model: str,
    embed_model: str,
    k: int,
    max_qa: int,
    seed: int,
    skip_generation: bool,
    unanswerable: int,
    out: Path | None,
) -> None:
    """Run the full benchmark pipeline: index → qa-gen → retrieval → [generation].

    \b
    Generation is skipped automatically if Ollama is not available.

    \b
    Example:
      rag-eval bench parts_v1.csv --model gemma3:4b --out bench_report.json
    """
    # Load and index
    docs = load_corpus(corpus)
    chunks = chunk_corpus(docs)
    embedder = _make_embedder(embed_model)
    store = VectorStore(embedder=embedder)
    store.add_documents(chunks)
    console.print(
        f"[green]✓ Indexed {store.count():,} passages from {len(docs):,} documents[/green]"
    )

    # QA generation
    qa_items = generate_qa_dataset(docs, max_items=max_qa, seed=seed)
    qa_items = deduplicate_questions(qa_items)
    if unanswerable > 0:
        qa_items += generate_unanswerable_items(docs, n=unanswerable, seed=seed)
    console.print(
        f"[green]✓ Generated {len(qa_items):,} QA pairs"
        + (f" ({unanswerable} unanswerable)" if unanswerable else "")
        + "[/green]"
    )

    # Retrieval eval
    retr_results = evaluate_retrieval(qa_items, store, k=k)
    retr_metrics = compute_retrieval_metrics(retr_results)
    report_builder.print_retrieval_metrics(retr_metrics)

    # Generation eval (optional)
    gen_metrics: dict = {}
    gen_results = []
    if skip_generation:
        console.print("[yellow]Skipping generation eval by request.[/yellow]")
    elif _is_ollama_available():
        console.print(f"[cyan]Ollama available — running generation eval with {model}...[/cyan]")
        gen_items = select_generation_items(qa_items, max_items=50)
        gen_results = evaluate_generation(gen_items, store, model=model, k=k)
        gen_metrics = compute_generation_metrics(gen_results)
        report_builder.print_generation_metrics(gen_metrics)
    else:
        console.print("[yellow]Ollama not available — skipping generation eval.[/yellow]")
        console.print(f"  To enable: [dim]ollama serve && ollama pull {model}[/dim]")

    report_builder.print_summary(retr_metrics, gen_metrics or None)

    if out:
        report_builder.save_report(
            {
                "corpus": str(corpus),
                "embedding_model": embed_model,
                "generation_model": model if gen_metrics else None,
                "k": k,
                "retrieval": {
                    "metrics": retr_metrics,
                    "results": [r.to_dict() for r in retr_results],
                },
                "generation": {
                    "metrics": gen_metrics,
                    "results": [r.to_dict() for r in gen_results],
                }
                if gen_metrics
                else None,
            },
            out,
        )


# ---------------------------------------------------------------------------
# judge (LLM-as-judge over a generation report)
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("report", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--judge-model", required=True, help="Ollama judge model (must differ from the answer model)."
)
@click.option(
    "--ollama-url", default=DEFAULT_OLLAMA_URL, show_default=True, help="Ollama server URL."
)
@click.option(
    "--out", type=click.Path(path_type=Path), default=None, help="Write judged JSON report to file."
)
@click.option(
    "--grade-sheet",
    type=click.Path(path_type=Path),
    default=None,
    help="Also export a CSV of judged answers for hand-grading (agreement study).",
)
@click.option(
    "--grade-n",
    default=20,
    show_default=True,
    help="How many answers to include in the grade sheet.",
)
@click.option(
    "--grade-seed", default=42, show_default=True, help="Seed for the blinded grade sample."
)
def judge(
    report: Path,
    judge_model: str,
    ollama_url: str,
    out: Path | None,
    grade_sheet: Path | None,
    grade_n: int,
    grade_seed: int,
) -> None:
    """Grade a generation report with a rubric-driven LLM judge.

    \b
    REPORT is a JSON report produced by `rag-eval generation --out` or
    `rag-eval bench --out`. The judge model must differ from the answer
    model — self-grading inflates scores, and the tool refuses to do it.
    The rubric lives in rag/eval/prompts/ and is versioned.

    \b
    Example:
      rag-eval generation qa.json parts.csv --model gemma3:4b --out gen.json
      rag-eval judge gen.json --judge-model qwen3:8b --grade-sheet grades.csv
    """
    if not _is_ollama_available(ollama_url):
        console.print(f"[red]Ollama not available at {ollama_url}[/red]")
        sys.exit(1)

    data = json.loads(report.read_text(encoding="utf-8"))
    raw_results = data.get("results") or (data.get("generation") or {}).get("results") or []
    if not raw_results:
        console.print(f"[red]No generation results found in {report}[/red]")
        sys.exit(1)
    gen_results = [GenerationResult.from_dict(r) for r in raw_results]

    console.print(f"[cyan]Judging {len(gen_results)} answers with {judge_model}...[/cyan]")
    try:
        judged = judge_results(gen_results, judge_model=judge_model, base_url=ollama_url)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    metrics = compute_judge_metrics(judged)
    report_builder.print_judge_metrics(metrics)

    if out:
        report_builder.save_report(
            {"type": "judge", "metrics": metrics, "results": [j.to_dict() for j in judged]},
            out,
        )
    if grade_sheet:
        n_written = export_grade_sheet(judged, grade_sheet, n=grade_n, seed=grade_seed)
        console.print(
            f"[green]✓ Grade sheet ({n_written} rows) → {grade_sheet}[/green]\n"
            f"  Judge key → [dim]{grade_sheet.with_suffix('.judge-key.csv')}[/dim]\n"
            "  Fill in the [bold]human_verdict[/bold] column, then run:\n"
            f"  [dim]rag-eval agreement {grade_sheet}[/dim]"
        )


# ---------------------------------------------------------------------------
# agreement (human/judge agreement study)
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("grade_sheet", type=click.Path(exists=True, path_type=Path))
def agreement(grade_sheet: Path) -> None:
    """Compute human/judge agreement from a hand-graded sheet.

    \b
    GRADE_SHEET is the CSV exported by `rag-eval judge --grade-sheet`,
    with the human_verdict column filled in. Reports raw agreement and
    Cohen's kappa. Publish these next to any judge score — a judge you
    haven't checked against a human is just a second opinion, not a metric.
    """
    try:
        stats = compute_agreement(grade_sheet)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if stats.get("n_graded", 0) == 0:
        console.print("[red]No rows with human_verdict filled in.[/red]")
        sys.exit(1)
    report_builder.print_agreement(stats)


if __name__ == "__main__":
    cli()
