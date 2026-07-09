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
from rag.eval.retrieval import compute_retrieval_metrics, evaluate_retrieval
from rag.index.embedder import SentenceTransformerEmbedder
from rag.index.store import VectorStore
from rag.qa.generator import deduplicate_questions, generate_qa_dataset
from rag.qa.types import QAItem
from rag.report import builder as report_builder

console = Console()


def _load_qa(path: Path) -> list[QAItem]:
    items = json.loads(path.read_text(encoding="utf-8"))
    return [QAItem.from_dict(i) for i in items]


def _save_qa(items: list[QAItem], path: Path) -> None:
    path.write_text(json.dumps([i.to_dict() for i in items], indent=2), encoding="utf-8")


@click.group()
@click.version_option("1.0.0", prog_name="rag-eval")
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

    embedder = SentenceTransformerEmbedder(model_name=model)
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
def qa_gen(corpus: Path, out: Path, max_items: int, seed: int) -> None:
    """Generate ground-truth QA pairs from a corpus file.

    \b
    Example:
      rag-eval qa-gen parts_v1.csv --out qa.json --max-items 100
    """
    docs = load_corpus(corpus)
    console.print(f"[cyan]Loaded {len(docs):,} documents[/cyan]")

    items = generate_qa_dataset(docs, max_items=max_items, seed=seed)
    items = deduplicate_questions(items)
    console.print(f"[cyan]Generated {len(items):,} QA pairs (deduplicated)[/cyan]")

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
    embedder = SentenceTransformerEmbedder(model_name=model)
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
    help="Ollama model name (e.g. gemma:2b, mistral).",
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
      ollama pull gemma:2b
      ollama serve

    \b
    Example:
      rag-eval generation qa.json parts_v1.csv --model gemma:2b
    """
    if not _is_ollama_available(ollama_url):
        console.print(f"[red]Ollama not available at {ollama_url}[/red]")
        console.print("  Start Ollama: [dim]ollama serve[/dim]")
        console.print(f"  Pull model:   [dim]ollama pull {model}[/dim]")
        sys.exit(1)

    console.print(f"[green]✓ Ollama available — model: {model}[/green]")

    qa_items = _load_qa(qa)[:max_items]
    docs = load_corpus(corpus)
    chunks = chunk_corpus(docs)
    embedder = SentenceTransformerEmbedder()
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
@click.option(
    "--out", type=click.Path(path_type=Path), default=None, help="Write full JSON report to file."
)
def bench(
    corpus: Path, model: str, embed_model: str, k: int, max_qa: int, seed: int, out: Path | None
) -> None:
    """Run the full benchmark pipeline: index → qa-gen → retrieval → [generation].

    \b
    Generation is skipped automatically if Ollama is not available.

    \b
    Example:
      rag-eval bench parts_v1.csv --model gemma:2b --out bench_report.json
    """
    # Load and index
    docs = load_corpus(corpus)
    chunks = chunk_corpus(docs)
    embedder = SentenceTransformerEmbedder(model_name=embed_model)
    store = VectorStore(embedder=embedder)
    store.add_documents(chunks)
    console.print(
        f"[green]✓ Indexed {store.count():,} passages from {len(docs):,} documents[/green]"
    )

    # QA generation
    qa_items = generate_qa_dataset(docs, max_items=max_qa, seed=seed)
    qa_items = deduplicate_questions(qa_items)
    console.print(f"[green]✓ Generated {len(qa_items):,} QA pairs[/green]")

    # Retrieval eval
    retr_results = evaluate_retrieval(qa_items, store, k=k)
    retr_metrics = compute_retrieval_metrics(retr_results)
    report_builder.print_retrieval_metrics(retr_metrics)

    # Generation eval (optional)
    gen_metrics: dict = {}
    gen_results = []
    if _is_ollama_available():
        console.print(f"[cyan]Ollama available — running generation eval with {model}...[/cyan]")
        gen_results = evaluate_generation(qa_items[:50], store, model=model, k=k)
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


if __name__ == "__main__":
    cli()
