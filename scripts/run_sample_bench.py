"""Run the benchmark on the bundled sample corpus and write committed results.

This is the source of truth for the numbers in the README. Re-run it after any
change to the pipeline so the published numbers never drift from the code:

    python scripts/run_sample_bench.py                # retrieval only
    python scripts/run_sample_bench.py --model gemma3:4b   # + generation

Retrieval needs only the embedding model (downloads once, runs on CPU).
Generation is optional and skipped automatically if Ollama is not reachable.

All data is synthetic.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import date
from pathlib import Path

from rag.corpus.chunker import chunk_corpus
from rag.corpus.loader import load_corpus
from rag.eval.generation import (
    _is_ollama_available,
    compute_generation_metrics,
    evaluate_generation,
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

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "samples" / "parts_sample.csv"
QA_OUT = ROOT / "samples" / "qa_sample.json"
REPORT_DIR = ROOT / "reports" / "sample-run"
EMBED_MODEL = "all-MiniLM-L6-v2"


def build_store(chunks, model, name="rag_bench"):
    # Chroma reuses one in-memory client across VectorStore instances, so each
    # build needs its own collection or the chunk-size runs bleed together.
    embedder = (
        MockEmbedder(dim=16) if model == "mock" else SentenceTransformerEmbedder(model_name=model)
    )
    store = VectorStore(embedder=embedder, collection_name=name)
    store.reset()
    store.add_documents(chunks)
    return store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-model", default=EMBED_MODEL, help="Embedding model, or 'mock'.")
    ap.add_argument("--model", default=None, help="Ollama model for generation (optional).")
    ap.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL (point at a LAN GPU box or Ollama cloud).",
    )
    ap.add_argument("--gen-n", type=int, default=30, help="Questions to score for generation.")
    ap.add_argument(
        "--unanswerable",
        type=int,
        default=5,
        help="Unanswerable questions to append and include in generation scoring.",
    )
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_corpus(CORPUS)
    answerable_qa = deduplicate_questions(generate_qa_dataset(docs, max_items=150, seed=42))
    qa = answerable_qa + generate_unanswerable_items(docs, n=args.unanswerable, seed=42)
    QA_OUT.write_text(json.dumps([i.to_dict() for i in qa], indent=2), encoding="utf-8")
    print(f"{len(docs)} docs, {len(qa)} QA pairs")

    # retrieval, plus a small chunk-size sweep so the comparison is real
    retrieval = {}
    for chunk_size in (128, 256, 512):
        chunks = chunk_corpus(docs, max_chars=chunk_size)
        store = build_store(chunks, args.embed_model, name=f"sweep_{chunk_size}")
        t0 = time.perf_counter()
        results = evaluate_retrieval(qa, store, k=5)
        elapsed = time.perf_counter() - t0
        m = compute_retrieval_metrics(results)
        m["chunks"] = len(chunks)
        m["query_ms_avg"] = round(1000 * elapsed / len(results), 1)
        retrieval[chunk_size] = m
        print(
            f"chunk={chunk_size:>3}  hit@1={m['hit_at_1']:.3f}  hit@5={m['hit_at_5']:.3f}  mrr={m['mrr']:.3f}"
        )

    (REPORT_DIR / "retrieval.json").write_text(
        json.dumps({"embedding_model": args.embed_model, "by_chunk_size": retrieval}, indent=2),
        encoding="utf-8",
    )

    # generation (optional)
    generation = None
    if args.model and _is_ollama_available(args.ollama_url):
        print(
            f"running generation with {args.model} at {args.ollama_url} on {args.gen_n} questions..."
        )
        chunks = chunk_corpus(docs, max_chars=512)
        store = build_store(chunks, args.embed_model, name="gen")
        t0 = time.perf_counter()
        gen_items = select_generation_items(qa, max_items=args.gen_n)
        gen_results = evaluate_generation(
            gen_items, store, model=args.model, base_url=args.ollama_url, k=3
        )
        gm = compute_generation_metrics(gen_results)
        gm["sec_per_answer_avg"] = round((time.perf_counter() - t0) / max(len(gen_results), 1), 2)
        generation = gm
        (REPORT_DIR / "generation.json").write_text(json.dumps(gm, indent=2), encoding="utf-8")
        (REPORT_DIR / "generation-details.json").write_text(
            json.dumps(
                {
                    "type": "generation",
                    "model": args.model,
                    "k": 3,
                    "metrics": gm,
                    "results": [result.to_dict() for result in gen_results],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"faithfulness avg={gm['avg_faithfulness_score']:.3f}  errors={gm['errors']}")
    elif args.model:
        print("Ollama not reachable; skipping generation.")

    write_summary(
        len(docs), len(answerable_qa), args.unanswerable, args.embed_model, retrieval, generation
    )
    print(f"wrote {REPORT_DIR / 'summary.md'}")


def write_summary(n_docs, n_answerable, n_unanswerable, embed_model, retrieval, generation) -> None:
    best = retrieval[512]
    lines = [
        "# Sample run results",
        "",
        f"_Generated by `scripts/run_sample_bench.py` on {date.today().isoformat()} "
        f"({platform.system()} {platform.machine()}). All data is synthetic._",
        "",
        f"Corpus: `samples/parts_sample.csv` ({n_docs} parts). QA set: {n_answerable} "
        f"answerable template questions plus {n_unanswerable} unanswerable questions "
        "(`samples/qa_sample.json`, seed 42).",
        f"Embedding model: `{embed_model}`.",
        "",
        "## Retrieval by chunk size",
        "",
        "| Chunk chars | Passages | Hit@1 | Hit@5 | MRR | Avg query |",
        "|------------:|---------:|------:|------:|----:|----------:|",
    ]
    for cs in (128, 256, 512):
        m = retrieval[cs]
        lines.append(
            f"| {cs} | {m['chunks']} | {m['hit_at_1']:.3f} | {m['hit_at_5']:.3f} | "
            f"{m['mrr']:.3f} | {m['query_ms_avg']} ms |"
        )
    lines += [
        "",
        f"Default config (512 chars): Hit@1 {best['hit_at_1']:.3f}, "
        f"Hit@5 {best['hit_at_5']:.3f}, MRR {best['mrr']:.3f}.",
    ]
    if generation:
        lines += [
            "",
            "## Generation",
            "",
            f"Model `{generation['model']}` over {generation['n_questions']} questions, "
            "top-3 retrieved chunks as context.",
            "",
            f"- Faithfulness (keyword-overlap proxy): **{generation['avg_faithfulness_score']:.3f}** "
            f"(min {generation['min_faithfulness_score']:.3f}, max {generation['max_faithfulness_score']:.3f})",
            f"- Errors: {generation['errors']}",
            f"- Correct-refusal rate: {generation.get('correct_refusal_rate', 0):.3f} "
            f"over {generation.get('n_unanswerable', 0)} unanswerable questions",
            f"- False-refusal rate: {generation.get('false_refusal_rate', 0):.3f} "
            f"over {generation.get('n_answerable', 0)} answerable questions",
            f"- Speed: {generation['sec_per_answer_avg']} s/answer",
        ]
    (REPORT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
