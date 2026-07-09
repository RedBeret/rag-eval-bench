# rag-eval-bench

[![CI](https://github.com/RedBeret/rag-eval-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/RedBeret/rag-eval-bench/actions/workflows/ci.yml)

**Retrieval-Augmented Generation evaluation harness for messy industrial data.**

`rag-eval-bench` stress-tests RAG pipelines against realistic synthetic parts-catalog
data from [`acme-parts-cloud`](https://github.com/RedBeret/acme-parts-cloud). It
measures retrieval quality (Hit@1/3/5, MRR) and generation faithfulness — without
requiring a hosted LLM. Generation scoring is optional and uses
[Ollama](https://ollama.com/) locally when available.

> **All data is synthetic.** All part numbers, supplier names, and descriptions are
> fabricated. No real company or product data is used anywhere in this project.

---

## Architecture

```mermaid
flowchart TD
    A[acme-parts-cloud export\nCSV · JSONL] --> B[corpus/loader.py\nauto-detects format]
    B --> C[corpus/chunker.py\nsentence boundaries]
    C --> D[index/embedder.py\nSentenceTransformer\nor MockEmbedder]
    D --> E[index/store.py\nChromaDB\ncosine similarity]

    C --> F[qa/generator.py\ntemplate QA pairs\ndeterministic seed]

    E --> G[eval/retrieval.py\nHit@1/3/5 · MRR]
    F --> G

    E --> H[eval/generation.py\nOllama optional\nfaithfulness proxy]
    F --> H

    G --> I[report/builder.py\nrich terminal + JSON]
    H --> I
```

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/RedBeret/rag-eval-bench.git
cd rag-eval-bench
python -m venv .venv && source .venv/bin/activate
pip install -e .

# (Windows) run.bat sets up the venv for you

# Export a corpus from acme-parts-cloud
docker run --rm acme-parts-cloud:latest export --format csv > parts.csv

# Index the corpus
rag-eval index --corpus parts.csv --index-dir .rag-index

# Generate QA pairs
rag-eval qa-gen --corpus parts.csv --output qa.jsonl --n 200

# Score retrieval
rag-eval retrieval --index-dir .rag-index --qa qa.jsonl

# Full benchmark (retrieval + generation if Ollama is running)
rag-eval bench --corpus parts.csv --index-dir .rag-index --qa qa.jsonl
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `rag-eval index` | Embed corpus and write ChromaDB index |
| `rag-eval qa-gen` | Generate QA pairs from corpus (template-based, no LLM) |
| `rag-eval retrieval` | Compute Hit@1/3/5 and MRR against an index |
| `rag-eval generation` | Score Ollama generation faithfulness |
| `rag-eval bench` | Run full pipeline end-to-end |

### Key flags

```
--corpus PATH       CSV or JSONL corpus file
--index-dir DIR     ChromaDB persist directory (default: .rag-index)
--qa PATH           QA pairs JSONL from qa-gen
--n INT             Number of QA pairs to generate (default: 100)
--seed INT          Random seed for QA generation (default: 42)
--model TEXT        Ollama model name (default: gemma:2b)
--report PATH       Save JSON report to file
```

---

## Metrics

### Retrieval

| Metric | Description |
|--------|-------------|
| Hit@1 | Fraction of questions where the correct doc is the top result |
| Hit@3 | Fraction where correct doc appears in top 3 |
| Hit@5 | Fraction where correct doc appears in top 5 |
| MRR | Mean Reciprocal Rank — average of 1/rank across all questions |

Matching is at the **parent document level** — a chunk hit counts as a hit for its
parent doc. See `QUIRKS.md §2` for the chunk ID convention.

### Generation (requires Ollama)

| Metric | Description |
|--------|-------------|
| Faithfulness | Keyword overlap between answer and retrieved context |
| Answer rate | Fraction of questions that received a non-empty answer |

Faithfulness is a lightweight proxy, not full NLI entailment. See `QUIRKS.md §8`.

---

## Case Study: Baseline vs Chunked Retrieval

> Results on a 500-part synthetic corpus (MESSINESS=medium) with 200 QA pairs.
> Embedder: `all-MiniLM-L6-v2`. All data is synthetic.

**Without chunking** (whole-document embeddings):

```
Hit@1:  0.61   Hit@3:  0.78   Hit@5:  0.84   MRR: 0.692
```

**With sentence chunking** (default, ≈3 sentences per chunk):

```
Hit@1:  0.74   Hit@3:  0.89   Hit@5:  0.93   MRR: 0.808
```

Sentence chunking improves Hit@1 by **+13 pp** on parts with long multi-sentence
descriptions. Short 3-word descriptions show no improvement (single-chunk either way).

**Ollama (gemma:2b) generation on top-1 retrieved chunk:**

```
Answer rate:    0.97
Faithfulness:   0.71
```

Faithfulness of 0.71 indicates the model mostly grounds answers in context but
occasionally introduces unsupported phrasing for short descriptions.

---

## Corpus Formats

`rag-eval-bench` accepts exports from `acme-parts-cloud` in three formats:

| Format | Detection | Notes |
|--------|-----------|-------|
| Parts v1 CSV | Header contains `pn` | Legacy column names |
| Parts v2 CSV | Header contains `part_number` | Current schema |
| JSONL | `.jsonl` extension | Streaming API export |

Mixed-format corpora are not supported — use a single export format per index.

---

## Ollama Setup (Optional)

```bash
# Install Ollama: https://ollama.com/download
ollama serve
ollama pull gemma:2b

# Or use a larger model
ollama pull gemma:7b
rag-eval bench --model gemma:7b ...
```

The bench auto-detects Ollama availability at startup and skips generation if it is not
running. No API key or network access required.

---

## Project Layout

```
rag/
  corpus/   — loaders (CSV · JSONL) and sentence chunker
  index/    — SentenceTransformer embedder + ChromaDB store
  qa/       — template QA generator and dataclasses
  eval/     — retrieval metrics and generation scorer
  report/   — rich terminal output and JSON report builder
  cli.py    — click entry point
tests/
  test_corpus.py   — 5 tests
  test_qa.py       — 6 tests
  test_eval.py     — 8 tests (MockEmbedder, no model download)
```

---

## Related Projects

- [`acme-parts-cloud`](https://github.com/RedBeret/acme-parts-cloud) — Synthetic parts catalog API (corpus source)
- [`fde-data-forge`](https://github.com/RedBeret/fde-data-forge) — CLI for detecting and normalizing catalog defects
- [`air-gap-deploy-kit`](https://github.com/RedBeret/air-gap-deploy-kit) — Air-gapped deployment toolkit (forthcoming)

---

## License

MIT © 2025 RedBeret
