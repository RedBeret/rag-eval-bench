# rag-eval-bench

[![CI](https://github.com/RedBeret/rag-eval-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/RedBeret/rag-eval-bench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A small, local-first harness for **measuring** a RAG pipeline instead of eyeballing it.
Point it at a corpus, generate a ground-truth question set, and get retrieval numbers
(Hit@k, MRR) plus optional answer scoring from a local Ollama model. No API keys, no
hosted services, runs offline.

The bundled corpus is a synthetic parts catalog from the same fictional company I use
across my other projects, so you can clone and run it in one step without standing up
anything else.

> **All data is synthetic.** Part numbers, supplier names, and descriptions are
> fabricated. Meridian Fabrication Co. is not a real company.

## Quick start

```bash
git clone https://github.com/RedBeret/rag-eval-bench.git
cd rag-eval-bench
python -m venv .venv && source .venv/bin/activate
make setup

# verify the whole pipeline with no model download (deterministic MockEmbedder)
make smoke

# real run: embeds the bundled corpus with all-MiniLM-L6-v2 and sweeps chunk sizes
make sample-bench
```

`make smoke` proves the plumbing works; it does not measure model quality.
`make sample-bench` downloads the embedding model once (~90 MB) and then runs entirely
offline. On Windows, `run.bat` sets up the venv for you.

## Results

Every number below is produced by `scripts/run_sample_bench.py` on the bundled corpus.
Re-run `make sample-bench` and it regenerates `reports/sample-run/summary.md` — nothing
here is hand-typed.

Corpus: 90 synthetic parts, 150 template questions (seed 42). Embedder `all-MiniLM-L6-v2`.

| Chunk chars | Passages | Hit@1 | Hit@5 | MRR | Avg query |
|------------:|---------:|------:|------:|----:|----------:|
| 128 | 100 | 0.460 | 0.680 | 0.537 | 10.2 ms |
| 256 |  90 | 0.473 | 0.727 | 0.563 |  9.9 ms |
| 512 |  90 | 0.473 | 0.727 | 0.563 | 10.6 ms |

The mildly interesting result: shrinking chunks to 128 chars made retrieval *worse*
(Hit@5 0.68 vs 0.73). Most part descriptions are short, so 256 and 512 never split and
score identically; 128 fragments the longer descriptions and scatters the signal across
chunks. "Smaller chunks = better retrieval" is not a law — it depends on your documents,
which is the whole reason to measure it.

Generation with `gemma4:e4b` over 30 questions (top-3 chunks as context): faithfulness
0.70 (keyword-overlap proxy), 0 errors, ~6.1 s/answer on an M-series laptop. See
[Metrics](#metrics) for what that score does and does not mean.

## How it works

Six small modules, one job each:

```mermaid
flowchart LR
    A[corpus<br>CSV / JSONL] --> B[chunk]
    B --> C[embed<br>MiniLM or Mock]
    C --> D[(ChromaDB<br>index)]
    A --> Q[qa-gen<br>templates + seed]
    D --> R[retrieval<br>Hit@k · MRR]
    Q --> R
    D --> G[generation<br>Ollama, optional]
    Q --> G
    R --> S[report<br>terminal + JSON]
    G --> S
```

QA pairs come from fixed templates with a fixed seed, so ground truth is reproducible and
needs no LLM. Retrieval is scored at the parent-document level — a chunk hit counts for
its parent doc (`QUIRKS.md §2`). Generation, when enabled, hands the retrieved context to
a local Ollama model and scores the answer.

## CLI

The Makefile wraps the common paths; underneath it is the `rag-eval` command.

| Command | What it does |
|---------|--------------|
| `rag-eval index CORPUS --persist-dir DIR` | embed a corpus and write a ChromaDB index |
| `rag-eval qa-gen CORPUS --out qa.json` | generate template QA pairs from a corpus |
| `rag-eval retrieval QA CORPUS --k 5` | Hit@1/3/5 and MRR against an index |
| `rag-eval generation QA CORPUS --model gemma3:4b` | score local Ollama answers |
| `rag-eval bench CORPUS` | index → qa-gen → retrieve → (generation if Ollama is up) |

`--embed-model mock` swaps in the deterministic MockEmbedder (no download);
`--skip-generation` forces retrieval-only even when Ollama is running. Run any command
with `--help` for the full option list.

## Metrics

**Retrieval** (no LLM, objective):

- **Hit@k** — fraction of questions whose source document is in the top *k* results.
- **MRR** — mean reciprocal rank of the source document.

**Generation** (needs Ollama):

- **Faithfulness** — fraction of answer keywords that also appear in the retrieved
  context. It is a cheap proxy for "is the model grounding its answer in what it
  retrieved," not an entailment/NLI check. A fluent answer that invents facts scores low;
  a terse answer that echoes the context scores high. `QUIRKS.md §8` covers the limits.
- **Errors** — generation calls that failed (model not pulled, timeout, etc.).

## Bring your own corpus

Swap the bundled data for your own. Two supported shapes:

- **CSV** with `part_number,name,category,uom,status` (parts) or
  `supplier_code,name,country,contact_email,active` (suppliers).
- **JSONL**, one object per line: `{"doc_id": "...", "text": "...", "metadata": {...}}`.

Then run `rag-eval qa-gen yourdata.csv --out qa.json` and
`rag-eval retrieval qa.json yourdata.csv`. For anything that isn't a parts/supplier table,
JSONL is the general path.

## Running on a GPU box

Retrieval only needs CPU. For generation and larger sweeps, use a machine with a real GPU:

```bash
make setup
ollama serve
ollama pull gemma4:e4b          # or any local model you like
make sample-generation GEN_MODEL=gemma4:e4b
```

Results land in `reports/sample-run/`. The committed run in this repo was produced with
`gemma4:e4b`; point `GEN_MODEL` at a bigger model and re-run to compare.

## What this is, and isn't

It is a teaching-sized, honest measurement harness: readable code, reproducible numbers,
one concern per file. It is deliberately **not** a RAG framework or a LangChain
replacement.

Known limits, all documented in `QUIRKS.md`: the template questions are easy by
construction (the answer sits in the source row), so retrieval numbers here are optimistic
versus production; there are no unanswerable or adversarial questions yet; faithfulness is
a keyword proxy, not a judge model. Those are the honest edges, not bugs.

## Related projects

- [`acme-parts-cloud`](https://github.com/RedBeret/acme-parts-cloud) — synthetic enterprise data and messy exports (the corpus source)
- [`fde-data-forge`](https://github.com/RedBeret/fde-data-forge) — clean and reconcile those messy exports
- [`air-gap-deploy-kit`](https://github.com/RedBeret/air-gap-deploy-kit) — package any of these for offline / air-gapped install

## License

MIT, © 2026 Steven Espinoza. See [LICENSE](LICENSE).
