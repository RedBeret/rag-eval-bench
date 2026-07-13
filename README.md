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

<p align="center"><img src="demo/demo.gif" alt="rag-eval-bench measuring a RAG config on synthetic data" width="820"></p>

## Quick start

```bash
git clone https://github.com/RedBeret/rag-eval-bench.git
cd rag-eval-bench
python -m venv .venv && source .venv/bin/activate
make setup

# verify the whole pipeline with no model download (deterministic MockEmbedder)
make smoke

# real retrieval run: embeds the corpus and sweeps chunk sizes
make sample-bench

# optional real generation run, including five unanswerable questions
make sample-generation GEN_MODEL=gemma3:4b
```

`make smoke` proves the plumbing works; it does not measure model quality.
`make sample-bench` downloads the embedding model once (~90 MB) and then runs entirely
offline. On Windows, `run.bat` sets up the venv for you.

## Results

Every number below is copied from a committed machine-readable artifact. Run
`make sample-bench` for retrieval, `make sample-generation` for generation, and
`make sample-judge JUDGE_MODEL=phi4:latest` for the separate-model judge.

Corpus: 90 synthetic parts, 150 answerable template questions plus 5 unanswerable
questions (seed 42). Embedder `all-MiniLM-L6-v2`.

| Chunk chars | Passages | Hit@1 | Hit@5 | MRR | Avg query |
|------------:|---------:|------:|------:|----:|----------:|
| 128 | 100 | 0.460 | 0.680 | 0.537 | 12.0 ms |
| 256 |  90 | 0.473 | 0.727 | 0.563 | 10.2 ms |
| 512 |  90 | 0.473 | 0.727 | 0.563 | 10.7 ms |

The mildly interesting result: shrinking chunks to 128 chars made retrieval *worse*
(Hit@5 0.68 vs 0.73). Most part descriptions are short, so 256 and 512 never split and
score identically; 128 fragments the longer descriptions and scatters the signal across
chunks. "Smaller chunks = better retrieval" is not a law — it depends on your documents,
which is the whole reason to measure it.

Generation with local `gemma3:4b` over 30 questions (25 answerable, 5 unanswerable;
top-3 chunks as context): keyword-overlap faithfulness **0.760**, 0 errors, **1.000**
correct-refusal rate, **0.240** false-refusal rate, and 1.95 s/answer. The high false
refusal rate is useful evidence that retrieval misses still matter even when the model
is instructed not to hallucinate. See the committed
[`generation.json`](reports/sample-run/generation.json) and answer-level
[`generation-details.json`](reports/sample-run/generation-details.json).

A separate local `phi4:latest` judge using rubric v1 marked **0.800 correct** and
**0.800 grounded** with 0 judge errors. That score is committed as raw evidence, but it
is not yet a validated metric: the seeded 20-row human grade sheet is intentionally
blank, so agreement and Cohen's kappa remain pending human review.
The raw verdicts and reasoning are in
[`judged.json`](reports/sample-run/judged.json).

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
| `rag-eval judge REPORT --judge-model qwen3:8b` | rubric-driven LLM judge over a generation report |
| `rag-eval agreement grades.csv` | human/judge agreement (raw % + Cohen's kappa) |

`qa-gen --unanswerable N` (and `bench`, 5 by default) adds questions the corpus
provably cannot answer — refusing correctly is scored, answering them is a
hallucination.

`make sample-generation` writes both aggregate `generation.json` metrics and an
answer-level `generation-details.json`. `make sample-judge` consumes the detailed
artifact. The committed judge run also includes a blinded grade sheet and a separate
answer key; fill only `human_verdict` before running `rag-eval agreement`.

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
- **Correct-refusal rate** — on questions the corpus provably can't answer, how often
  the model refuses instead of hallucinating. Its mirror, **false-refusal rate**,
  counts refusals on answerable questions — a system that refuses everything is
  useless, not safe.
- **Errors** — generation calls that failed (model not pulled, timeout, etc.).

**Judge** (needs Ollama, second model):

- `rag-eval judge` grades each answer **correct / partial / incorrect** plus a
  **grounded** flag, using a rubric versioned in-repo
  (`rag/eval/prompts/judge_v1.md`) so you can audit what "correct" means before
  trusting a number. The judge model must differ from the answer model — the tool
  refuses to let a model grade its own homework.
- Judge scores can be checked with `--grade-sheet`, which exports a seeded sample with
  verdicts hidden in a separate companion key. After a human fills the sheet,
  `rag-eval agreement` reports agreement plus Cohen's kappa. The committed sheet is
  blank, so the current judge score remains an opinion rather than a validated metric.

## Bring your own corpus

Swap the bundled data for your own. Two supported shapes:

- **CSV** with `part_number,name,category,uom,status` (parts) or
  `supplier_code,name,country,contact_email,active` (suppliers).
- **JSONL**, one object per line: `{"doc_id": "...", "text": "...", "metadata": {...}}`.

Then run `rag-eval qa-gen yourdata.csv --out qa.json` and
`rag-eval retrieval qa.json yourdata.csv`. For anything that isn't a parts/supplier table,
JSONL is the general path.

## Running bigger models

Retrieval only needs CPU. Generation talks to any Ollama endpoint, so there are three
ways to use a larger model without touching the code:

**1. Local GPU box** — pull a model and run it on the same machine:

```bash
ollama pull gpt-oss:20b
make sample-generation GEN_MODEL=gpt-oss:20b
```

**2. A GPU box over your LAN** — bind Ollama to the network on the GPU machine:

```bash
OLLAMA_HOST=0.0.0.0 ollama serve
```

then point the run at it from anywhere on the LAN:

```bash
make sample-generation GEN_MODEL=gpt-oss:20b OLLAMA_URL=http://<gpu-box-ip>:11434
```

Keep that endpoint on a trusted network — Ollama has no auth.

**3. Ollama cloud models** — sign in once (`ollama signin`) and use a `-cloud` tag:

```bash
make sample-generation GEN_MODEL=gpt-oss:120b-cloud
```

Any other OpenAI-compatible endpoint would need a small adapter in
`rag/eval/generation.py` — left out on purpose to keep the default path local and
verifiable. Results land in `reports/sample-run/` and the committed numbers regenerate.

## What this is, and isn't

It is a teaching-sized, honest measurement harness: readable code, reproducible numbers,
one concern per file. It is deliberately **not** a RAG framework or a LangChain
replacement.

Known limits, all documented in `QUIRKS.md`: the template questions are easy by
construction (the answer sits in the source row), so retrieval numbers here are optimistic
versus production; refusal detection is pattern-based, not a classifier; and the LLM
judge is only as trustworthy as its published human-agreement number — which is why the
tooling to measure that agreement ships in the box. Those are the honest edges, not bugs.

## Related projects

- [`acme-parts-cloud`](https://github.com/RedBeret/acme-parts-cloud) — synthetic enterprise data and messy exports (the corpus source)
- [`fde-data-forge`](https://github.com/RedBeret/fde-data-forge) — clean and reconcile those messy exports
- [`air-gap-deploy-kit`](https://github.com/RedBeret/air-gap-deploy-kit) — package any of these for offline / air-gapped install

## License

MIT, © 2026 Steven Espinoza. See [LICENSE](LICENSE).
