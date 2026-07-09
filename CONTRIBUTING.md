# Contributing to rag-eval-bench

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
pip install ruff pytest pytest-httpx
```

## Run Tests

```bash
pytest -v
```

CI tests use `MockEmbedder` — no model downloads required.

## Lint

```bash
ruff check .
ruff format .
```

All PRs must pass `ruff check` with zero errors.

## Layout

```
rag/
  corpus/     — loaders and chunkers
  index/      — embedder + ChromaDB wrapper
  qa/         — QA dataclasses and template generator
  eval/       — retrieval and generation metrics
  report/     — rich terminal output and JSON save
  cli.py      — click entry point (rag-eval command)
tests/
  test_corpus.py
  test_qa.py
  test_eval.py
```

## Ollama (Optional)

Generation scoring requires Ollama running locally:

```bash
ollama serve
ollama pull gemma:2b
```

The bench degrades gracefully without Ollama — retrieval metrics still run.

## PR Workflow

1. Branch off `main` (no direct pushes to main)
2. `pytest -v` + `ruff check .` must pass locally before opening PR
3. CI runs lint + tests — wait for green before merging

## Synthetic Data Only

All corpus data used in tests and examples must be synthetic. No real company, product,
or supplier data. Add "All data is synthetic" to any example README or demo script.
