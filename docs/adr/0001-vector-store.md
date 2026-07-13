# ADR 0001: ChromaDB as the vector store

**Status:** Accepted (documented retroactively — the decision was made in code
during v0; this records the reasoning so it isn't lost)
**Date:** 2026-07-12

## Context

The harness needs a local vector store with zero external services. Two
candidates fit the local-first ethos:

- **sqlite-vec** — a SQLite extension. Lighter, zero daemon, single file,
  arguably a purer fit for "no services, no accounts."
- **ChromaDB** — embedded mode, pure `pip install`, no daemon either, but a
  heavier dependency tree.

## Decision

ChromaDB, embedded mode.

## Reasoning

1. **Audience recognition.** This repo is a template people fork to evaluate
   *their* RAG stack, and ChromaDB is what a large share of RAG tutorials and
   production prototypes already use. Benchmarking on the store people
   actually run beats benchmarking on a purer but less common one.
2. **API stability for the swap path.** Chroma's collection API made the
   `VectorStore` wrapper trivial (~1 file), keeping "one concern per file"
   intact. sqlite-vec would have needed hand-rolled top-k and metadata
   filtering — more code to read before trusting the harness.
3. **In-memory and persistent modes share one code path**, which is what lets
   `make smoke` run in CI with the MockEmbedder and no filesystem state.

## Consequences

- Heavier install than sqlite-vec (accepted; `make setup` is still one
  command).
- The `VectorStore` wrapper is the seam: if sqlite-vec (or anything else)
  becomes worth comparing, it enters as a second implementation of the same
  interface — and becomes a benchmark axis, which is very much in the spirit
  of this repo.

## Revisit when

- A published run needs a store-vs-store comparison axis, or
- Chroma's dependency tree starts breaking the offline/air-gapped install
  story (see `air-gap-deploy-kit`).
