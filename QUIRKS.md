# QUIRKS — Known Challenges in rag-eval-bench

> **All data is synthetic.** All part numbers, supplier names, and product descriptions are
> fabricated. No real company, product, or supplier data is used anywhere in this project.

This document catalogs the intentional rough edges and design choices that make
`rag-eval-bench` a realistic test harness rather than a toy example.

---

## 1. Corpus Heterogeneity

The bench accepts two table schemas plus a general JSONL shape:

| Format | Source | Key columns |
|--------|--------|-------------|
| Parts CSV | Parts export | `part_number`, `name`, `category`, `uom`, `status` |
| Suppliers CSV | Supplier export | `supplier_code`, `name`, `country`, `contact_email`, `active` |
| JSONL | General documents | `doc_id`, `text`, optional `metadata` |

`loader.py` distinguishes the two CSV shapes by header and treats any other CSV as a
parts table. Unsupported legacy column names may load as blank fields, so use JSONL for
arbitrary schemas rather than assuming loose CSV aliases are normalized.

---

## 2. Chunk ID Suffix Convention

`chunker.py` appends `#chunk{N}` to every parent doc ID:

```
PN-0042 → PN-0042#chunk0, PN-0042#chunk1, ...
```

`retrieval.py` strips this suffix before matching, so a query that returns chunk 3 of
PN-0042 still counts as a hit for the question "what is part PN-0042?"

**Gotcha:** if a doc ID naturally contains `#`, the suffix stripping will misfire. The
corpus loaders never emit `#` in IDs, but external JSONL files should be validated.

---

## 3. Part Number Era Collisions

Part numbers from `acme-parts-cloud` span three naming eras:

- **Current:** `PN-{N:04d}` (zero-padded, e.g. `PN-0042`)
- **2019 migration:** `2019-PN-{N}` (no zero-padding, e.g. `2019-PN-42`)
- **Legacy:** `P{N}` (no prefix, e.g. `P42`)

All three refer to different physical parts (not the same part renamed). The QA generator
uses the exact part number from the corpus row, so questions will contain all three
formats. A retriever that strips prefixes or normalizes during indexing may return the
wrong part for cross-era queries.

---

## 4. Description Length Variance

Part descriptions range from 3-word abbreviated labels (`"HEX BOLT GR8"`) to
multi-sentence engineering notes (`"Stainless steel hex-head bolt, grade 8, 1/2-13 UNC
thread, 2-inch length, zinc dichromate plated for corrosion resistance, torque spec 75 ft-lb"`).

Short descriptions produce very short chunks. The default `chunk_by_sentences()` splitter
uses `". "` as a boundary — descriptions with no period stay as a single chunk regardless
of length. Metric variance between short- and long-description parts is expected and
intentional.

---

## 5. Supplier Near-Duplicate Embeddings

`acme-parts-cloud` intentionally injects supplier name variants:

- `"Vortex Metals"` / `"VORTEX METALS"` / `"Vortex Metals Inc."`

If supplier rows are included in the corpus, embeddings for these near-duplicates will
cluster tightly. A question about `"Vortex Metals"` may retrieve chunks from any variant.
This is realistic supplier dedup noise, not an error in the bench.

---

## 6. MockEmbedder vs Real Embedder Metric Gap

CI runs use `MockEmbedder` (deterministic hash-based vectors, no model download).
`MockEmbedder` produces random-quality embeddings — Hit@1 will be near-random (≈10%
for 10 docs). This is intentional: CI tests correctness of metric computation, not
embedding quality.

The committed `all-MiniLM-L6-v2` sample scores Hit@1 **47.3%** and Hit@5 **72.7%** at
256/512-character chunks. Treat those artifacts as the baseline instead of assuming a
60-80% Hit@1 range.

---

## 7. Ollama Generation — Graceful Degradation

`generation.py` calls `GET /api/tags` to probe Ollama before sending any generation
request. If Ollama is not running, the standalone generation command exits nonzero;
`bench` auto-skips generation and still produces retrieval metrics.

Default model: `gemma3:4b`. Override it with `--model`. The bench does not validate that
the requested model is pulled; a missing model is captured as a `GenerationResult`
whose generated answer starts with `[ERROR:` and is counted in error metrics.

---

## 8. Faithfulness Scoring Is a Proxy

`_faithfulness_score()` computes keyword overlap between the generated answer and the
retrieved context — not semantic entailment. A fluent-sounding answer with no shared
keywords scores 0.0. A terse answer that repeats exact keywords from context scores 1.0.

This is intentionally a lightweight proxy, not a full NLI faithfulness check. It is a
directional lexical-overlap signal and cannot by itself prove factual grounding or
distinguish every hallucination.

For per-answer verdicts, use `rag-eval judge` — a rubric-driven LLM judge with a
versioned prompt (`rag/eval/prompts/judge_v1.md`). The judge model must differ from the
answer model (the tool refuses self-grading). Judge scores should themselves be
validated: `rag-eval judge --grade-sheet` exports a seeded, blinded CSV plus a separate
judge key, and `rag-eval agreement` reports human/judge agreement plus Cohen's kappa.
**A judge score published without an agreement number is an opinion, not a metric.**

---

## 9. QA Deduplication Is Whitespace/Case Only

`deduplicate_questions()` normalizes whitespace and lowercases before comparing.
Semantically duplicate questions with different wording (e.g. "What is the revision of
PN-0042?" vs "What revision does PN-0042 have?") are both kept. Expect some semantic
duplicates in large QA datasets generated from small corpora.

---

## 10. Unanswerable QA — Refusal Detection Is Pattern-Based

`qa-gen --unanswerable N` (and `bench`, 5 by default) generates questions about part
numbers and suppliers that provably do not exist in the corpus. The correct behavior is
a refusal; answering anyway is a hallucination. The generation prompt instructs the
model to reply with a fixed refusal sentence, and `is_refusal()` matches a broad set of
refusal phrasings — but it is a regex, not a classifier. A model that refuses in a
sufficiently creative way will be miscounted as answering. Both rates are published:
`correct_refusal_rate` (unanswerable questions refused — higher is better) and
`false_refusal_rate` (answerable questions refused — lower is better; a system that
refuses everything is useless, not safe).

Generation errors remain explicit and count against correct-refusal success. They are
not silently removed from the negative-question denominator.

Unanswerable items are excluded from retrieval metrics: with no correct document to
retrieve, Hit@k over them is noise.
