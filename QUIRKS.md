# QUIRKS — Known Challenges in rag-eval-bench

> **All data is synthetic.** All part numbers, supplier names, and product descriptions are
> fabricated. No real company, product, or supplier data is used anywhere in this project.

This document catalogs the intentional rough edges and design choices that make
`rag-eval-bench` a realistic test harness rather than a toy example.

---

## 1. Corpus Heterogeneity

The bench accepts three corpus formats from `acme-parts-cloud` exports:

| Format | Source | Key columns |
|--------|--------|-------------|
| Parts v1 CSV | Legacy export | `pn`, `desc`, `rev`, `state` |
| Parts v2 CSV | Current export | `part_number`, `description`, `revision`, `lifecycle_state` |
| JSONL | Streaming API export | `id`, `part_number`, `description` |

`loader.py` auto-detects format by peeking at the CSV header row. A corpus built from
multiple export formats will have mixed metadata richness — some chunks will carry
revision and state fields; others will not. Retrieval metrics are computed purely on
text similarity, so these gaps don't break scoring, but generation quality varies.

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

When running with the real `SentenceTransformerEmbedder` (`all-MiniLM-L6-v2`), expect
Hit@1 ≈ 60-80% on template-generated QA pairs (questions are derived from description
text, so retrieval should be strong).

---

## 7. Ollama Generation — Graceful Degradation

`generation.py` calls `GET /api/tags` to probe Ollama before sending any generation
request. If Ollama is not running, the generation command exits with a warning rather
than an error. The `bench` command auto-skips the generation phase when Ollama is
unavailable and still produces retrieval metrics.

Default model: `gemma:2b`. Override with `--model` flag. The bench does not validate
that the requested model is actually pulled — if the model is missing, Ollama will
return an error that surfaces as a `GenerationResult` with `answer=None`.

---

## 8. Faithfulness Scoring Is a Proxy

`_faithfulness_score()` computes keyword overlap between the generated answer and the
retrieved context — not semantic entailment. A fluent-sounding answer with no shared
keywords scores 0.0. A terse answer that repeats exact keywords from context scores 1.0.

This is intentionally a lightweight proxy, not a full NLI faithfulness check. It is
sufficient to distinguish "Ollama is hallucinating" from "Ollama is grounding answers in
context" at a population level.

---

## 9. QA Deduplication Is Whitespace/Case Only

`deduplicate_questions()` normalizes whitespace and lowercases before comparing.
Semantically duplicate questions with different wording (e.g. "What is the revision of
PN-0042?" vs "What revision does PN-0042 have?") are both kept. Expect some semantic
duplicates in large QA datasets generated from small corpora.

---

## 10. No Negative / Adversarial QA

All generated QA pairs have a ground-truth answer in the corpus. There are no
unanswerable questions (questions whose answer is not present). Precision at recall
curves will be optimistic compared to production RAG deployments that must handle
out-of-scope queries.
