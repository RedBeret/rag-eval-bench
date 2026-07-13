"""Retrieval evaluation metrics: Hit@K, MRR, Precision@K."""

from __future__ import annotations

from rag.index.store import VectorStore
from rag.qa.types import QAItem, RetrievalResult


def _parent_id(doc_id: str) -> str:
    """Strip chunk suffix to get parent document id.

    'part:PN-0001#chunk2' → 'part:PN-0001'
    """
    return doc_id.split("#")[0]


def evaluate_retrieval(
    qa_items: list[QAItem],
    store: VectorStore,
    k: int = 5,
) -> list[RetrievalResult]:
    """Evaluate retrieval quality for a list of QA items.

    Matching is done at the parent document level — a chunk from the correct
    document counts as a hit.

    Args:
        qa_items: Ground-truth QA pairs with source_doc_id
        store: Populated VectorStore to query
        k: Maximum number of results to retrieve per query

    Returns:
        List of RetrievalResult with per-question metrics
    """
    results: list[RetrievalResult] = []

    for item in qa_items:
        if not item.source_doc_id or item.metadata.get("unanswerable"):
            # Unanswerable items have no correct document; scoring them here
            # would only add noise. They are evaluated by correct-refusal rate
            # in the generation stage instead.
            continue
        retrieved = store.query(item.question, k=k)
        retrieved_ids = [r["doc_id"] for r in retrieved]
        retrieved_parents = [_parent_id(rid) for rid in retrieved_ids]
        expected_parent = _parent_id(item.source_doc_id)

        hit_at_1 = expected_parent in retrieved_parents[:1]
        hit_at_3 = expected_parent in retrieved_parents[:3]
        hit_at_5 = expected_parent in retrieved_parents[:5]

        # Reciprocal rank
        rr = 0.0
        for rank, pid in enumerate(retrieved_parents, start=1):
            if pid == expected_parent:
                rr = 1.0 / rank
                break

        results.append(
            RetrievalResult(
                qa_id=item.qa_id,
                question=item.question,
                expected_doc_id=item.source_doc_id,
                retrieved_ids=retrieved_ids,
                hit_at_1=hit_at_1,
                hit_at_3=hit_at_3,
                hit_at_5=hit_at_5,
                reciprocal_rank=rr,
            )
        )

    return results


def compute_retrieval_metrics(results: list[RetrievalResult]) -> dict:
    """Aggregate retrieval results into summary metrics."""
    n = len(results)
    if n == 0:
        return {}
    return {
        "n_questions": n,
        "hit_at_1": round(sum(r.hit_at_1 for r in results) / n, 4),
        "hit_at_3": round(sum(r.hit_at_3 for r in results) / n, 4),
        "hit_at_5": round(sum(r.hit_at_5 for r in results) / n, 4),
        "mrr": round(sum(r.reciprocal_rank for r in results) / n, 4),
    }
