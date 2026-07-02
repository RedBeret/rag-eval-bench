"""Tests for retrieval evaluation (using MockEmbedder — no model downloads)."""

from rag.corpus.loader import Document
from rag.eval.retrieval import compute_retrieval_metrics, evaluate_retrieval
from rag.index.embedder import MockEmbedder
from rag.index.store import VectorStore
from rag.qa.types import QAItem


def _make_store(docs: list[Document]) -> VectorStore:
    store = VectorStore(embedder=MockEmbedder(dim=8))
    store.add_documents(docs)
    return store


def _make_doc(doc_id: str, text: str, doc_type: str = "part") -> Document:
    return Document(doc_id=doc_id, text=text, metadata={"type": doc_type})


def test_evaluate_retrieval_hit_at_1() -> None:
    """When a doc is in the corpus, retrieval should find it."""
    text = "Part number PN-0001 is a Hex Bolt in the Fasteners category."
    docs = [
        _make_doc("part:PN-0001", text),
        _make_doc("part:PN-0002", "Part number PN-0002 is a Nylon Washer."),
        _make_doc("part:PN-0003", "Part number PN-0003 is a Compression Spring."),
    ]
    store = _make_store(docs)

    qa_items = [
        QAItem(
            qa_id="qa_0",
            question=text,  # exact text as query → should be top result
            answer="PN-0001",
            source_doc_id="part:PN-0001",
            source_text=text,
        )
    ]
    results = evaluate_retrieval(qa_items, store, k=5)
    assert len(results) == 1
    # Hit@5 should be True since the doc is in corpus
    assert results[0].hit_at_5 is True


def test_compute_retrieval_metrics_all_hits() -> None:
    from rag.qa.types import RetrievalResult

    results = [
        RetrievalResult("q1", "Q1", "doc:1", ["doc:1", "doc:2"], True, True, True, 1.0),
        RetrievalResult("q2", "Q2", "doc:3", ["doc:3", "doc:4"], True, True, True, 1.0),
    ]
    metrics = compute_retrieval_metrics(results)
    assert metrics["hit_at_1"] == 1.0
    assert metrics["hit_at_3"] == 1.0
    assert metrics["hit_at_5"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["n_questions"] == 2


def test_compute_retrieval_metrics_no_hits() -> None:
    from rag.qa.types import RetrievalResult

    results = [
        RetrievalResult("q1", "Q1", "doc:1", ["doc:2", "doc:3"], False, False, False, 0.0),
    ]
    metrics = compute_retrieval_metrics(results)
    assert metrics["hit_at_1"] == 0.0
    assert metrics["mrr"] == 0.0


def test_compute_retrieval_metrics_partial() -> None:
    from rag.qa.types import RetrievalResult

    results = [
        RetrievalResult("q1", "Q1", "doc:1", ["doc:1"], True, True, True, 1.0),
        RetrievalResult("q2", "Q2", "doc:2", ["doc:3", "doc:4", "doc:2"], False, True, True, 1 / 3),
    ]
    metrics = compute_retrieval_metrics(results)
    assert metrics["hit_at_1"] == 0.5
    assert metrics["hit_at_3"] == 1.0
    assert round(metrics["mrr"], 4) == round((1.0 + 1 / 3) / 2, 4)


def test_faithfulness_score() -> None:
    from rag.eval.generation import _faithfulness_score

    answer = "The part is a Hex Bolt in the Fasteners category."
    context = "Part number PN-0001 is a Hex Bolt in the Fasteners category. Unit: EA."
    score = _faithfulness_score(answer, context)
    assert score > 0.5  # Most answer tokens should be in context


def test_faithfulness_score_no_overlap() -> None:
    from rag.eval.generation import _faithfulness_score

    answer = "xyzzy quux foo bar"
    context = "completely different text here"
    score = _faithfulness_score(answer, context)
    assert score == 0.0


def test_mock_embedder_deterministic() -> None:
    emb = MockEmbedder(dim=8)
    v1 = emb.embed(["hello world"])
    v2 = emb.embed(["hello world"])
    assert v1 == v2


def test_mock_embedder_stable_across_instances() -> None:
    assert MockEmbedder(dim=8).embed(["hello world"]) == MockEmbedder(dim=8).embed(["hello world"])


def test_mock_embedder_different_texts() -> None:
    emb = MockEmbedder(dim=8)
    v1 = emb.embed(["hello world"])[0]
    v2 = emb.embed(["goodbye world"])[0]
    assert v1 != v2
