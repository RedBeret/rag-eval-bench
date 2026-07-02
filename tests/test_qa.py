"""Tests for QA generation."""

from rag.corpus.loader import Document
from rag.qa.generator import deduplicate_questions, generate_qa_dataset, generate_qa_from_doc
from rag.qa.types import QAItem


def make_part_doc(
    pn: str = "PN-0001", name: str = "Hex Bolt", category: str = "Fasteners"
) -> Document:
    return Document(
        doc_id=f"part:{pn}",
        text=f"Part number {pn} is a {name} in the {category} category. Unit of measure: EA. Status: active.",
        metadata={
            "type": "part",
            "part_number": pn,
            "name": name,
            "category": category,
            "status": "active",
            "uom": "EA",
        },
    )


def make_supplier_doc(
    code: str = "VRT-001", name: str = "Vortex Metals", country: str = "US"
) -> Document:
    return Document(
        doc_id=f"supplier:{code}",
        text=f"Supplier {name} (code: {code}) is located in {country}. Contact: proc@vortex.com. Active: 1.",
        metadata={"type": "supplier", "code": code, "name": name, "country": country},
    )


def test_generate_qa_from_part_doc() -> None:
    doc = make_part_doc()
    items = generate_qa_from_doc(doc)
    assert len(items) >= 1
    assert all(isinstance(i, QAItem) for i in items)
    assert all(i.source_doc_id == "part:PN-0001" for i in items)
    assert all(i.question and i.answer for i in items)


def test_generate_qa_from_supplier_doc() -> None:
    doc = make_supplier_doc()
    items = generate_qa_from_doc(doc)
    assert len(items) >= 1
    assert any("country" in i.question.lower() or "US" in i.answer for i in items)


def test_generate_qa_dataset_respects_max() -> None:
    docs = [make_part_doc(f"PN-{i:04d}", f"Part {i}", "Fasteners") for i in range(50)]
    items = generate_qa_dataset(docs, max_items=10, seed=1)
    assert len(items) <= 10


def test_generate_qa_dataset_deterministic() -> None:
    docs = [make_part_doc(f"PN-{i:04d}", f"Part {i}", "Fasteners") for i in range(30)]
    items_a = generate_qa_dataset(docs, max_items=20, seed=42)
    items_b = generate_qa_dataset(docs, max_items=20, seed=42)
    assert [i.qa_id for i in items_a] == [i.qa_id for i in items_b]


def test_deduplicate_questions_removes_dupes() -> None:
    items = [
        QAItem("q1", "What is the part number for Hex Bolt?", "PN-0001", "part:PN-0001", "..."),
        QAItem("q2", "What is the part number for Hex Bolt?", "PN-0001", "part:PN-0001", "..."),
        QAItem("q3", "What category does PN-0002 belong to?", "Fasteners", "part:PN-0002", "..."),
    ]
    deduped = deduplicate_questions(items)
    assert len(deduped) == 2
    assert deduped[0].qa_id == "q1"
    assert deduped[1].qa_id == "q3"


def test_qa_item_roundtrip() -> None:
    item = QAItem("q1", "Question?", "Answer", "doc:1", "source text", {"key": "val"})
    assert QAItem.from_dict(item.to_dict()) == item
