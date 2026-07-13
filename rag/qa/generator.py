"""Template-based QA pair generation from corpus documents.

Generates deterministic question-answer pairs without requiring a running LLM.
This ensures CI can run the full benchmark pipeline offline.

For richer QA generation using a local Ollama model, see rag/eval/generation.py.
"""

from __future__ import annotations

import random
import re

from rag.corpus.loader import Document
from rag.qa.types import QAItem

# Question templates keyed by document type
PART_TEMPLATES = [
    ("What is the part number for {name}?", "{part_number}"),
    ("What category does part {part_number} belong to?", "{category}"),
    ("What is the status of part {part_number}?", "{status}"),
    ("Which part number is known as {name}?", "{part_number}"),
]

SUPPLIER_TEMPLATES = [
    ("What country is supplier {name} located in?", "{country}"),
    ("What is the supplier code for {name}?", "{code}"),
    ("Where is {name} based?", "{country}"),
]

GENERIC_TEMPLATES = [
    ("What information is available about {topic}?", "{summary}"),
]

# Questions the corpus deliberately cannot answer. A correct system refuses;
# answering anyway is a hallucination. Refusing correctly is scored — most RAG
# evals skip this entirely.
UNANSWERABLE_TEMPLATES = [
    "What is the status of part {fake_pn}?",
    "What category does part {fake_pn} belong to?",
    "What country is supplier {fake_supplier} located in?",
    "What is the unit price history for part {fake_pn}?",
    "Who approved the latest revision of part {fake_pn}?",
]


def _extract_fields(doc: Document) -> dict[str, str]:
    """Extract template fields from document metadata + text."""
    meta = doc.metadata
    fields: dict[str, str] = {k: str(v) for k, v in meta.items() if v}
    # Try to extract a short name from text
    words = doc.text.split()[:6]
    fields.setdefault("topic", " ".join(words))
    fields.setdefault("summary", doc.text[:120])
    return fields


def _fill_template(template: str, fields: dict[str, str]) -> str | None:
    """Fill a template string with fields. Returns None if any field is missing/empty."""
    try:
        result = template.format(**fields)
        # Reject if any field was empty or 'None'
        if "None" in result or "{" in result:
            return None
        return result
    except KeyError:
        return None


def generate_qa_from_doc(doc: Document, qa_index: int = 0) -> list[QAItem]:
    """Generate QA pairs from a single document using templates.

    Returns 0–2 QA pairs per document depending on available metadata.
    """
    doc_type = doc.metadata.get("type", "generic")
    fields = _extract_fields(doc)

    if doc_type == "part":
        templates = PART_TEMPLATES
    elif doc_type == "supplier":
        templates = SUPPLIER_TEMPLATES
    else:
        templates = GENERIC_TEMPLATES

    items: list[QAItem] = []
    for i, (q_tmpl, a_tmpl) in enumerate(templates):
        question = _fill_template(q_tmpl, fields)
        answer = _fill_template(a_tmpl, fields)
        if question and answer and answer.strip() not in ("", "None", "unknown"):
            items.append(
                QAItem(
                    qa_id=f"qa_{qa_index:04d}_{i}",
                    question=question,
                    answer=answer,
                    source_doc_id=doc.doc_id,
                    source_text=doc.text,
                    metadata={"doc_type": doc_type, "template_index": i},
                )
            )

    return items


def generate_qa_dataset(
    docs: list[Document],
    max_items: int = 200,
    seed: int = 42,
) -> list[QAItem]:
    """Generate a QA dataset from a corpus.

    Args:
        docs: List of corpus documents
        max_items: Maximum number of QA pairs to generate
        seed: Random seed for reproducible sampling

    Returns:
        List of QAItem objects
    """
    rng = random.Random(seed)
    sampled = rng.sample(docs, min(len(docs), max_items * 2))

    all_items: list[QAItem] = []
    for idx, doc in enumerate(sampled):
        items = generate_qa_from_doc(doc, qa_index=idx)
        all_items.extend(items)
        if len(all_items) >= max_items:
            break

    return all_items[:max_items]


def generate_unanswerable_items(
    docs: list[Document],
    n: int = 5,
    seed: int = 42,
) -> list[QAItem]:
    """Generate questions the corpus provably cannot answer.

    Fake part numbers / supplier names are drawn deterministically from the seed
    and checked against the corpus so they are guaranteed absent. The expected
    behavior is a refusal; ``metadata["unanswerable"]`` marks these items so the
    evaluators can score correct-refusal rate separately.

    These items carry an empty ``source_doc_id`` and are excluded from
    retrieval metrics (there is no right document to retrieve).
    """
    rng = random.Random(seed)

    # Collect every identifier that actually exists so fakes never collide.
    existing: set[str] = set()
    corpus_text = "\n".join(doc.text.casefold() for doc in docs)
    for doc in docs:
        for value in doc.metadata.values():
            if value:
                existing.add(str(value).lower())
        existing.update(w.lower() for w in doc.text.split())

    def fake_part_number() -> str:
        while True:
            candidate = f"PN-{rng.randint(90000, 99999)}"
            normalized = candidate.casefold()
            if normalized not in existing and normalized not in corpus_text:
                existing.add(normalized)
                return candidate

    def fake_supplier() -> str:
        while True:
            candidate = f"Synthetic Supplier {rng.randint(90000, 99999)}"
            normalized = candidate.casefold()
            if normalized not in existing and normalized not in corpus_text:
                existing.add(normalized)
                return candidate

    items: list[QAItem] = []
    for i in range(n):
        template = UNANSWERABLE_TEMPLATES[i % len(UNANSWERABLE_TEMPLATES)]
        fields = {}
        if "{fake_pn}" in template:
            fields["fake_pn"] = fake_part_number()
        if "{fake_supplier}" in template:
            fields["fake_supplier"] = fake_supplier()
        question = template.format(**fields)
        items.append(
            QAItem(
                qa_id=f"qa_unanswerable_{i:03d}",
                question=question,
                answer="",  # correct behavior is a refusal, not an answer
                source_doc_id="",
                source_text="",
                metadata={"unanswerable": True, "template_index": i % len(UNANSWERABLE_TEMPLATES)},
            )
        )
    return items


def select_generation_items(items: list[QAItem], max_items: int) -> list[QAItem]:
    """Select a bounded generation set without dropping appended negatives."""
    if max_items <= 0:
        return []
    unanswerable = [item for item in items if item.metadata.get("unanswerable")]
    answerable = [item for item in items if not item.metadata.get("unanswerable")]
    selected_unanswerable = unanswerable[:max_items]
    answerable_slots = max_items - len(selected_unanswerable)
    return answerable[:answerable_slots] + selected_unanswerable


def deduplicate_questions(items: list[QAItem]) -> list[QAItem]:
    """Remove QA pairs with duplicate questions."""
    seen: set[str] = set()
    result: list[QAItem] = []
    for item in items:
        norm = re.sub(r"\s+", " ", item.question.lower().strip())
        if norm not in seen:
            seen.add(norm)
            result.append(item)
    return result
