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
