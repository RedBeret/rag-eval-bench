"""Data types for QA pairs and evaluation results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QAItem:
    """A single question-answer pair with grounding metadata."""

    qa_id: str
    question: str
    answer: str
    source_doc_id: str
    source_text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "qa_id": self.qa_id,
            "question": self.question,
            "answer": self.answer,
            "source_doc_id": self.source_doc_id,
            "source_text": self.source_text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> QAItem:
        return cls(
            qa_id=d["qa_id"],
            question=d["question"],
            answer=d["answer"],
            source_doc_id=d["source_doc_id"],
            source_text=d["source_text"],
            metadata=d.get("metadata", {}),
        )


@dataclass
class RetrievalResult:
    """Result of a single retrieval query."""

    qa_id: str
    question: str
    expected_doc_id: str
    retrieved_ids: list[str]
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    reciprocal_rank: float

    def to_dict(self) -> dict:
        return {
            "qa_id": self.qa_id,
            "question": self.question,
            "expected_doc_id": self.expected_doc_id,
            "retrieved_ids": self.retrieved_ids,
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "hit_at_5": self.hit_at_5,
            "reciprocal_rank": self.reciprocal_rank,
        }


@dataclass
class GenerationResult:
    """Result of a single generation evaluation."""

    qa_id: str
    question: str
    expected_answer: str
    generated_answer: str
    context_used: str
    faithfulness_score: float  # 0.0–1.0 keyword overlap proxy
    model: str

    def to_dict(self) -> dict:
        return {
            "qa_id": self.qa_id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "generated_answer": self.generated_answer,
            "context_used": self.context_used,
            "faithfulness_score": self.faithfulness_score,
            "model": self.model,
        }
