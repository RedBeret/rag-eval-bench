"""Tests for LLM-as-judge, refusal detection, and the agreement study.

Everything here runs offline: judge calls are intercepted with pytest-httpx,
no models are downloaded, no Ollama server is needed.
"""

from pathlib import Path

import pytest

from rag.corpus.loader import Document
from rag.eval.generation import compute_generation_metrics, is_refusal
from rag.eval.judge import (
    JUDGE_PROMPT_VERSION,
    compute_agreement,
    compute_judge_metrics,
    export_grade_sheet,
    judge_results,
    load_judge_prompt,
    parse_judge_response,
)
from rag.qa.generator import generate_unanswerable_items, select_generation_items
from rag.qa.types import GenerationResult, QAItem


def _gen_result(
    qa_id: str = "qa_1", answer: str = "The status is active.", **kw
) -> GenerationResult:
    defaults = dict(
        qa_id=qa_id,
        question="What is the status of part PN-0001?",
        expected_answer="active",
        generated_answer=answer,
        context_used="Part PN-0001 Hex Bolt status active.",
        faithfulness_score=0.9,
        model="gemma3:4b",
    )
    defaults.update(kw)
    return GenerationResult(**defaults)


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "I don't know based on the provided context.",
        "I dont know the answer to that.",
        "The context does not contain information about PN-99999.",
        "There is no information available about that part.",
        "That part is not found in the provided context.",
        "I cannot determine the status from the context.",
        "Unable to find any record of this supplier.",
        "There is insufficient information to answer.",
    ],
)
def test_is_refusal_positive(answer: str) -> None:
    assert is_refusal(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "The status of part PN-0001 is active.",
        "Zenith Alloys is located in Germany.",
        "It belongs to the Fasteners category.",
    ],
)
def test_is_refusal_negative(answer: str) -> None:
    assert not is_refusal(answer)


# ---------------------------------------------------------------------------
# Unanswerable generation
# ---------------------------------------------------------------------------


def _docs() -> list[Document]:
    return [
        Document(
            doc_id="part:PN-0001",
            text="Part number PN-0001 is a Hex Bolt in the Fasteners category.",
            metadata={"type": "part", "part_number": "PN-0001", "name": "Hex Bolt"},
        ),
        Document(
            doc_id="supplier:VM01",
            text="Vortex Metals is a supplier based in Germany.",
            metadata={"type": "supplier", "name": "Vortex Metals", "code": "VM01"},
        ),
    ]


def test_unanswerable_items_do_not_collide_with_corpus() -> None:
    docs = _docs()
    items = generate_unanswerable_items(docs, n=5, seed=42)
    assert len(items) == 5
    corpus_text = " ".join(d.text.lower() for d in docs)
    for item in items:
        assert item.metadata["unanswerable"] is True
        assert item.source_doc_id == ""
        # the fake identifier must not exist anywhere in the corpus
        for token in item.question.split():
            if token.startswith("PN-9"):
                assert token.lower() not in corpus_text


def test_unanswerable_items_deterministic() -> None:
    docs = _docs()
    a = generate_unanswerable_items(docs, n=3, seed=7)
    b = generate_unanswerable_items(docs, n=3, seed=7)
    assert [i.question for i in a] == [i.question for i in b]


def test_unanswerable_items_are_unique() -> None:
    items = generate_unanswerable_items(_docs(), n=100, seed=42)
    assert len({item.question for item in items}) == 100


def test_generation_selection_keeps_appended_unanswerables() -> None:
    answerable = [QAItem(f"qa_{i}", "Q?", "A", f"doc:{i}", "source") for i in range(100)]
    unanswerable = generate_unanswerable_items(_docs(), n=5, seed=42)
    selected = select_generation_items(answerable + unanswerable, max_items=50)
    assert len(selected) == 50
    assert sum(bool(item.metadata.get("unanswerable")) for item in selected) == 5


# ---------------------------------------------------------------------------
# Generation metrics: refusal accounting
# ---------------------------------------------------------------------------


def test_metrics_split_answerable_and_unanswerable() -> None:
    results = [
        _gen_result("qa_1"),  # answerable, answered
        _gen_result("qa_2", answer="I don't know based on the provided context.", refused=True),
        _gen_result(
            "qa_u1",
            answer="I don't know based on the provided context.",
            unanswerable=True,
            refused=True,
        ),
        _gen_result(
            "qa_u2",
            answer="The status of PN-99999 is active.",  # hallucinated
            unanswerable=True,
            refused=False,
        ),
    ]
    m = compute_generation_metrics(results)
    assert m["n_answerable"] == 2
    assert m["n_unanswerable"] == 2
    assert m["correct_refusal_rate"] == 0.5
    assert m["false_refusal_rate"] == 0.5


def test_metrics_no_unanswerables_omits_refusal_rate() -> None:
    m = compute_generation_metrics([_gen_result()])
    assert "correct_refusal_rate" not in m
    assert m["false_refusal_rate"] == 0.0


def test_unanswerable_errors_count_against_correct_refusal_rate() -> None:
    results = [
        _gen_result("qa_u1", unanswerable=True, refused=True),
        _gen_result("qa_u2", answer="[ERROR: timeout]", unanswerable=True),
        _gen_result("qa_u3", answer="[ERROR: timeout]", unanswerable=True),
    ]
    metrics = compute_generation_metrics(results)
    assert metrics["unanswerable_errors"] == 2
    assert metrics["correct_refusal_rate"] == round(1 / 3, 4)


# ---------------------------------------------------------------------------
# Judge prompt + parsing
# ---------------------------------------------------------------------------


def test_load_judge_prompt_has_placeholders() -> None:
    prompt = load_judge_prompt(JUDGE_PROMPT_VERSION)
    for placeholder in (
        "{answerability}",
        "{question}",
        "{expected_answer}",
        "{context}",
        "{generated_answer}",
    ):
        assert placeholder in prompt


def test_parse_judge_response_plain_json() -> None:
    parsed = parse_judge_response(
        '{"verdict": "correct", "grounded": true, "reasoning": "matches"}'
    )
    assert parsed == {"verdict": "correct", "grounded": True, "reasoning": "matches"}


def test_parse_judge_response_wrapped_in_prose() -> None:
    raw = 'Sure! Here is my grade:\n```json\n{"verdict": "Partial", "grounded": false, "reasoning": "extra claims"}\n```'
    parsed = parse_judge_response(raw)
    assert parsed["verdict"] == "partial"
    assert parsed["grounded"] is False


def test_parse_judge_response_allows_braces_inside_reasoning() -> None:
    parsed = parse_judge_response(
        '{"verdict":"correct","grounded":true,"reasoning":"matches {context}"}'
    )
    assert parsed["reasoning"] == "matches {context}"


def test_parse_judge_response_rejects_non_boolean_grounded() -> None:
    with pytest.raises(ValueError):
        parse_judge_response('{"verdict":"correct","grounded":"false","reasoning":"not grounded"}')


def test_parse_judge_response_invalid_raises() -> None:
    with pytest.raises(ValueError):
        parse_judge_response("The answer looks fine to me!")


# ---------------------------------------------------------------------------
# Judge orchestration (Ollama mocked)
# ---------------------------------------------------------------------------


def test_judge_refuses_self_grading() -> None:
    with pytest.raises(ValueError, match="own answers"):
        judge_results([_gen_result()], judge_model="gemma3:4b")


def test_judge_results_with_mocked_ollama(httpx_mock) -> None:
    httpx_mock.add_response(
        url="http://localhost:11434/api/generate",
        json={"response": '{"verdict": "correct", "grounded": true, "reasoning": "ok"}'},
    )
    judged = judge_results([_gen_result()], judge_model="qwen3:8b")
    assert len(judged) == 1
    assert judged[0].verdict == "correct"
    assert judged[0].grounded is True
    assert judged[0].judge_model == "qwen3:8b"
    assert judged[0].prompt_version == JUDGE_PROMPT_VERSION
    assert judged[0].context_used == "Part PN-0001 Hex Bolt status active."
    assert judged[0].answer_model == "gemma3:4b"
    assert '"verdict": "correct"' in judged[0].raw_response


def test_judge_skips_generation_errors() -> None:
    judged = judge_results(
        [_gen_result(answer="[ERROR: timeout]")],
        judge_model="qwen3:8b",
    )
    assert judged[0].verdict == "error"


def test_compute_judge_metrics() -> None:
    judged = judge_results([_gen_result(answer="[ERROR: x]")], judge_model="qwen3:8b")
    judged[0].verdict = "correct"
    judged[0].grounded = True
    m = compute_judge_metrics(judged)
    assert m["n_judged"] == 1
    assert m["correct_rate"] == 1.0
    assert m["grounded_rate"] == 1.0


# ---------------------------------------------------------------------------
# Agreement study
# ---------------------------------------------------------------------------


def test_grade_sheet_roundtrip(tmp_path: Path, httpx_mock) -> None:
    for _ in range(3):
        httpx_mock.add_response(
            url="http://localhost:11434/api/generate",
            json={"response": '{"verdict": "correct", "grounded": true, "reasoning": "ok"}'},
        )
    judged = judge_results([_gen_result(f"qa_{i}") for i in range(3)], judge_model="qwen3:8b")
    sheet = tmp_path / "grades.csv"
    n = export_grade_sheet(judged, sheet, n=20)
    assert n == 3
    assert "judge_verdict" not in sheet.read_text(encoding="utf-8").splitlines()[0]
    assert sheet.with_suffix(".judge-key.csv").exists()

    # Simulate a human grading: agree twice, disagree once.
    text = sheet.read_text(encoding="utf-8")
    lines = text.strip().splitlines()
    lines[1] += "correct"
    lines[2] += "correct"
    lines[3] += "incorrect"
    sheet.write_text("\n".join(lines) + "\n", encoding="utf-8")

    stats = compute_agreement(sheet)
    assert stats["n_graded"] == 3
    assert stats["agreement"] == round(2 / 3, 4)
    assert len(stats["disagreements"]) == 1
    assert "cohens_kappa" in stats


def test_agreement_empty_sheet(tmp_path: Path) -> None:
    sheet = tmp_path / "grades.csv"
    sheet.write_text(
        "qa_id,question,expected_answer,generated_answer,human_verdict\n",
        encoding="utf-8",
    )
    assert compute_agreement(sheet) == {"n_graded": 0}


def test_agreement_rejects_invalid_human_verdict(tmp_path: Path) -> None:
    sheet = tmp_path / "grades.csv"
    sheet.write_text(
        "qa_id,question,expected_answer,generated_answer,human_verdict\nq1,Q,A,A,maybe\n",
        encoding="utf-8",
    )
    sheet.with_suffix(".judge-key.csv").write_text(
        "qa_id,judge_verdict\nq1,correct\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Invalid verdict"):
        compute_agreement(sheet)
