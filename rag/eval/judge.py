"""LLM-as-judge scoring for generation results.

Replaces the keyword-overlap faithfulness proxy with a rubric-driven verdict
from a second model. Three rules keep the scores honest:

1. The rubric + prompt are versioned in-repo (rag/eval/prompts/judge_vN.md) —
   anyone can audit what "correct" means before trusting a number.
2. The judge model must differ from the answer model. A model grading its own
   homework is a known failure mode; the code enforces the separation.
3. Judge verdicts are meant to be spot-checked by a human. `export_grade_sheet`
   emits a CSV for hand-grading and `compute_agreement` publishes the
   human/judge agreement rate — if you haven't measured how often the judge is
   right, you don't know what its scores mean.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from rag.eval.generation import DEFAULT_OLLAMA_URL, GENERATE_TIMEOUT
from rag.qa.types import GenerationResult

JUDGE_PROMPT_VERSION = "v1"
_PROMPTS_DIR = Path(__file__).parent / "prompts"

VERDICTS = ("correct", "partial", "incorrect")


@dataclass
class JudgeResult:
    """A judge verdict for one generated answer."""

    qa_id: str
    question: str
    expected_answer: str
    generated_answer: str
    context_used: str
    answer_model: str
    verdict: str  # correct | partial | incorrect | error
    grounded: bool
    reasoning: str
    judge_model: str
    prompt_version: str = JUDGE_PROMPT_VERSION
    raw_response: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "qa_id": self.qa_id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "generated_answer": self.generated_answer,
            "context_used": self.context_used,
            "answer_model": self.answer_model,
            "verdict": self.verdict,
            "grounded": self.grounded,
            "reasoning": self.reasoning,
            "judge_model": self.judge_model,
            "prompt_version": self.prompt_version,
            "raw_response": self.raw_response,
            "metadata": self.metadata,
        }


def load_judge_prompt(version: str = JUDGE_PROMPT_VERSION) -> str:
    """Load the versioned judge prompt template (text after the `---` marker)."""
    path = _PROMPTS_DIR / f"judge_{version}.md"
    text = path.read_text(encoding="utf-8")
    marker = "\n---\n"
    if marker not in text:
        raise ValueError(f"Judge prompt {path} is missing the '---' template marker")
    return text.split(marker, 1)[1].strip()


def parse_judge_response(raw: str) -> dict:
    """Extract the verdict JSON from a judge response.

    Models wrap JSON in prose or code fences no matter how firmly asked not
    to, so this scans for the first parseable object with a valid verdict.
    Raises ValueError if none is found.
    """
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        verdict = str(obj.get("verdict", "")).lower().strip()
        grounded = obj.get("grounded")
        reasoning = obj.get("reasoning")
        if verdict in VERDICTS and isinstance(grounded, bool) and isinstance(reasoning, str):
            return {
                "verdict": verdict,
                "grounded": grounded,
                "reasoning": reasoning.strip(),
            }
    raise ValueError(f"No valid judge verdict found in response: {raw[:200]!r}")


def _call_judge(prompt: str, model: str, base_url: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 200},
    }
    resp = httpx.post(f"{base_url}/api/generate", json=payload, timeout=GENERATE_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def judge_results(
    results: list[GenerationResult],
    judge_model: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    prompt_version: str = JUDGE_PROMPT_VERSION,
) -> list[JudgeResult]:
    """Judge a list of generation results with a rubric-driven LLM judge.

    Raises ValueError if the judge model is the same as the answer model —
    self-grading is not a measurement.
    """
    normalized_judge = judge_model.casefold().strip()
    answer_models = {r.model.casefold().strip() for r in results}
    if normalized_judge in answer_models:
        raise ValueError(
            f"Judge model {judge_model!r} is also the answer model. "
            "Use a different model to judge — a model grading its own answers "
            "systematically inflates scores."
        )

    template = load_judge_prompt(prompt_version)
    judged: list[JudgeResult] = []

    for r in results:
        if r.generated_answer.startswith("[ERROR:"):
            judged.append(
                JudgeResult(
                    qa_id=r.qa_id,
                    question=r.question,
                    expected_answer=r.expected_answer,
                    generated_answer=r.generated_answer,
                    context_used=r.context_used,
                    answer_model=r.model,
                    verdict="error",
                    grounded=False,
                    reasoning="generation failed; nothing to judge",
                    judge_model=judge_model,
                    prompt_version=prompt_version,
                    metadata={"unanswerable": r.unanswerable, "refused": r.refused},
                )
            )
            continue

        prompt = template.format(
            answerability="unanswerable" if r.unanswerable else "answerable",
            question=r.question,
            expected_answer=r.expected_answer or "(none; the correct response is a refusal)",
            context=r.context_used,
            generated_answer=r.generated_answer,
        )
        raw_response = ""
        try:
            raw_response = _call_judge(prompt, judge_model, base_url)
            parsed = parse_judge_response(raw_response)
            verdict, grounded, reasoning = (
                parsed["verdict"],
                parsed["grounded"],
                parsed["reasoning"],
            )
        except (httpx.HTTPError, ValueError) as exc:
            verdict, grounded, reasoning = "error", False, f"judge call failed: {exc}"

        judged.append(
            JudgeResult(
                qa_id=r.qa_id,
                question=r.question,
                expected_answer=r.expected_answer,
                generated_answer=r.generated_answer,
                context_used=r.context_used,
                answer_model=r.model,
                verdict=verdict,
                grounded=grounded,
                reasoning=reasoning,
                judge_model=judge_model,
                prompt_version=prompt_version,
                raw_response=raw_response,
                metadata={"unanswerable": r.unanswerable, "refused": r.refused},
            )
        )

    return judged


def compute_judge_metrics(judged: list[JudgeResult]) -> dict:
    """Aggregate judge verdicts into summary metrics."""
    n = len(judged)
    if n == 0:
        return {}
    ok = [j for j in judged if j.verdict != "error"]
    return {
        "n_judged": n,
        "judge_model": judged[0].judge_model,
        "prompt_version": judged[0].prompt_version,
        "errors": n - len(ok),
        "correct_rate": round(sum(j.verdict == "correct" for j in ok) / len(ok), 4) if ok else 0.0,
        "partial_rate": round(sum(j.verdict == "partial" for j in ok) / len(ok), 4) if ok else 0.0,
        "incorrect_rate": round(sum(j.verdict == "incorrect" for j in ok) / len(ok), 4)
        if ok
        else 0.0,
        "grounded_rate": round(sum(j.grounded for j in ok) / len(ok), 4) if ok else 0.0,
    }


# ---------------------------------------------------------------------------
# Human/judge agreement study
# ---------------------------------------------------------------------------

GRADE_SHEET_COLUMNS = [
    "qa_id",
    "question",
    "expected_answer",
    "generated_answer",
    "human_verdict",  # fill in by hand: correct | partial | incorrect
]
JUDGE_KEY_COLUMNS = ["qa_id", "judge_verdict"]


def _judge_key_path(grade_sheet: Path) -> Path:
    return grade_sheet.with_suffix(".judge-key.csv")


def export_grade_sheet(judged: list[JudgeResult], path: Path, n: int = 20, seed: int = 42) -> int:
    """Write a seeded, blinded grade sample and a separate judge key.

    The grade sheet omits judge verdicts so the human grade is independent.
    ``compute_agreement`` joins it to the companion key by ``qa_id``.
    """
    eligible = [j for j in judged if j.verdict != "error"]
    rows = random.Random(seed).sample(eligible, k=min(n, len(eligible)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(GRADE_SHEET_COLUMNS)
        for j in rows:
            writer.writerow([j.qa_id, j.question, j.expected_answer, j.generated_answer, ""])
    with _judge_key_path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(JUDGE_KEY_COLUMNS)
        writer.writerows((j.qa_id, j.verdict) for j in rows)
    return len(rows)


def compute_agreement(grade_sheet: Path) -> dict:
    """Compute human/judge agreement from a filled-in grade sheet.

    Returns raw agreement plus Cohen's kappa (agreement corrected for chance —
    with three verdict labels, 33% raw agreement is what guessing gets you).
    """
    with grade_sheet.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(GRADE_SHEET_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Grade sheet is missing columns: {', '.join(sorted(missing))}")
        rows = [r for r in reader if r.get("human_verdict", "").strip()]
    if not rows:
        return {"n_graded": 0}

    key_path = _judge_key_path(grade_sheet)
    if not key_path.exists():
        raise ValueError(f"Judge key not found: {key_path}")
    with key_path.open(newline="", encoding="utf-8") as f:
        key_reader = csv.DictReader(f)
        missing = set(JUDGE_KEY_COLUMNS) - set(key_reader.fieldnames or [])
        if missing:
            raise ValueError(f"Judge key is missing columns: {', '.join(sorted(missing))}")
        judge_verdicts = {row["qa_id"]: row["judge_verdict"] for row in key_reader}

    for row in rows:
        if row["qa_id"] not in judge_verdicts:
            raise ValueError(f"Judge key has no verdict for {row['qa_id']}")
        judge_verdict = judge_verdicts[row["qa_id"]].strip().lower()
        human_verdict = row["human_verdict"].strip().lower()
        if judge_verdict not in VERDICTS or human_verdict not in VERDICTS:
            raise ValueError(
                f"Invalid verdict for {row['qa_id']}: "
                f"judge={judge_verdict!r}, human={human_verdict!r}"
            )

    pairs = [
        (judge_verdicts[r["qa_id"]].strip().lower(), r["human_verdict"].strip().lower())
        for r in rows
    ]
    n = len(pairs)
    agree = sum(j == h for j, h in pairs)
    p_observed = agree / n

    # Cohen's kappa
    labels = set(v for pair in pairs for v in pair)
    p_expected = sum(
        (sum(j == label for j, _ in pairs) / n) * (sum(h == label for _, h in pairs) / n)
        for label in labels
    )
    kappa = (p_observed - p_expected) / (1 - p_expected) if p_expected < 1 else 1.0

    return {
        "n_graded": n,
        "agreement": round(p_observed, 4),
        "cohens_kappa": round(kappa, 4),
        "disagreements": [
            {"qa_id": r["qa_id"], "judge": j, "human": h}
            for r, (j, h) in zip(rows, pairs)
            if j != h
        ],
    }
