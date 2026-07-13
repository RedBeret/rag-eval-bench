"""Generation evaluation via Ollama HTTP API.

Calls a local Ollama instance to generate answers given retrieved context,
then scores faithfulness using keyword overlap as a lightweight proxy metric.

If Ollama is not available (no server running or --skip-generation flag),
this module is skipped gracefully.
"""

from __future__ import annotations

import re

import httpx

from rag.index.store import VectorStore
from rag.qa.types import GenerationResult, QAItem

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:4b"
GENERATE_TIMEOUT = 60.0  # seconds

REFUSAL_SENTENCE = "I don't know based on the provided context."

PROMPT_TEMPLATE = f"""You are a manufacturing data assistant. Answer the question using ONLY the information provided in the context below. Be concise and factual.

If the context does not contain the information needed to answer, reply exactly:
"{REFUSAL_SENTENCE}"

Context:
{{context}}

Question: {{question}}

Answer:"""

# Phrases that count as a refusal. Deliberately broad: models paraphrase even
# when told not to, and a paraphrased refusal is still a refusal.
_REFUSAL_PATTERNS = [
    r"i don'?t know",
    r"do(?:es)? not (?:contain|include|provide|mention|appear)",
    r"no (?:information|mention|record|data) (?:about|on|of|available|found)",
    r"not (?:found|available|provided|mentioned|present) in the (?:provided )?context",
    r"cannot (?:answer|determine|find)",
    r"unable to (?:answer|determine|find)",
    r"insufficient (?:information|context)",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal(answer: str) -> bool:
    """True if the answer declines to answer rather than asserting facts."""
    return bool(_REFUSAL_RE.search(answer))


def _faithfulness_score(answer: str, context: str) -> float:
    """Compute a keyword overlap score between answer and context.

    This is a lightweight proxy for faithfulness — it checks what fraction
    of non-stopword tokens in the answer appear in the context.
    A proper faithfulness metric would use an LLM judge.
    """
    stopwords = {"the", "a", "an", "is", "in", "of", "for", "to", "and", "or", "it", "its"}

    def tokens(text: str) -> set[str]:
        return {w.lower() for w in re.findall(r"\b\w+\b", text) if w.lower() not in stopwords}

    answer_tokens = tokens(answer)
    context_tokens = tokens(context)

    if not answer_tokens:
        return 0.0
    overlap = answer_tokens & context_tokens
    return round(len(overlap) / len(answer_tokens), 4)


def _is_ollama_available(base_url: str = DEFAULT_OLLAMA_URL) -> bool:
    """Check if Ollama server is reachable."""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=3.0)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def generate_answer(
    question: str,
    context: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> str:
    """Call Ollama to generate an answer given context."""
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 150},
    }
    resp = httpx.post(
        f"{base_url}/api/generate",
        json=payload,
        timeout=GENERATE_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def evaluate_generation(
    qa_items: list[QAItem],
    store: VectorStore,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    k: int = 3,
) -> list[GenerationResult]:
    """Evaluate generation quality for a list of QA items.

    Retrieves context from the store, calls Ollama to generate an answer,
    and scores the answer for faithfulness.

    Args:
        qa_items: Ground-truth QA pairs
        store: Populated VectorStore for retrieval
        model: Ollama model name (e.g. 'gemma3:4b', 'mistral')
        base_url: Ollama server base URL
        k: Number of context documents to retrieve

    Returns:
        List of GenerationResult with per-question metrics
    """
    results: list[GenerationResult] = []

    for item in qa_items:
        retrieved = store.query(item.question, k=k)
        context = "\n\n".join(r["text"] for r in retrieved)

        try:
            generated = generate_answer(item.question, context, model=model, base_url=base_url)
        except (httpx.HTTPError, Exception) as exc:
            generated = f"[ERROR: {exc}]"

        unanswerable = bool(item.metadata.get("unanswerable"))
        refused = is_refusal(generated) and not generated.startswith("[ERROR:")
        score = _faithfulness_score(generated, context)
        results.append(
            GenerationResult(
                qa_id=item.qa_id,
                question=item.question,
                expected_answer=item.answer,
                generated_answer=generated,
                # The judge needs the same complete context the answer model saw.
                context_used=context,
                faithfulness_score=score,
                model=model,
                unanswerable=unanswerable,
                refused=refused,
            )
        )

    return results


def compute_generation_metrics(results: list[GenerationResult]) -> dict:
    """Aggregate generation results into summary metrics.

    Answerable and unanswerable questions are scored separately:
    faithfulness only makes sense for answerable questions, while
    unanswerable questions are scored on correct-refusal rate.
    A false refusal (refusing an answerable question) is also counted —
    a system that refuses everything is useless, not safe.
    """
    n = len(results)
    if n == 0:
        return {}
    errors = sum(1 for r in results if r.generated_answer.startswith("[ERROR:"))
    answerable = [r for r in results if not r.unanswerable]
    unanswerable = [r for r in results if r.unanswerable]
    answerable_ok = [r for r in answerable if not r.generated_answer.startswith("[ERROR:")]
    unanswerable_ok = [r for r in unanswerable if not r.generated_answer.startswith("[ERROR:")]
    scores = [r.faithfulness_score for r in answerable_ok]

    metrics = {
        "n_questions": n,
        "model": results[0].model if results else "unknown",
        "errors": errors,
        "avg_faithfulness_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "min_faithfulness_score": round(min(scores), 4) if scores else 0.0,
        "max_faithfulness_score": round(max(scores), 4) if scores else 0.0,
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "answerable_errors": len(answerable) - len(answerable_ok),
        "unanswerable_errors": len(unanswerable) - len(unanswerable_ok),
        "false_refusal_rate": round(sum(r.refused for r in answerable_ok) / len(answerable), 4)
        if answerable
        else 0.0,
    }
    if unanswerable:
        metrics["correct_refusal_rate"] = round(
            sum(r.refused for r in unanswerable_ok) / len(unanswerable), 4
        )
    return metrics
