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
DEFAULT_MODEL = "gemma:2b"
GENERATE_TIMEOUT = 60.0  # seconds

PROMPT_TEMPLATE = """You are a manufacturing data assistant. Answer the question using ONLY the information provided in the context below. Be concise and factual.

Context:
{context}

Question: {question}

Answer:"""


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
        model: Ollama model name (e.g. 'gemma:2b', 'mistral')
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

        score = _faithfulness_score(generated, context)
        results.append(
            GenerationResult(
                qa_id=item.qa_id,
                question=item.question,
                expected_answer=item.answer,
                generated_answer=generated,
                context_used=context[:500],
                faithfulness_score=score,
                model=model,
            )
        )

    return results


def compute_generation_metrics(results: list[GenerationResult]) -> dict:
    """Aggregate generation results into summary metrics."""
    n = len(results)
    if n == 0:
        return {}
    errors = sum(1 for r in results if r.generated_answer.startswith("[ERROR:"))
    scores = [r.faithfulness_score for r in results if not r.generated_answer.startswith("[ERROR:")]
    return {
        "n_questions": n,
        "model": results[0].model if results else "unknown",
        "errors": errors,
        "avg_faithfulness_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "min_faithfulness_score": round(min(scores), 4) if scores else 0.0,
        "max_faithfulness_score": round(max(scores), 4) if scores else 0.0,
    }
