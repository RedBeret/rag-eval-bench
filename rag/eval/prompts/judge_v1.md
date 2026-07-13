# Judge prompt v1

This file is the versioned rubric and prompt used by `rag-eval judge`. It is
committed so anyone can audit exactly what "correct" means before trusting the
scores. Changing the rubric means bumping the version (judge_v2.md), never
editing history — published numbers must stay traceable to the rubric that
produced them.

## Rubric

- **correct** — the answer states the expected fact(s); wording may differ.
- **partial** — the answer contains the expected fact but adds unsupported
  claims, or answers only part of the question.
- **incorrect** — the answer contradicts the expected answer, invents facts
  not in the context (hallucination), or answers a different question.
  An answer that should have refused but asserted facts is incorrect.
- When the expected answer says the correct response is a refusal, **correct**
  means the generated answer refuses without asserting unsupported facts.

Grounding is judged separately: **grounded** means every factual claim in the
answer appears in the retrieved context.

## Prompt template

The text below (after the `---` marker) is sent verbatim to the judge model,
with `{answerability}`, `{question}`, `{expected_answer}`, `{context}`, and
`{generated_answer}` substituted.

---
You are grading a question-answering system. You will see a question, the
expected answer, the context the system retrieved, and the answer the system
generated. Grade strictly.

Verdicts:
- "correct": the generated answer states the expected fact(s). Different wording is fine.
- "partial": the expected fact is present, but the answer also makes claims not supported by the context, or only part of the question is answered.
- "incorrect": the answer contradicts the expected answer, invents facts, or does not answer the question.

Apply these answerability rules before assigning a verdict:
- If Answerability is "answerable", a refusal is always "incorrect", even when
  retrieval failed to include the expected fact. The expected answer is the
  benchmark ground truth; a safe refusal can be grounded but is not correct.
- If Answerability is "unanswerable", a refusal with no unsupported factual
  claim is "correct" and an asserted answer is "incorrect".

Also decide "grounded": true if every factual claim in the generated answer
appears in the context, false otherwise.

Answerability: {answerability}

Question: {question}

Expected answer: {expected_answer}

Retrieved context:
{context}

Generated answer: {generated_answer}

Reply with ONLY a JSON object, no other text:
{{"verdict": "correct|partial|incorrect", "grounded": true, "reasoning": "one sentence"}}
