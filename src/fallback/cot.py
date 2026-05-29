"""LLM chain-of-thought fallback path.

Used when:
  - the question is open-ended (Z3 can't enumerate)
  - the K Z3 verdicts don't reach a confident majority
  - the translator fails to produce valid SMT-LIB

We re-use the translator's vLLM handle but disable the LoRA adapter, so the
base instruction-tuned model handles free-form reasoning. Same self-
consistency K, majority vote on the final answer line.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from data.types import AnswerType, Record
from translator.infer import LLMBackend


@dataclass
class CotConfig:
    k_samples: int = 5
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 2048


SYSTEM_YNU = """You answer Yes/No/Uncertain questions about university regulations using the given premises.

Reason step by step using only the premises. Do not use outside knowledge. At the end, write a single line:
FINAL ANSWER: Yes
or
FINAL ANSWER: No
or
FINAL ANSWER: Uncertain
"""

SYSTEM_MCQ = """You answer multiple-choice questions about university regulations using the given premises.

Reason step by step using only the premises. Do not use outside knowledge. At the end, write a single line:
FINAL ANSWER: <letter>
where <letter> is the option label (A, B, C, ...) of the correct choice. If NONE of the listed options follows from the premises, write:
FINAL ANSWER: Unknown
"""

SYSTEM_OPEN = """You answer questions about university regulations using the given premises.

Reason step by step using only the premises. Do not use outside knowledge. Keep the final answer short. At the end, write a single line:
FINAL ANSWER: <your concise answer>
"""


_FINAL_LINE = re.compile(r"FINAL ANSWER\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _system_for(rtype: AnswerType) -> str:
    return {
        AnswerType.YES_NO_UNCERTAIN: SYSTEM_YNU,
        AnswerType.MCQ: SYSTEM_MCQ,
        AnswerType.OPEN_ENDED: SYSTEM_OPEN,
    }[rtype]


def _user_for(record: Record) -> str:
    prem = "\n".join(f"- {p}" for p in record.premises_nl)
    parts = [f"Premises:\n{prem}\n", f"Question: {record.question_nl}"]
    if record.answer_type == AnswerType.MCQ and record.options:
        opts = "\n".join(
            f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(record.options)
        )
        parts.append(f"\nOptions:\n{opts}")
    return "\n".join(parts)


def _extract_final(text: str) -> str | None:
    m = _FINAL_LINE.search(text)
    return m.group(1).strip() if m else None


def _normalize_answer(answer: str, record: Record) -> str:
    a = answer.strip().rstrip(".")
    if record.answer_type == AnswerType.YES_NO_UNCERTAIN:
        low = a.lower()
        if low.startswith("y"):
            return "Yes"
        if low.startswith("n"):
            return "No"
        return "Uncertain"
    if record.answer_type == AnswerType.MCQ and record.options:
        if a.lower().strip().rstrip(".") in {"unknown", "none", "none of the above", "n/a"}:
            return "Unknown"
        # Accept "A", "A.", "A) text", or the option text itself.
        first = a.split()[0].strip("().,") if a else ""
        if len(first) == 1 and first.upper().isalpha():
            idx = ord(first.upper()) - ord("A")
            if 0 <= idx < len(record.options):
                return record.options[idx]
        return a
    return a


def run_cot(
    backend: LLMBackend,
    record: Record,
    cfg: CotConfig,
) -> tuple[str | None, str, float]:
    """Return (final_answer, full_cot_trace, confidence)."""
    messages = [
        {"role": "system", "content": _system_for(record.answer_type)},
        {"role": "user", "content": _user_for(record)},
    ]
    raw = backend.chat_generate(
        batch_messages=[messages],
        n=cfg.k_samples,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
        lora_path=None,  # base model only on the fallback path
    )
    samples = raw[0] if raw else []

    answers: list[str] = []
    for s in samples:
        final = _extract_final(s)
        if final:
            answers.append(_normalize_answer(final, record))

    trace = samples[0] if samples else ""

    if not answers:
        return None, trace, 0.0
    counter = Counter(answers)
    top, votes = counter.most_common(1)[0]
    confidence = votes / cfg.k_samples
    return top, trace, confidence
