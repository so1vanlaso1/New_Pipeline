"""Render the final NL explanation and assemble the submission JSON.

Two sources of explanation:

  - Stage 3a (Z3 succeeded): the unsat-core labels point to which premises
    were actually used in the proof. We list them in NL and emit a short
    one-line justification.
  - Stage 3b (CoT fallback): we take the first sample's trace, strip the
    'FINAL ANSWER:' line, and use the reasoning as the explanation.

P1 is the priority, so we keep explanation prose terse — no LLM call just to
"polish" it.
"""

from __future__ import annotations

from data.types import FinalAnswer, Record, Translation


def _premise_indices_from_core(core_labels: list[str]) -> list[int]:
    """Convert ['p0', 'p3', 'p7'] back into [0, 3, 7]."""
    out: list[int] = []
    for label in core_labels:
        if label.startswith("p") and label[1:].isdigit():
            out.append(int(label[1:]))
    return sorted(set(out))


def from_symbolic(
    record: Record,
    answer: str,
    confidence: float,
    unsat_core: list[str],
    winning_translation: Translation | None,
) -> FinalAnswer:
    used_idx = _premise_indices_from_core(unsat_core)
    used_premises = [record.premises_nl[i] for i in used_idx if i < len(record.premises_nl)]

    if used_premises:
        bullets = "; ".join(used_premises)
        explanation = f"From the premises ({bullets}), the answer is {answer}."
    else:
        explanation = f"Symbolic derivation from the given premises yields {answer}."

    fol = None
    if winning_translation is not None:
        # Emit the whole Z3 Python program as a single FOL trace block.
        fol = [winning_translation.code]

    return FinalAnswer(
        answer=answer,
        explanation=explanation,
        fol=fol,
        cot=None,
        premises=used_idx or None,
        confidence=confidence,
        debug={"source": "symbolic"},
    )


def from_cot(
    record: Record,
    answer: str,
    confidence: float,
    cot_trace: str,
) -> FinalAnswer:
    # Drop the FINAL ANSWER tail so the explanation reads as prose.
    cleaned = cot_trace.rsplit("FINAL ANSWER", 1)[0].strip()
    explanation = cleaned if cleaned else f"The answer is {answer}."
    return FinalAnswer(
        answer=answer,
        explanation=explanation,
        fol=None,
        cot=cot_trace,
        premises=None,
        confidence=confidence,
        debug={"source": "cot"},
    )


def from_failure(record: Record) -> FinalAnswer:
    """Last-resort default when both paths fail (shouldn't happen if CoT runs)."""
    default = {
        # Best-prior guess for each answer type when we have nothing to go on.
        "yes_no_uncertain": "Uncertain",
        "mcq": (record.options[0] if record.options else ""),
        "open_ended": "",
    }[record.answer_type.value]
    return FinalAnswer(
        answer=default,
        explanation="Unable to derive an answer from the premises within the time budget.",
        confidence=0.0,
        debug={"source": "failure_default"},
    )
