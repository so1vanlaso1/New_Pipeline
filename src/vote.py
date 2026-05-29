"""Aggregate K Z3 verdicts into one answer + confidence.

Voting is over `solved` verdicts only. `parse_error`, `timeout`, and
`unknown` are dropped — they signify a broken translation, not a "No" or
"Uncertain" signal. If too few verdicts survive (or there is no clear
majority), we return None and the pipeline routes to the CoT fallback.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from data.types import SolverVerdict


def aggregate(
    verdicts: Iterable[SolverVerdict],
    k: int,
    high_threshold: int = 4,
    medium_threshold: int = 3,
) -> tuple[str | None, float, list[str]]:
    """Return (answer, confidence, unsat_core).

    `high_threshold` and `medium_threshold` are inclusive vote counts. With
    K=5: 4 or 5 agreeing votes → 0.95, 3 → 0.70, else None (fallback).
    """
    solved = [v for v in verdicts if v.status == "solved" and v.answer is not None]
    if not solved:
        return None, 0.0, []

    counter = Counter(v.answer for v in solved)
    top_answer, top_votes = counter.most_common(1)[0]

    if top_votes >= high_threshold:
        confidence = 0.95
    elif top_votes >= medium_threshold:
        confidence = 0.70
    else:
        return None, 0.0, []

    # Use the unsat-core from one of the winning verdicts (the first).
    winning_core = next(
        (v.unsat_core for v in solved if v.answer == top_answer and v.unsat_core),
        [],
    )
    return top_answer, confidence, winning_core
