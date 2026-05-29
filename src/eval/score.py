"""P1 scoring + per-answer-type breakdown + latency stats.

Compares submission JSON against ground-truth records. Answers are
normalized (case, whitespace, trailing punctuation) before comparison.
Open-ended answers use a relaxed match: exact-after-normalize OR
ground-truth substring of prediction.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from data.types import AnswerType, FinalAnswer, Record


_PUNCT = re.compile(r"[\.\,\;\:\!\?\'\"]+$")


def normalize(s: str) -> str:
    return _PUNCT.sub("", s.strip().lower())


def is_correct(pred: str, gold: str, answer_type: AnswerType) -> bool:
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    if answer_type == AnswerType.OPEN_ENDED:
        # Relaxed: gold is a substring of prediction.
        return g in p
    return False


@dataclass
class TypeStats:
    total: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class EvalReport:
    overall: TypeStats = field(default_factory=TypeStats)
    by_type: dict[str, TypeStats] = field(default_factory=dict)
    latencies_s: list[float] = field(default_factory=list)
    missing: int = 0

    def record(self, atype: AnswerType, correct: bool) -> None:
        self.overall.total += 1
        self.overall.correct += int(correct)
        slot = self.by_type.setdefault(atype.value, TypeStats())
        slot.total += 1
        slot.correct += int(correct)

    def latency_summary(self) -> dict[str, float]:
        if not self.latencies_s:
            return {}
        sorted_l = sorted(self.latencies_s)
        return {
            "mean": statistics.mean(sorted_l),
            "p50": statistics.median(sorted_l),
            "p95": sorted_l[int(0.95 * len(sorted_l))],
            "p99": sorted_l[int(0.99 * len(sorted_l))] if len(sorted_l) >= 100 else max(sorted_l),
            "max": max(sorted_l),
        }

    def to_dict(self) -> dict:
        return {
            "overall": {"total": self.overall.total, "correct": self.overall.correct,
                        "accuracy": self.overall.accuracy},
            "by_type": {k: {"total": v.total, "correct": v.correct, "accuracy": v.accuracy}
                        for k, v in self.by_type.items()},
            "latency_s": self.latency_summary(),
            "missing": self.missing,
        }


def score(records: Iterable[Record], predictions: dict[str, FinalAnswer],
          latencies_s: dict[str, float] | None = None) -> EvalReport:
    report = EvalReport()
    for r in records:
        pred = predictions.get(r.id)
        if pred is None:
            report.missing += 1
            report.record(r.answer_type, correct=False)
            continue
        correct = r.answer is not None and is_correct(pred.answer, r.answer, r.answer_type)
        report.record(r.answer_type, correct)
        if latencies_s and r.id in latencies_s:
            report.latencies_s.append(latencies_s[r.id])
    return report


def write_report(report: EvalReport, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report.to_dict(), indent=2))
