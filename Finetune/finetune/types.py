from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class AnswerType(str, Enum):
    YES_NO_UNCERTAIN = "yes_no_uncertain"
    MCQ = "mcq"
    OPEN_ENDED = "open_ended"


class Record(BaseModel):
    """One training/dev/test item. Source field names vary across releases —
    `data.load.parse_record` normalizes them to this shape.
    """

    id: str
    premises_nl: list[str]
    premises_fol: list[str] | None = None
    question_nl: str
    question_fol: str | None = None
    answer_type: AnswerType
    answer: str | None = None
    options: list[str] | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class Translation(BaseModel):
    """One LLM-produced FOL translation candidate.

    `code` is a Z3-Python DSL program that, when exec'd, populates
    `premises: list[BoolRef]` and `goal: BoolRef` in its namespace.
    """

    code: str
    goal_expr: str | None = None  # populated for MCQ where goal is per-option
    raw_text: str
    sample_index: int


class SolverVerdict(BaseModel):
    """Result of running Z3 on one Translation."""

    answer: str | None
    status: Literal["solved", "unknown", "timeout", "parse_error", "skipped"]
    unsat_core: list[str] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    error: str | None = None


class FinalAnswer(BaseModel):
    """What gets serialized to the submission JSON."""

    answer: str
    explanation: str
    fol: list[str] | None = None
    cot: str | None = None
    premises: list[int] | None = None
    confidence: float = 0.0
    debug: dict[str, Any] = Field(default_factory=dict)
