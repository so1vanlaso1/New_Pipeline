"""End-to-end orchestration for one record.

  Stage 1: NL → SMT-LIB (K samples) via translator (vLLM + optional LoRA)
  Stage 2: Z3 entailment check per sample
  Stage 3a: majority vote across surviving verdicts (high/medium confidence)
  Stage 3b: CoT fallback (open-ended, low confidence, or all-failed)
  Stage 4: assemble FinalAnswer (answer, explanation, fol, cot, premises, confidence)

Each stage logs wall time; `process_record` enforces an overall budget.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from data.types import AnswerType, FinalAnswer, Record, SolverVerdict, Translation
from explain import from_cot, from_failure, from_symbolic
from fallback.cot import CotConfig, run_cot
from solver.z3_runner import run_mcq, run_yes_no_uncertain
from translator.infer import Translator
from vote import aggregate

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    wall_clock_budget_s: float = 55.0
    solver_timeout_ms: int = 5000
    emit_unsat_core: bool = True
    vote_high_threshold: int = 4
    vote_medium_threshold: int = 3
    cot: CotConfig = field(default_factory=CotConfig)


@dataclass
class StageTimings:
    translate_s: float = 0.0
    solve_s: float = 0.0
    vote_s: float = 0.0
    cot_s: float = 0.0
    total_s: float = 0.0


_GOAL_LINE = __import__("re").compile(r"^\s*goal\s*=\s*(.+?)\s*$", __import__("re").MULTILINE)


def _extract_goal_expr(code: str) -> str | None:
    """Pull the right-hand side of the last `goal = ...` line out of a Z3-Python program."""
    matches = _GOAL_LINE.findall(code)
    return matches[-1].strip() if matches else None


def _solve_ynu(
    translations: list[Translation],
    cfg: PipelineConfig,
) -> tuple[list[SolverVerdict], Translation | None]:
    """Run Z3 on each of the K Yes/No/Uncertain translations."""
    verdicts: list[SolverVerdict] = []
    for t in translations:
        v = run_yes_no_uncertain(
            t.code,
            timeout_ms=cfg.solver_timeout_ms,
            emit_unsat_core=cfg.emit_unsat_core,
        )
        verdicts.append(v)
    return verdicts, (translations[0] if translations else None)


def _solve_mcq(
    per_option_translations: list[list[Translation]],
    options: list[str],
    cfg: PipelineConfig,
) -> tuple[list[SolverVerdict], Translation | None]:
    """Pair up sample-k across options into one MCQ pass.

    Each Z3 Python program is self-contained (declares its own sort, predicates,
    constants). For MCQ we share the FIRST option's program as the premise
    environment and treat each option's `goal` line as the per-option goal —
    the runner re-evaluates each goal expression in the shared namespace.
    """
    if not per_option_translations or not all(per_option_translations):
        return [], None

    k = min(len(group) for group in per_option_translations)
    verdicts: list[SolverVerdict] = []
    winning: Translation | None = None
    for sample_i in range(k):
        sample_translations = [group[sample_i] for group in per_option_translations]
        premise_code = sample_translations[0].code
        option_goals: list[str] = []
        for t in sample_translations:
            g = _extract_goal_expr(t.code)
            if g is None:
                # If we can't find a goal line, skip this option's slot.
                option_goals.append("False")
            else:
                option_goals.append(g)
        v = run_mcq(
            premise_code, option_goals,
            timeout_ms=cfg.solver_timeout_ms,
            emit_unsat_core=cfg.emit_unsat_core,
        )
        if v.answer is not None and v.answer.isdigit():
            opt_idx = int(v.answer)
            if 0 <= opt_idx < len(options):
                v = SolverVerdict(
                    answer=options[opt_idx],
                    status=v.status,
                    unsat_core=v.unsat_core,
                    elapsed_ms=v.elapsed_ms,
                )
            if winning is None:
                winning = sample_translations[0]
        verdicts.append(v)
    return verdicts, winning


def process_record(
    record: Record,
    translator: Translator,
    cfg: PipelineConfig | None = None,
) -> tuple[FinalAnswer, StageTimings]:
    cfg = cfg or PipelineConfig()
    t = StageTimings()
    t0 = time.perf_counter()

    # ── Stage 1: translate ────────────────────────────────────────────────
    t1 = time.perf_counter()
    translations_grouped: list[list[Translation]] = []
    if record.answer_type != AnswerType.OPEN_ENDED:
        translations_grouped = translator.translate(record)
    t.translate_s = time.perf_counter() - t1

    # ── Stage 2: Z3 ────────────────────────────────────────────────────────
    t2 = time.perf_counter()
    verdicts: list[SolverVerdict] = []
    winning_translation: Translation | None = None
    if translations_grouped:
        if record.answer_type == AnswerType.YES_NO_UNCERTAIN:
            verdicts, winning_translation = _solve_ynu(translations_grouped[0], cfg)
        elif record.answer_type == AnswerType.MCQ:
            verdicts, winning_translation = _solve_mcq(
                translations_grouped, record.options or [], cfg
            )
    t.solve_s = time.perf_counter() - t2

    # ── Stage 3a: vote ─────────────────────────────────────────────────────
    t3 = time.perf_counter()
    answer, confidence, unsat_core = aggregate(
        verdicts, k=len(verdicts) if verdicts else 0,
        high_threshold=cfg.vote_high_threshold,
        medium_threshold=cfg.vote_medium_threshold,
    )
    t.vote_s = time.perf_counter() - t3

    # ── Stage 3b: CoT fallback if needed ───────────────────────────────────
    elapsed_so_far = time.perf_counter() - t0
    budget_left = cfg.wall_clock_budget_s - elapsed_so_far
    need_fallback = (
        answer is None
        or record.answer_type == AnswerType.OPEN_ENDED
        or confidence < 0.7
    )
    final: FinalAnswer
    if need_fallback and budget_left > 5.0:
        t4 = time.perf_counter()
        cot_answer, cot_trace, cot_conf = run_cot(
            translator.backend, record, cfg.cot
        )
        t.cot_s = time.perf_counter() - t4
        if cot_answer is not None:
            final = from_cot(record, cot_answer, cot_conf, cot_trace)
        elif answer is not None:
            # CoT failed but we have a low-confidence symbolic answer — keep it.
            final = from_symbolic(record, answer, confidence, unsat_core, winning_translation)
        else:
            final = from_failure(record)
    elif answer is not None:
        final = from_symbolic(record, answer, confidence, unsat_core, winning_translation)
    else:
        final = from_failure(record)

    t.total_s = time.perf_counter() - t0
    if cfg and cfg.wall_clock_budget_s < t.total_s:
        log.warning("record %s exceeded wall clock: %.2fs > %.2fs", record.id, t.total_s,
                    cfg.wall_clock_budget_s)
    return final, t
