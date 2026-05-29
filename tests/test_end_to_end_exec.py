"""End-to-end: real EXACT FOL → converter → Z3 Python program → exec → solver verdict.

These tests catch integration bugs that the per-module tests miss — things like
sort-name collisions, missing bound-var declarations, or the renderer emitting
something the safe-exec sandbox doesn't allow.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import z3

from solver.z3_runner import _exec_program, run_yes_no_uncertain, run_mcq
from translator.fol_converter import convert_premises_to_z3py


def _build_program(fol_premises: list[str], goal_fol: str | None) -> str:
    setup, premises, goal, _skipped = convert_premises_to_z3py(fol_premises, goal_fol=goal_fol)
    code = "\n".join(setup) + "\n"
    code += "premises = [\n"
    code += ",\n".join(f"    {p}" for p in premises)
    code += "\n]\n"
    code += f"goal = {goal if goal else 'True'}\n"
    return code


# ─── Round-trip exec on real dataset shapes ──────────────────────────────


def test_record_0_premises_exec_clean():
    """The first record (the Python-rules example) — converter output must exec."""
    fol = [
        "∀x (WT(x) → O(x))",
        "∀x (¬PEP8(x) → ¬WT(x))",
        "∀x (EM(x))",
        "∀x (WT(x))",
        "∀x (PEP8(x) → EM(x))",
        "∀x (WT(x) → PEP8(x))",
        "∀x (WS(x) → O(x))",
        "∀x (EM(x) → WT(x))",
        "∀x (O(x) → CR(x))",
        "∀x (WS(x))",
        "∀x (CR(x))",
        "∃x (BP(x))",
        "∃x (O(x))",
        "∀x (¬WS(x) → ¬PEP8(x))",
    ]
    code = _build_program(fol, goal_fol=None)
    ns = _exec_program(code)
    assert isinstance(ns["premises"], list)
    assert len(ns["premises"]) == len(fol)
    for p in ns["premises"]:
        assert isinstance(p, z3.BoolRef)


def test_collision_safe_with_predicate_named_U():
    """Predicate `U` is fine because the sort is `Universe`, not `U`."""
    fol = ["∀x (U(x) → S(x))", "∀x (U(x))"]
    code = _build_program(fol, goal_fol=None)
    ns = _exec_program(code)
    assert len(ns["premises"]) == 2


def test_yes_branch_via_full_pipeline_on_real_record():
    """Q2 from record 0: 'if all Python projects are well-structured, then all
    are optimized'. Premise 6 (`WS → O`) entails this directly."""
    fol = [
        "∀x (WT(x) → O(x))",
        "∀x (¬PEP8(x) → ¬WT(x))",
        "∀x (EM(x))",
        "∀x (WT(x))",
        "∀x (PEP8(x) → EM(x))",
        "∀x (WT(x) → PEP8(x))",
        "∀x (WS(x) → O(x))",
        "∀x (EM(x) → WT(x))",
        "∀x (O(x) → CR(x))",
        "∀x (WS(x))",
        "∀x (CR(x))",
        "∃x (BP(x))",
        "∃x (O(x))",
        "∀x (¬WS(x) → ¬PEP8(x))",
    ]
    goal_fol = "∀x (WS(x) → O(x))"
    code = _build_program(fol, goal_fol=goal_fol)
    v = run_yes_no_uncertain(code)
    assert v.status == "solved", v
    assert v.answer == "Yes", v


def test_mcq_unknown_when_no_option_entailed():
    """If we ask about a predicate not constrained by the premises, MCQ returns Unknown."""
    fol = ["∀x (A(x) → B(x))", "A(c)"]
    code = _build_program(fol, goal_fol=None)
    v = run_mcq(code, ["C(c)", "D(c)"])  # neither C nor D in any premise
    assert v.status == "solved"
    assert v.answer == "Unknown"


def test_mcq_picks_only_entailed_option():
    fol = ["∀x (A(x) → B(x))", "A(c)"]
    code = _build_program(fol, goal_fol=None)
    v = run_mcq(code, ["B(c)", "Not(B(c))"])
    assert v.status == "solved"
    assert v.answer == "0", v


def test_pythonic_with_arithmetic_exec_clean():
    """A record with multi-arg predicates and arithmetic from rec 20."""
    fol = [
        "ForAll(s, ForAll(m, (attendance(s,m) >= 80) -> allowed_exam(s,m)))",
        "ForAll(s, ForAll(m, (allowed_exam(s,m) ∧ completes_exam(s,m)) -> can_pass(s,m)))",
    ]
    code = _build_program(fol, goal_fol=None)
    ns = _exec_program(code)
    assert len(ns["premises"]) == 2
