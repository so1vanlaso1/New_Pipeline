"""Smoke tests for the Z3 wrapper, parser, and voter — no GPU needed."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.types import SolverVerdict
from solver.z3_runner import run_yes_no_uncertain, run_mcq
from translator.parse import parse_translator_output
from vote import aggregate


# ─── Z3 Python DSL runner ────────────────────────────────────────────────

BASIC_PROGRAM = """
U = DeclareSort('U')
passes = Function('passes', U, BoolSort())
gpa = Function('gpa', U, RealSort())
graduates = Function('graduates', U, BoolSort())
alice = Const('alice', U)
s = Const('s', U)
premises = [
    ForAll([s], Implies(And(passes(s), gpa(s) >= 2.0), graduates(s))),
    passes(alice),
    gpa(alice) == 3.5,
]
goal = graduates(alice)
"""


def test_yes_branch() -> None:
    v = run_yes_no_uncertain(BASIC_PROGRAM)
    assert v.status == "solved", v
    assert v.answer == "Yes", v


def test_uncertain_branch() -> None:
    code = BASIC_PROGRAM + "\nbob = Const('bob', U)\ngoal = graduates(bob)\n"
    v = run_yes_no_uncertain(code)
    assert v.status == "solved", v
    assert v.answer == "Uncertain", v


def test_no_branch() -> None:
    code = """
U = DeclareSort('U')
gpa = Function('gpa', U, RealSort())
merit = Function('merit', U, BoolSort())
bob = Const('bob', U)
s = Const('s', U)
premises = [
    ForAll([s], Implies(merit(s), gpa(s) >= 3.6)),
    gpa(bob) == 3.4,
]
goal = merit(bob)
"""
    v = run_yes_no_uncertain(code)
    assert v.status == "solved", v
    assert v.answer == "No", v


def test_parse_error_on_invalid_program() -> None:
    v = run_yes_no_uncertain("this is not valid python")
    assert v.status == "parse_error", v


def test_rejects_unsafe_program() -> None:
    # Import statement must be rejected by AST validator.
    v = run_yes_no_uncertain("import os\npremises = []\ngoal = Bool('x')")
    assert v.status == "parse_error", v
    assert "Import" in (v.error or "") or "disallowed" in (v.error or "").lower()


def test_rejects_attribute_access() -> None:
    v = run_yes_no_uncertain("premises = []\ngoal = (1).bit_length()")
    assert v.status == "parse_error", v


def test_unsat_core_tracking() -> None:
    v = run_yes_no_uncertain(BASIC_PROGRAM)
    assert v.status == "solved"
    assert v.unsat_core, "expected an unsat core when answer is Yes"


def test_mcq_picks_entailed_option() -> None:
    v = run_mcq(BASIC_PROGRAM, ["graduates(alice)", "Not(graduates(alice))"])
    assert v.status == "solved", v
    assert v.answer == "0", v


def test_mcq_returns_unknown_when_no_option_entailed() -> None:
    code = """
U = DeclareSort('U')
P = Function('P', U, BoolSort())
Q = Function('Q', U, BoolSort())
R = Function('R', U, BoolSort())
a = Const('a', U)
premises = [P(a)]
goal = P(a)
"""
    # None of these are entailed (premises don't say anything about Q or R).
    v = run_mcq(code, ["Q(a)", "R(a)", "And(Q(a), R(a))"])
    assert v.status == "solved", v
    assert v.answer == "Unknown", v


# ─── Translator parser ──────────────────────────────────────────────────


def test_parse_clean_z3py_tag() -> None:
    text = """Here is the translation:
<z3py>
U = DeclareSort('U')
P = Function('P', U, BoolSort())
a = Const('a', U)
premises = [P(a)]
goal = P(a)
</z3py>
"""
    code = parse_translator_output(text)
    assert code is not None
    assert "DeclareSort" in code
    assert "goal = P(a)" in code


def test_parse_fallback_python_fence() -> None:
    text = """```python
U = DeclareSort('U')
premises = []
goal = True
```"""
    code = parse_translator_output(text)
    assert code is not None
    assert "DeclareSort" in code


def test_parse_fallback_bare_program() -> None:
    # No fence, no tag — but recognizable shape.
    text = """U = DeclareSort('U')
P = Function('P', U, BoolSort())
a = Const('a', U)
premises = [P(a)]
goal = P(a)"""
    code = parse_translator_output(text)
    assert code is not None


def test_parse_garbage() -> None:
    assert parse_translator_output("just words, nothing") is None


# ─── Voter ──────────────────────────────────────────────────────────────


def _v(answer: str | None, status: str = "solved") -> SolverVerdict:
    return SolverVerdict(answer=answer, status=status)


def test_vote_high_confidence_5_of_5() -> None:
    ans, conf, _ = aggregate([_v("Yes")] * 5, k=5)
    assert ans == "Yes"
    assert conf == 0.95


def test_vote_high_confidence_4_of_5() -> None:
    ans, conf, _ = aggregate([_v("Yes")] * 4 + [_v("No")], k=5)
    assert ans == "Yes"
    assert conf == 0.95


def test_vote_medium_confidence_3_of_5() -> None:
    ans, conf, _ = aggregate(
        [_v("Yes")] * 3 + [_v("No"), _v("Uncertain")], k=5
    )
    assert ans == "Yes"
    assert conf == 0.70


def test_vote_no_majority_returns_none() -> None:
    ans, conf, _ = aggregate(
        [_v("Yes"), _v("Yes"), _v("No"), _v("No"), _v("Uncertain")], k=5
    )
    assert ans is None
    assert conf == 0.0


def test_vote_drops_failed_verdicts_and_uses_conservative_count() -> None:
    # 3 surviving "Yes" + 2 dropped → 3-of-5 ⇒ medium confidence (not 5-of-5).
    # Translation failure is treated as signal, not silently ignored.
    verdicts = [
        _v("Yes"),
        _v(None, "parse_error"),
        _v(None, "timeout"),
        _v("Yes"),
        _v("Yes"),
    ]
    ans, conf, _ = aggregate(verdicts, k=5)
    assert ans == "Yes"
    assert conf == 0.70

    # Sanity: 4 surviving "Yes" + 1 dropped → high confidence.
    verdicts2 = [_v("Yes")] * 4 + [_v(None, "parse_error")]
    ans2, conf2, _ = aggregate(verdicts2, k=5)
    assert ans2 == "Yes"
    assert conf2 == 0.95
