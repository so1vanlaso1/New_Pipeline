"""Smoke tests for the FOL → Z3 Python DSL converter on real EXACT data.

These hit a range of syntactic shapes we saw in `Logic_Based_Educational_Queries.json`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from translator.fol_converter import (
    FolParseError,
    collect_signature,
    convert_premises_to_z3py,
    parse,
    render_expr,
    render_setup,
)


def _convert_one(formula: str) -> tuple[list[str], str]:
    """Helper: parse + collect + render a single formula."""
    node = parse(formula)
    from translator.fol_converter import Signature

    sig = Signature()
    collect_signature(node, sig)
    return render_setup(sig), render_expr(node, sig)


# ─── Unicode style ───────────────────────────────────────────────────────


def test_simple_unary_implication():
    setup, expr = _convert_one("∀x (WT(x) → O(x))")
    assert "Universe = DeclareSort('Universe')" in setup
    assert "WT = Function('WT', Universe, BoolSort())" in setup
    assert "O = Function('O', Universe, BoolSort())" in setup
    assert expr == "ForAll([x], Implies(WT(x), O(x)))"


def test_double_negation_implication():
    _, expr = _convert_one("∀x (¬PEP8(x) → ¬WT(x))")
    assert expr == "ForAll([x], Implies(Not(PEP8(x)), Not(WT(x))))"


def test_simple_existence():
    _, expr = _convert_one("∃x (S(x) ∧ E(x))")
    assert expr == "Exists([x], And(S(x), E(x)))"


def test_bare_universal_predicate():
    _, expr = _convert_one("∀x (EM(x))")
    assert expr == "ForAll([x], EM(x))"


# ─── Pythonic style ──────────────────────────────────────────────────────


def test_pythonic_forall_simple():
    _, expr = _convert_one("ForAll(x, E(x) → U(x))")
    assert expr == "ForAll([x], Implies(E(x), U(x)))"


def test_pythonic_exists():
    _, expr = _convert_one("Exists(x, Professor(x) ∧ Concern(x))")
    assert expr == "Exists([x], And(Professor(x), Concern(x)))"


def test_pythonic_nested_quantifier():
    _, expr = _convert_one(
        "ForAll(s, ForAll(m, allowed_exam(s,m) → can_pass(s,m)))"
    )
    # Inner same-kind quantifiers should flatten into one binder.
    assert expr == "ForAll([s, m], Implies(allowed_exam(s, m), can_pass(s, m)))"


# ─── Multi-arg + constants + arithmetic ─────────────────────────────────


def test_arithmetic_comparison_promotes_function_to_int():
    setup, expr = _convert_one(
        "ForAll(s, ForAll(m, (attendance(s,m) ≥ 80) → allowed_exam(s,m)))"
    )
    # attendance should become a Real-valued function (it's compared to 80).
    assert any("attendance = Function('attendance', Universe, Universe, RealSort())" in s for s in setup)
    # allowed_exam stays a predicate.
    assert any("allowed_exam = Function('allowed_exam', Universe, Universe, BoolSort())" in s for s in setup)
    assert "(attendance(s, m) >= 80)" in expr
    assert "Implies(" in expr


def test_named_constants_get_declared():
    setup, expr = _convert_one("CreatesClass(John, Subject)")
    # John, Subject are uppercase identifiers used as predicate args ⇒ constants.
    assert any("John = Const('John', Universe)" in s for s in setup)
    assert any("Subject = Const('Subject', Universe)" in s for s in setup)
    assert any("CreatesClass = Function('CreatesClass', Universe, Universe, BoolSort())" in s for s in setup)
    assert expr == "CreatesClass(John, Subject)"


def test_inequality_on_variables():
    # m1 ≠ m2 — both bound by some outer quantifier elsewhere; in isolation we
    # treat them as free vars that the renderer demotes to constants.
    _, expr = _convert_one("m1 ≠ m2")
    assert expr == "(m1 != m2)"


# ─── Multi-formula batch ────────────────────────────────────────────────


def test_convert_batch_shares_signature():
    setup, premises, goal, skipped = convert_premises_to_z3py(
        [
            "∀x (WT(x) → O(x))",
            "∀x (¬PEP8(x) → ¬WT(x))",
            "∀x (EM(x))",
            "∀x (WT(x))",
            "∀x (PEP8(x) → EM(x))",
        ],
        goal_fol="∀x (O(x))",
    )
    assert skipped == []
    assert len(premises) == 5
    assert goal == "ForAll([x], O(x))"
    # Each predicate declared exactly once.
    decls = [s for s in setup if "Function(" in s]
    names = [d.split(" = ")[0] for d in decls]
    assert sorted(names) == sorted(set(names))


def test_skips_unparseable():
    # The 2nd item is a higher-order construct we don't try to handle.
    setup, premises, goal, skipped = convert_premises_to_z3py(
        [
            "∀x (P(x) → Q(x))",
            "∀x P(x) → ∀x (R(x) → S(x))",  # top-level implication between two quant forms — supported
            "this is not FOL at all{{}}",
        ],
    )
    # 3rd should be skipped.
    assert 2 in skipped
    assert len(premises) == 2  # first two parsed
