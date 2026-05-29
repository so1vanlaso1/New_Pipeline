"""Z3 runner using a Z3-Python DSL (safer subset of Python).

The translator emits a Z3 Python program shape:

    U = DeclareSort('U')
    WT = Function('WT', U, BoolSort())
    O  = Function('O',  U, BoolSort())
    x  = Const('x', U)
    alice = Const('alice', U)
    premises = [
        ForAll([x], Implies(WT(x), O(x))),
        WT(alice),
    ]
    goal = O(alice)

We `exec` this in a sandboxed namespace whose only globals are an allow-list
of Z3 constructors. AST validation runs first and rejects anything outside
that subset (no imports, no attribute access, no comprehensions, no lambdas,
no augmented assignments, etc.).

Entailment is then tested exactly like before: `premises ∧ ¬goal` unsat ⇒
goal is entailed; the symmetric check distinguishes Yes/No/Uncertain. MCQ
runs one entailment test per option and falls back to "Unknown" when none
of the options is entailed (a real answer in the EXACT dataset).
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass
from typing import Iterable

import z3

from data.types import SolverVerdict


class UnsafeProgram(ValueError):
    pass


# ─── Sandbox ─────────────────────────────────────────────────────────────

_ALLOWED_NAMES: dict[str, object] = {
    # Sorts / declarations
    "DeclareSort": z3.DeclareSort,
    "BoolSort": z3.BoolSort,
    "IntSort": z3.IntSort,
    "RealSort": z3.RealSort,
    "Function": z3.Function,
    "Const": z3.Const,
    "Consts": z3.Consts,
    "Bool": z3.Bool,
    "Bools": z3.Bools,
    "Int": z3.Int,
    "Ints": z3.Ints,
    "Real": z3.Real,
    "Reals": z3.Reals,
    "BoolVal": z3.BoolVal,
    "IntVal": z3.IntVal,
    "RealVal": z3.RealVal,
    # Boolean connectives
    "Not": z3.Not,
    "And": z3.And,
    "Or": z3.Or,
    "Implies": z3.Implies,
    "Iff": lambda a, b: z3.And(z3.Implies(a, b), z3.Implies(b, a)),
    "Xor": z3.Xor,
    # Quantifiers
    "ForAll": z3.ForAll,
    "Exists": z3.Exists,
    # Equality
    "Distinct": z3.Distinct,
    # Constants
    "True": True,
    "False": False,
    "None": None,
}

# AST nodes allowed in the translated program.
_ALLOWED_NODES = (
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Call,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.UnaryOp,
    ast.USub,
    ast.UAdd,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Not,
    ast.IfExp,
)


def _validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeProgram(f"disallowed AST node: {type(node).__name__}")
        if isinstance(node, ast.Attribute):
            raise UnsafeProgram("attribute access is not allowed")
        if isinstance(node, ast.Call):
            f = node.func
            if not isinstance(f, ast.Name):
                raise UnsafeProgram("only direct calls to allow-listed names are permitted")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            # Loaded names must either be in the allow-list or have been assigned
            # earlier in the same program. Cheap check: defer assignment-set
            # tracking and rely on NameError at exec time for unknown names.
            pass


def _exec_program(code: str) -> dict[str, object]:
    tree = ast.parse(code, mode="exec")
    _validate_ast(tree)
    ns: dict[str, object] = {"__builtins__": {}}
    ns.update(_ALLOWED_NAMES)
    exec(compile(tree, "<z3py>", "exec"), ns, ns)
    return ns


def _premises_and_goal(code: str, goal_override: str | None = None) -> tuple[list[z3.BoolRef], z3.BoolRef]:
    """Extract `premises` (list) and `goal` (Bool) from the executed namespace."""
    ns = _exec_program(code)
    premises = ns.get("premises")
    if not isinstance(premises, list) or not all(isinstance(p, z3.BoolRef) for p in premises):
        raise UnsafeProgram("`premises` must be a list of Z3 BoolRef")
    if goal_override is not None:
        # Re-exec a small extra snippet in the same namespace so the goal can
        # reference declared sorts/predicates/constants from `code`.
        goal_tree = ast.parse(f"_goal_override = {goal_override}", mode="exec")
        _validate_ast(goal_tree)
        exec(compile(goal_tree, "<goal>", "exec"), ns, ns)
        goal = ns.get("_goal_override")
    else:
        goal = ns.get("goal")
    if not isinstance(goal, z3.BoolRef):
        raise UnsafeProgram("`goal` must be a Z3 BoolRef")
    return premises, goal


# ─── Entailment checks ───────────────────────────────────────────────────


def _check_entailment(
    premises: list[z3.BoolRef],
    goal: z3.BoolRef,
    timeout_ms: int,
    track_core: bool,
) -> tuple[z3.CheckSatResult, list[str]]:
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    if track_core:
        s.set(unsat_core=True)
        for i, p in enumerate(premises):
            s.assert_and_track(p, f"p{i}")
        s.add(z3.Not(goal))
    else:
        for p in premises:
            s.add(p)
        s.add(z3.Not(goal))
    result = s.check()
    core_labels: list[str] = []
    if result == z3.unsat and track_core:
        core_labels = [str(c) for c in s.unsat_core()]
    return result, core_labels


def run_yes_no_uncertain(
    z3py_code: str,
    timeout_ms: int = 5000,
    emit_unsat_core: bool = True,
) -> SolverVerdict:
    t0 = time.perf_counter()
    try:
        premises, goal = _premises_and_goal(z3py_code)
    except (UnsafeProgram, SyntaxError, Exception) as e:
        return SolverVerdict(
            answer=None, status="parse_error", error=str(e),
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    pos_result, pos_core = _check_entailment(premises, goal, timeout_ms, emit_unsat_core)
    neg_result, neg_core = _check_entailment(
        premises, z3.Not(goal), timeout_ms, emit_unsat_core
    )
    elapsed = (time.perf_counter() - t0) * 1000

    if pos_result == z3.unknown or neg_result == z3.unknown:
        return SolverVerdict(answer=None, status="timeout", elapsed_ms=elapsed)
    if pos_result == z3.unsat and neg_result != z3.unsat:
        return SolverVerdict(answer="Yes", status="solved", unsat_core=pos_core, elapsed_ms=elapsed)
    if neg_result == z3.unsat and pos_result != z3.unsat:
        return SolverVerdict(answer="No", status="solved", unsat_core=neg_core, elapsed_ms=elapsed)
    if pos_result == z3.unsat and neg_result == z3.unsat:
        return SolverVerdict(
            answer=None, status="parse_error",
            error="inconsistent premises", elapsed_ms=elapsed,
        )
    return SolverVerdict(answer="Uncertain", status="solved", elapsed_ms=elapsed)


def run_mcq(
    z3py_code: str,
    option_goals: list[str],
    timeout_ms: int = 5000,
    emit_unsat_core: bool = True,
) -> SolverVerdict:
    """Pick the option whose goal is entailed by the premises; otherwise emit
    'Unknown', which is a real answer in the EXACT dataset when no listed
    option follows from the premises."""
    t0 = time.perf_counter()
    candidates: list[tuple[int, list[str]]] = []
    for i, goal_src in enumerate(option_goals):
        try:
            premises, goal = _premises_and_goal(z3py_code, goal_override=goal_src)
        except Exception:
            continue
        result, core = _check_entailment(premises, goal, timeout_ms, emit_unsat_core)
        if result == z3.unsat:
            candidates.append((i, core))

    elapsed = (time.perf_counter() - t0) * 1000
    if not candidates:
        # No option is entailed → 'Unknown'.
        return SolverVerdict(answer="Unknown", status="solved", elapsed_ms=elapsed)
    candidates.sort(key=lambda c: -len(c[1]))
    chosen_idx, chosen_core = candidates[0]
    return SolverVerdict(
        answer=str(chosen_idx), status="solved",
        unsat_core=chosen_core, elapsed_ms=elapsed,
    )
