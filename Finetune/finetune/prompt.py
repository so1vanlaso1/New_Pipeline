"""Prompt construction for NL → Z3 Python DSL translation.

The translator emits a single block:

    <z3py>
    # Sort + predicate + constant declarations
    U = DeclareSort('U')
    WT = Function('WT', U, BoolSort())
    ...
    # Premise list
    premises = [
        ForAll([x], Implies(WT(x), O(x))),
        ...
    ]
    # Single goal (Bool-valued)
    goal = O(alice)
    </z3py>

For MCQ we call the translator once per option, rewriting the option as a
declarative statement; the goal line then captures the option's claim.
"""

from __future__ import annotations

from .types import AnswerType, Record


SYSTEM = """You are a logical translator. Convert natural-language premises and a question into a Z3 Python program that the solver can execute directly.

Rules:
1. Declare every sort, predicate, function, and constant you use. The universe sort is always `U = DeclareSort('U')`.
2. Predicates returning a truth value use `Function('P', U, ..., BoolSort())`.
3. Functions returning a number (used inside comparisons like `attendance(s,m) >= 80`) use `Function('f', U, ..., RealSort())`.
4. Constants use `Const('name', U)`; quantifier variables use `Const('x', U)` too — only their use distinguishes them.
5. Use `ForAll([x], body)`, `Exists([x], body)`, `Implies(a, b)`, `And(a, b, ...)`, `Or(a, b, ...)`, `Not(a)`.
6. Comparisons: `a == b`, `a != b`, `a >= b`, etc.
7. Build a Python list called `premises` containing one Z3 BoolRef per natural-language premise, in order.
8. Build a single `goal` (Z3 BoolRef) that captures the question.
9. Output exactly one block:
   <z3py>...code...</z3py>
   Do not output anything else. No markdown fences, no commentary.
"""


FEWSHOT_EXAMPLES = [
    {
        "premises": [
            "Every student who passes all required courses with a GPA of at least 2.0 graduates.",
            "Alice passes all required courses.",
            "Alice has a GPA of 3.5.",
        ],
        "question": "Does Alice graduate?",
        "code": (
            "U = DeclareSort('U')\n"
            "passes_required = Function('passes_required', U, BoolSort())\n"
            "gpa = Function('gpa', U, RealSort())\n"
            "graduates = Function('graduates', U, BoolSort())\n"
            "alice = Const('alice', U)\n"
            "s = Const('s', U)\n"
            "premises = [\n"
            "    ForAll([s], Implies(And(passes_required(s), gpa(s) >= 2.0), graduates(s))),\n"
            "    passes_required(alice),\n"
            "    gpa(alice) == 3.5,\n"
            "]\n"
            "goal = graduates(alice)"
        ),
    },
    {
        "premises": [
            "A student is eligible for the merit scholarship only if their GPA is at least 3.6.",
            "Bob's GPA is 3.4.",
        ],
        "question": "Is Bob eligible for the merit scholarship?",
        "code": (
            "U = DeclareSort('U')\n"
            "gpa = Function('gpa', U, RealSort())\n"
            "merit = Function('merit', U, BoolSort())\n"
            "bob = Const('bob', U)\n"
            "s = Const('s', U)\n"
            "premises = [\n"
            "    ForAll([s], Implies(merit(s), gpa(s) >= 3.6)),\n"
            "    gpa(bob) == 3.4,\n"
            "]\n"
            "goal = merit(bob)"
        ),
    },
]


def _render_example_user(ex: dict) -> str:
    prem = "\n".join(f"- {p}" for p in ex["premises"])
    return f"Premises:\n{prem}\nQuestion: {ex['question']}"


def _render_example_assistant(ex: dict) -> str:
    return f"<z3py>\n{ex['code']}\n</z3py>"


def build_user_prompt(premises: list[str], question: str) -> str:
    prem = "\n".join(f"- {p}" for p in premises)
    return f"Premises:\n{prem}\nQuestion: {question}"


def build_messages(premises: list[str], question: str, n_fewshot: int = 2) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM}]
    for ex in FEWSHOT_EXAMPLES[:n_fewshot]:
        messages.append({"role": "user", "content": _render_example_user(ex)})
        messages.append({"role": "assistant", "content": _render_example_assistant(ex)})
    messages.append({"role": "user", "content": build_user_prompt(premises, question)})
    return messages


def mcq_option_as_statement(question: str, option: str) -> str:
    opt = option.strip()
    if opt.endswith(".") or len(opt.split()) >= 5:
        return opt
    return f"{question.rstrip('?.').strip()}: {opt}."


def build_messages_for_record(r: Record, n_fewshot: int = 2) -> list[list[dict]]:
    """Return one prompt batch (Yes/No/Uncertain) or one per option (MCQ)."""
    if r.answer_type == AnswerType.YES_NO_UNCERTAIN:
        return [build_messages(r.premises_nl, r.question_nl, n_fewshot=n_fewshot)]
    if r.answer_type == AnswerType.MCQ and r.options:
        return [
            build_messages(
                r.premises_nl, mcq_option_as_statement(r.question_nl, opt), n_fewshot=n_fewshot
            )
            for opt in r.options
        ]
    return []
