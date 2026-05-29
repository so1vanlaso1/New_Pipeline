"""Convert the dataset's FOL formulas (Unicode AND Pythonic flavors) into Z3 Python DSL.

Input examples we have to handle:

    ∀x (WT(x) → O(x))
    ∀x (¬PEP8(x) → ¬WT(x))
    ∃x (S(x) ∧ E(x))
    ForAll(x, E(x) → U(x))
    ForAll(s, ForAll(m, (attendance(s,m) ≥ 80) → allowed_exam(s,m)))
    Exists(x, Professor(x) ∧ Concern(x))
    CreatesClass(John, Subject)
    AccessibleByInheritedClasses(Math) ∧ ¬AccessibleOutsideClass(Math)
    ∀x P(x) → ∀x (R(x) → S(x))
    (∀x (R(x) → S(x))) → (∃x A(x) → ∀x (¬E(x) → ¬R(x)))
    grade(s,m1) > 8.5
    m1 ≠ m2

Output is a list of Python statements + a top-level expression, e.g.:

    U = DeclareSort('U')
    WT = Function('WT', U, BoolSort())
    O = Function('O', U, BoolSort())
    x = Const('x', U)
    # expression:
    ForAll([x], Implies(WT(x), O(x)))

The renderer returns (setup_lines, expression). Multiple formulas sharing
the same predicate symbols are batched in `convert_premises_to_z3py`, which
deduplicates declarations.

Failure modes are explicit: if a formula uses something the parser doesn't
support yet (e.g. higher-order quantification, set-builder notation), the
parser raises `FolParseError` and the caller skips the record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


class FolParseError(ValueError):
    pass


# ─────────────────────────────────────────────────────────────────────────
# AST
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class Node:
    pass


@dataclass
class Var(Node):
    name: str


@dataclass
class Const(Node):
    name: str


@dataclass
class Num(Node):
    value: float
    is_int: bool


@dataclass
class App(Node):
    name: str
    args: list[Node]


@dataclass
class Not(Node):
    body: Node


@dataclass
class BinOp(Node):
    op: str  # 'and', 'or', 'implies', 'iff'
    left: Node
    right: Node


@dataclass
class Cmp(Node):
    op: str  # '=', '!=', '<', '>', '<=', '>='
    left: Node
    right: Node


@dataclass
class Arith(Node):
    op: str  # '+', '-', '*', '/'
    left: Node
    right: Node


@dataclass
class Quant(Node):
    kind: str  # 'forall', 'exists'
    vars: list[str]
    body: Node


# ─────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────


# Multi-char tokens first so the regex picks them before single-char ones.
_TOKEN_SPEC: list[tuple[str, str]] = [
    ("WS", r"[ \t\n\r]+"),
    ("NUMBER", r"\d+(?:\.\d+)?"),
    ("FORALL", r"∀|ForAll|Forall|forall|For_all|FORALL"),
    ("EXISTS", r"∃|Exists|exists|EXISTS"),
    ("NOT", r"¬|~|\bnot\b"),
    ("STRING", r"'[^']*'|\"[^\"]*\""),
    ("IFF", r"↔|<->|<=>"),
    ("IMPLIES", r"→|->|=>|\bimplies\b"),
    ("AND", r"∧|&&|/\\"),
    ("OR", r"∨|\|\||\\/"),
    ("NEQ", r"≠|!="),
    ("LEQ", r"≤|<="),
    ("GEQ", r"≥|>="),
    ("LT", r"<"),
    ("GT", r">"),
    ("EQ", r"="),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("MUL", r"\*"),
    ("DIV", r"/"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACK", r"\["),
    ("RBRACK", r"\]"),
    ("COMMA", r","),
    ("DOT", r"\."),
    ("IDENT", r"[A-Za-z_][A-Za-z_0-9]*"),
]

_TOKEN_RE = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in _TOKEN_SPEC))


@dataclass
class Token:
    kind: str
    value: str
    pos: int


def tokenize(src: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if not m:
            raise FolParseError(f"unexpected char {src[pos]!r} at {pos}")
        kind = m.lastgroup or ""
        if kind != "WS":
            tokens.append(Token(kind, m.group(), pos))
        pos = m.end()
    return tokens


# ─────────────────────────────────────────────────────────────────────────
# Parser (recursive descent, Pratt-style for precedence)
# ─────────────────────────────────────────────────────────────────────────


class Parser:
    """Grammar (precedence low → high):

        formula = iff
        iff     = implies (IFF implies)*
        implies = orx    (IMPLIES orx)*          (right-associative)
        orx     = andx   (OR andx)*
        andx    = unary  (AND unary)*
        unary   = NOT unary | quant | atom
        quant   = (FORALL|EXISTS) ('[' var (',' var)* ']' | var+) ('.'|',')? unary
                | (FORALL|EXISTS) '(' var (',' var)* ',' formula ')'
        atom    = comparison | predicate | '(' formula ')'
        comparison = term cmp_op term
        term    = atom_term ((PLUS|MINUS|MUL|DIV) atom_term)*
        atom_term = NUMBER | IDENT '(' term (',' term)* ')' | IDENT | '(' term ')' | -term
        predicate = IDENT '(' term (',' term)* ')' | IDENT     -- nullary
    """

    KEYWORDS = {"And", "Or", "Not", "Implies", "Iff", "ForAll", "Exists"}

    def __init__(self, tokens: list[Token], src: str):
        self.tokens = tokens
        self.pos = 0
        self.src = src

    def peek(self, offset: int = 0) -> Token | None:
        i = self.pos + offset
        return self.tokens[i] if i < len(self.tokens) else None

    def eat(self, kind: str) -> Token:
        t = self.peek()
        if t is None or t.kind != kind:
            raise FolParseError(
                f"expected {kind}, got {t.kind if t else 'EOF'} ({t.value if t else ''!r}) "
                f"in: {self.src!r}"
            )
        self.pos += 1
        return t

    def accept(self, *kinds: str) -> Token | None:
        t = self.peek()
        if t and t.kind in kinds:
            self.pos += 1
            return t
        return None

    # ── Top-level ────────────────────────────────────────────────────────

    def parse_formula(self) -> Node:
        return self.parse_iff()

    def parse_iff(self) -> Node:
        left = self.parse_implies()
        while self.accept("IFF"):
            right = self.parse_implies()
            left = BinOp("iff", left, right)
        return left

    def parse_implies(self) -> Node:
        # Right-associative.
        left = self.parse_or()
        if self.accept("IMPLIES"):
            right = self.parse_implies()
            return BinOp("implies", left, right)
        return left

    def parse_or(self) -> Node:
        left = self.parse_and()
        while self.accept("OR"):
            right = self.parse_and()
            left = BinOp("or", left, right)
        return left

    def parse_and(self) -> Node:
        left = self.parse_unary()
        while self.accept("AND"):
            right = self.parse_unary()
            left = BinOp("and", left, right)
        return left

    def parse_unary(self) -> Node:
        if self.accept("NOT"):
            return Not(self.parse_unary())
        if self.peek() and self.peek().kind in ("FORALL", "EXISTS"):
            return self.parse_quant()
        # Function-call style And/Or/Not/Implies/Iff(…)
        t = self.peek()
        if t and t.kind == "IDENT" and t.value in self.KEYWORDS and self._look_ahead_lparen():
            return self.parse_keyword_call()
        return self.parse_atom()

    def _look_ahead_lparen(self) -> bool:
        nxt = self.peek(1)
        return nxt is not None and nxt.kind == "LPAREN"

    def parse_keyword_call(self) -> Node:
        kw = self.eat("IDENT").value
        self.eat("LPAREN")
        args: list[Node] = []
        if not self.accept("RPAREN"):
            args.append(self.parse_formula())
            while self.accept("COMMA"):
                args.append(self.parse_formula())
            self.eat("RPAREN")
        if kw == "Not":
            if len(args) != 1:
                raise FolParseError(f"Not() expects 1 arg, got {len(args)}")
            return Not(args[0])
        if kw == "And":
            return self._fold_binop("and", args)
        if kw == "Or":
            return self._fold_binop("or", args)
        if kw == "Implies":
            if len(args) != 2:
                raise FolParseError(f"Implies expects 2 args, got {len(args)}")
            return BinOp("implies", args[0], args[1])
        if kw == "Iff":
            if len(args) != 2:
                raise FolParseError(f"Iff expects 2 args, got {len(args)}")
            return BinOp("iff", args[0], args[1])
        raise FolParseError(f"unexpected keyword {kw}")

    def _fold_binop(self, op: str, args: list[Node]) -> Node:
        if not args:
            raise FolParseError(f"empty {op}")
        if len(args) == 1:
            return args[0]
        out = args[0]
        for a in args[1:]:
            out = BinOp(op, out, a)
        return out

    # ── Quantifiers ──────────────────────────────────────────────────────

    def parse_quant(self) -> Node:
        t = self.eat(self.peek().kind)  # FORALL or EXISTS
        kind = "forall" if t.kind == "FORALL" else "exists"

        # Pythonic style: ForAll(x, body)  or  ForAll([x, y], body)
        if self.accept("LPAREN"):
            vars_: list[str] = []
            if self.accept("LBRACK"):
                vars_.append(self._eat_var_ident())
                while self.accept("COMMA"):
                    vars_.append(self._eat_var_ident())
                self.eat("RBRACK")
            else:
                vars_.append(self._eat_var_ident())
                # Allow ForAll(x1, x2, x3, body) — last is body.
                # We accumulate until we see a comma followed by a non-IDENT or a body.
                # Simpler: keep eating ident-then-comma greedily, then parse body.
                while self.accept("COMMA"):
                    # peek: is next another bare ident followed by comma? if so, it's a var.
                    t2 = self.peek()
                    t3 = self.peek(1)
                    if (
                        t2 is not None
                        and t2.kind == "IDENT"
                        and t3 is not None
                        and t3.kind == "COMMA"
                    ):
                        vars_.append(self._eat_var_ident())
                        continue
                    # Otherwise the next thing IS the body.
                    body = self.parse_formula()
                    self.eat("RPAREN")
                    return self._build_quant(kind, vars_, body)
                # If we got here, no comma — single arg form, error.
                raise FolParseError(f"{kind} expects ',' before body")
            self.eat("COMMA")
            body = self.parse_formula()
            self.eat("RPAREN")
            return self._build_quant(kind, vars_, body)

        # Unicode style: ∀x BODY  or  ∀x y z BODY  or  ∀x (BODY)
        vars_ = [self._eat_var_ident()]
        # Allow chained quantifiers: ∀x ∀y or ∀x y
        while True:
            nt = self.peek()
            if nt is None:
                break
            if nt.kind in ("FORALL", "EXISTS"):
                # Defer: outer quantifier ends here; the next one parses recursively.
                body = self.parse_unary()
                return self._build_quant(kind, vars_, body)
            if nt.kind == "IDENT" and self._is_bare_var_chain(nt):
                # Heuristic: a bare ident with no following '(' or operator is another var.
                vars_.append(self._eat_var_ident())
                continue
            break
        # Optional separator after vars: `.` or `,` (the latter appears in
        # mixed-flavor records like `∃d, has_degree(x, d) ∧ ...`).
        self.accept("DOT") or self.accept("COMMA")
        body = self.parse_unary()
        return self._build_quant(kind, vars_, body)

    def _is_bare_var_chain(self, t: Token) -> bool:
        """Heuristic: token at peek() is a single-letter ident with no `(` after."""
        if len(t.value) > 2:
            return False
        nxt = self.peek(1)
        return nxt is None or nxt.kind not in ("LPAREN", "LBRACK")

    def _build_quant(self, kind: str, vars_: list[str], body: Node) -> Node:
        # Pull off nested same-kind quantifiers into a single binder when convenient.
        if isinstance(body, Quant) and body.kind == kind:
            return Quant(kind, vars_ + body.vars, body.body)
        return Quant(kind, vars_, body)

    def _eat_var_ident(self) -> str:
        t = self.eat("IDENT")
        return t.value

    # ── Atoms / terms ────────────────────────────────────────────────────

    def parse_atom(self) -> Node:
        # Try comparison; if next sequence is a term-then-cmp-then-term, treat as Cmp.
        save = self.pos
        try:
            left = self.parse_term()
            t = self.peek()
            if t and t.kind in ("EQ", "NEQ", "LT", "GT", "LEQ", "GEQ"):
                op_map = {"EQ": "==", "NEQ": "!=", "LT": "<", "GT": ">", "LEQ": "<=", "GEQ": ">="}
                op = op_map[t.kind]
                self.pos += 1
                right = self.parse_term()
                return Cmp(op, left, right)
            # No comparison ⇒ the term itself must be a Boolean-valued atom (predicate
            # application or nullary predicate). A bare identifier (case-insensitive)
            # is promoted to a nullary predicate — signature collection elsewhere
            # already handles the bound-var-vs-nullary disambiguation.
            if isinstance(left, App):
                return left
            if isinstance(left, (Const, Var)):
                return App(left.name, [])
            raise FolParseError(f"bare term in boolean position: {left}")
        except FolParseError:
            self.pos = save
            # Maybe a parenthesized formula.
            if self.accept("LPAREN"):
                body = self.parse_formula()
                self.eat("RPAREN")
                return body
            raise

    def parse_term(self) -> Node:
        return self._parse_term_addsub()

    def _parse_term_addsub(self) -> Node:
        left = self._parse_term_muldiv()
        while True:
            t = self.peek()
            if t and t.kind in ("PLUS", "MINUS"):
                op = "+" if t.kind == "PLUS" else "-"
                self.pos += 1
                right = self._parse_term_muldiv()
                left = Arith(op, left, right)
            else:
                return left

    def _parse_term_muldiv(self) -> Node:
        left = self._parse_term_unary()
        while True:
            t = self.peek()
            if t and t.kind in ("MUL", "DIV"):
                op = "*" if t.kind == "MUL" else "/"
                self.pos += 1
                right = self._parse_term_unary()
                left = Arith(op, left, right)
            else:
                return left

    def _parse_term_unary(self) -> Node:
        if self.accept("MINUS"):
            inner = self._parse_term_unary()
            return Arith("-", Num(0, True), inner)
        return self._parse_term_atom()

    def _parse_term_atom(self) -> Node:
        t = self.peek()
        if t is None:
            raise FolParseError("unexpected EOF in term")
        if t.kind == "NUMBER":
            self.pos += 1
            is_int = "." not in t.value
            return Num(float(t.value), is_int)
        if t.kind == "STRING":
            # 'PoliticalIdeologies' → constant str_PoliticalIdeologies (sanitized).
            self.pos += 1
            raw = t.value[1:-1]
            safe = re.sub(r"[^A-Za-z0-9_]", "_", raw)
            return Const(f"str_{safe}")
        if t.kind == "LPAREN":
            self.pos += 1
            inner = self._parse_term_addsub()
            self.eat("RPAREN")
            return inner
        if t.kind == "IDENT":
            name = t.value
            self.pos += 1
            if self.accept("LPAREN"):
                args: list[Node] = []
                if not self.accept("RPAREN"):
                    args.append(self._parse_term_addsub())
                    while self.accept("COMMA"):
                        args.append(self._parse_term_addsub())
                    self.eat("RPAREN")
                return App(name, args)
            # Heuristic: lowercase identifier ⇒ variable; capitalized ⇒ constant.
            if name and name[0].islower():
                return Var(name)
            return Const(name)
        raise FolParseError(f"unexpected token {t.kind} {t.value!r} in term")


def parse(src: str) -> Node:
    src = src.strip()
    if not src:
        raise FolParseError("empty input")
    tokens = tokenize(src)
    p = Parser(tokens, src)
    node = p.parse_formula()
    if p.pos < len(tokens):
        raise FolParseError(
            f"trailing tokens after parse: {[t.value for t in tokens[p.pos:]]!r}"
        )
    return node


# ─────────────────────────────────────────────────────────────────────────
# Signature collection (to emit Z3 declarations)
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class Signature:
    # `U` collides with common single-letter predicate names in the dataset
    # (rec 0 already has a predicate named `U`). `Universe` is unambiguous.
    sort_name: str = "Universe"
    # name → arity (predicates always Bool-valued; arithmetic functions handled separately)
    predicates: dict[str, int] = field(default_factory=dict)
    # name → arity (returns U; used for things like attendance(s,m))
    int_functions: dict[str, int] = field(default_factory=dict)
    # set of bare constants
    constants: set[str] = field(default_factory=set)
    # set of bound variable names (must NOT be declared as constants)
    bound_vars: set[str] = field(default_factory=set)

    def merge(self, other: "Signature") -> None:
        for k, v in other.predicates.items():
            if k in self.predicates and self.predicates[k] != v:
                raise FolParseError(
                    f"predicate {k} used with arity {self.predicates[k]} and {v}"
                )
            self.predicates[k] = v
        for k, v in other.int_functions.items():
            if k in self.int_functions and self.int_functions[k] != v:
                raise FolParseError(
                    f"function {k} used with arity {self.int_functions[k]} and {v}"
                )
            self.int_functions[k] = v
        self.constants |= other.constants
        self.bound_vars |= other.bound_vars


def _collect_in_term(node: Node, sig: Signature, locals_: set[str]) -> bool:
    """Walk a term subtree, recording signatures. Returns True if any sub-term
    is "arithmetic-flavored" (number, comparison input, function returning Int).
    """
    if isinstance(node, Var):
        return False
    if isinstance(node, Const):
        if node.name not in locals_:
            sig.constants.add(node.name)
        return False
    if isinstance(node, Num):
        return True
    if isinstance(node, App):
        arity = len(node.args)
        any_arith = any(_collect_in_term(a, sig, locals_) for a in node.args)
        # If it's being used as an Int-valued function (i.e. inside a Cmp or Arith),
        # we record it as an int_function. We can't tell here in isolation, so default to
        # predicate; the Cmp handler upgrades it.
        if any_arith:
            sig.int_functions[node.name] = arity
        else:
            # Default: predicate. May be overwritten by Cmp/Arith context.
            sig.predicates.setdefault(node.name, arity)
        return any_arith
    if isinstance(node, Arith):
        l = _collect_in_term(node.left, sig, locals_)
        r = _collect_in_term(node.right, sig, locals_)
        return True
    if isinstance(node, Cmp):
        _collect_in_term(node.left, sig, locals_)
        _collect_in_term(node.right, sig, locals_)
        return False
    raise FolParseError(f"term position holds non-term: {node}")


def collect_signature(node: Node, sig: Signature | None = None,
                      locals_: set[str] | None = None) -> Signature:
    sig = sig or Signature()
    locals_ = locals_ or set()
    if isinstance(node, Quant):
        new_locals = locals_ | set(node.vars)
        sig.bound_vars |= set(node.vars)
        collect_signature(node.body, sig, new_locals)
        return sig
    if isinstance(node, Not):
        collect_signature(node.body, sig, locals_)
        return sig
    if isinstance(node, BinOp):
        collect_signature(node.left, sig, locals_)
        collect_signature(node.right, sig, locals_)
        return sig
    if isinstance(node, Cmp):
        # Both sides are Int-valued terms. Upgrade any App on either side
        # from predicate-default to int_function.
        for side in (node.left, node.right):
            _promote_apps_to_int(side, sig, locals_)
        return sig
    if isinstance(node, App):
        # Predicate position.
        arity = len(node.args)
        existing = sig.predicates.get(node.name)
        if existing is None:
            sig.predicates[node.name] = arity
        for a in node.args:
            _collect_in_term(a, sig, locals_)
        return sig
    if isinstance(node, Const):
        if node.name not in locals_:
            sig.constants.add(node.name)
        return sig
    if isinstance(node, Arith):
        # Top-level Arith in boolean position is meaningless; let caller deal.
        return sig
    raise FolParseError(f"unsupported node: {node}")


def _promote_apps_to_int(node: Node, sig: Signature, locals_: set[str]) -> None:
    if isinstance(node, App):
        arity = len(node.args)
        sig.int_functions[node.name] = arity
        sig.predicates.pop(node.name, None)
        for a in node.args:
            _collect_in_term(a, sig, locals_)
        return
    if isinstance(node, Arith):
        _promote_apps_to_int(node.left, sig, locals_)
        _promote_apps_to_int(node.right, sig, locals_)
        return
    if isinstance(node, (Const, Var, Num)):
        _collect_in_term(node, sig, locals_)
        return


# ─────────────────────────────────────────────────────────────────────────
# Renderer → Z3 Python DSL
# ─────────────────────────────────────────────────────────────────────────


def render_expr(node: Node, sig: Signature) -> str:
    if isinstance(node, Var):
        return node.name
    if isinstance(node, Const):
        return node.name
    if isinstance(node, Num):
        return repr(int(node.value)) if node.is_int else repr(node.value)
    if isinstance(node, App):
        if node.args:
            return f"{node.name}(" + ", ".join(render_expr(a, sig) for a in node.args) + ")"
        return node.name
    if isinstance(node, Not):
        return f"Not({render_expr(node.body, sig)})"
    if isinstance(node, BinOp):
        opmap = {"and": "And", "or": "Or", "implies": "Implies", "iff": "Iff"}
        return f"{opmap[node.op]}({render_expr(node.left, sig)}, {render_expr(node.right, sig)})"
    if isinstance(node, Cmp):
        return f"({render_expr(node.left, sig)} {node.op} {render_expr(node.right, sig)})"
    if isinstance(node, Arith):
        return f"({render_expr(node.left, sig)} {node.op} {render_expr(node.right, sig)})"
    if isinstance(node, Quant):
        kind = "ForAll" if node.kind == "forall" else "Exists"
        vars_ = ", ".join(node.vars)
        return f"{kind}([{vars_}], {render_expr(node.body, sig)})"
    raise FolParseError(f"unrenderable node: {node}")


def render_setup(sig: Signature) -> list[str]:
    """Emit Z3 Python declarations for the collected signature."""
    lines: list[str] = []
    sort = sig.sort_name
    lines.append(f"{sort} = DeclareSort('{sort}')")
    # Predicates: Function(name, U, U, ..., BoolSort())
    for name, arity in sorted(sig.predicates.items()):
        if arity == 0:
            lines.append(f"{name} = Const('{name}', BoolSort())")
        else:
            args = ", ".join([sort] * arity)
            lines.append(f"{name} = Function('{name}', {args}, BoolSort())")
    # Int-valued functions: returning IntSort (heuristic; handles `grade(s,m) > 8.5` as Int comparison).
    for name, arity in sorted(sig.int_functions.items()):
        if arity == 0:
            lines.append(f"{name} = Const('{name}', RealSort())")
        else:
            args = ", ".join([sort] * arity)
            lines.append(f"{name} = Function('{name}', {args}, RealSort())")
    # Bound quantifier variables also need a `Const` declaration so that
    # `ForAll([x], …)` can reference `x`. Constants and bound vars are declared
    # the same way (both are Z3 Const objects in the namespace).
    decl_names = sorted(sig.constants | sig.bound_vars)
    for c in decl_names:
        lines.append(f"{c} = Const('{c}', {sort})")
    return lines


def collect_free_vars(node: Node, bound: set[str] | None = None) -> set[str]:
    """Find variables that are referenced but never bound by a quantifier."""
    bound = bound or set()
    if isinstance(node, Var):
        return set() if node.name in bound else {node.name}
    if isinstance(node, (Const, Num)):
        return set()
    if isinstance(node, Quant):
        return collect_free_vars(node.body, bound | set(node.vars))
    if isinstance(node, (Not,)):
        return collect_free_vars(node.body, bound)
    if isinstance(node, (BinOp, Cmp, Arith)):
        return collect_free_vars(node.left, bound) | collect_free_vars(node.right, bound)
    if isinstance(node, App):
        out: set[str] = set()
        for a in node.args:
            out |= collect_free_vars(a, bound)
        return out
    return set()


# ─────────────────────────────────────────────────────────────────────────
# Top-level: convert a list of FOL formulas into a full Z3 Python program
# ─────────────────────────────────────────────────────────────────────────


def convert_premises_to_z3py(
    fol_premises: Iterable[str],
    goal_fol: str | None = None,
) -> tuple[list[str], list[str], str | None, list[int]]:
    """Convert a batch of FOL formulas (and an optional goal) to a Z3 Python
    program shape.

    Returns (setup_lines, premise_exprs, goal_expr_or_None, skipped_indices).
    Each premise_expr is a Z3 Python expression string suitable for
    `premises = [<expr_0>, <expr_1>, ...]`.

    Records that fail to parse are skipped — their indices are returned.
    """
    sig = Signature()
    premise_nodes: list[Node | None] = []
    skipped: list[int] = []
    for i, p in enumerate(fol_premises):
        try:
            node = parse(p)
            collect_signature(node, sig)
            premise_nodes.append(node)
        except FolParseError:
            skipped.append(i)
            premise_nodes.append(None)

    goal_node: Node | None = None
    if goal_fol is not None:
        try:
            goal_node = parse(goal_fol)
            collect_signature(goal_node, sig)
        except FolParseError:
            goal_node = None

    # If any premise contains a free variable that wasn't bound, declare it as
    # a fresh constant so the program still type-checks. (Some dataset items
    # write `∀x P(x)` correctly but others write `P(x)` at top level.)
    extra_free: set[str] = set()
    for n in premise_nodes:
        if n is not None:
            extra_free |= collect_free_vars(n)
    if goal_node is not None:
        extra_free |= collect_free_vars(goal_node)
    for v in extra_free:
        sig.constants.add(v)

    setup = render_setup(sig)
    premise_exprs = [render_expr(n, sig) for n in premise_nodes if n is not None]
    goal_expr = render_expr(goal_node, sig) if goal_node is not None else None
    return setup, premise_exprs, goal_expr, skipped
