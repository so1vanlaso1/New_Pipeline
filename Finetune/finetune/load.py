"""EXACT 2026 Track 1 dataset loader.

The release JSON shipped as `Logic_Based_Educational_Queries.json` uses:

    {
      "idx": [[record_idx?], [premise_indices?]],
      "premises-FOL": ["∀x (WT(x) → O(x))", ...],   # Unicode FOL, not SMT-LIB
      "premises-NL":  ["If a Python code is well-tested, then ...", ...],
      "questions":    ["Which conclusion follows ...\nA. ...\nB. ...", ...],
      "answers":      ["A", "Yes", ...],
      "explanation":  ["...", ...],
    }

So each record bundles ONE premise set with N parallel (question, answer,
explanation) tuples. We expand into N normalized `Record` instances, one per
question. MCQ options are embedded in the question string and we split them
out via `parse_mcq_question`.

The loader stays schema-flexible (FIELD_ALIASES) so the same code handles
future releases with different field names.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .types import AnswerType, Record

# Tried in order; first present field wins.
FIELD_ALIASES: dict[str, list[str]] = {
    "id": ["id", "qid", "question_id", "record_id", "idx"],
    "premises_nl": ["premises_nl", "premises-NL", "premises", "context", "context_nl", "facts_nl"],
    "premises_fol": ["premises_fol", "premises-FOL", "fol_premises", "facts_fol", "fol"],
    "questions": ["questions", "question_nl", "question", "queries"],
    "questions_nl": ["questions-NL", "questions_nl", "question-NL", "questions_text"],
    "questions_fol": ["questions-FOL", "questions_fol", "question-FOL", "goals-FOL", "goal_fol"],
    "answers": ["answers", "answer", "label", "gold", "gold_answer"],
    "explanations": ["explanation", "explanations", "rationale", "reasoning"],
}


# ─── MCQ question parsing ────────────────────────────────────────────────

# Match an option line: starts at start-of-line with "A.", "A)", "(A)", optionally with a space.
_MCQ_OPTION_LINE = re.compile(r"^\s*[\(\[]?([A-H])[\)\.\:\]\s]\s*(.+?)$", re.MULTILINE)


def parse_mcq_question(text: str) -> tuple[str, list[str]] | None:
    """Split 'stem\\nA. opt1\\nB. opt2\\n...' into (stem, [opt1, opt2, ...]).

    Returns None if no MCQ structure is detected.
    """
    matches = list(_MCQ_OPTION_LINE.finditer(text))
    # Need at least 2 options, and the labels should be consecutive starting from A.
    if len(matches) < 2:
        return None
    labels = [m.group(1) for m in matches]
    expected = [chr(ord("A") + i) for i in range(len(matches))]
    if labels != expected:
        return None
    stem = text[: matches[0].start()].rstrip()
    options = [m.group(2).strip() for m in matches]
    return stem, options


# ─── Field picking ───────────────────────────────────────────────────────


def _first_present(obj: dict[str, Any], aliases: list[str]) -> Any:
    for k in aliases:
        if k in obj and obj[k] not in (None, "", []):
            return obj[k]
    return None


def _coerce_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        return [v]
    return [str(v)]


def _classify_answer(answer: str | None, options: list[str] | None) -> AnswerType:
    if options:
        return AnswerType.MCQ
    if isinstance(answer, str):
        norm = answer.strip().lower().rstrip(".")
        if norm in {"yes", "no", "uncertain"}:
            return AnswerType.YES_NO_UNCERTAIN
        # Single-letter answer with no options string parsed → still treat as MCQ.
        if len(norm) == 1 and norm.isalpha() and norm.upper() <= "H":
            return AnswerType.MCQ
    return AnswerType.OPEN_ENDED


def _resolve_mcq_answer(answer: str, options: list[str]) -> str:
    """Map an MCQ answer letter (or option text) to the actual option string."""
    a = answer.strip().rstrip(".")
    if len(a) == 1 and a.isalpha():
        idx = ord(a.upper()) - ord("A")
        if 0 <= idx < len(options):
            return options[idx]
    return a


def _record_id(raw_idx: Any, record_pos: int, q_pos: int) -> str:
    """Build a stable string id from `idx` (which is a list-of-lists in this release)."""
    if isinstance(raw_idx, list):
        flat: list[str] = []
        for sub in raw_idx:
            if isinstance(sub, list):
                flat.extend(str(x) for x in sub)
            else:
                flat.append(str(sub))
        if flat:
            return f"{'-'.join(flat)}_q{q_pos}"
    if isinstance(raw_idx, (str, int)):
        return f"{raw_idx}_q{q_pos}"
    return f"rec{record_pos}_q{q_pos}"


# ─── Record expansion ────────────────────────────────────────────────────


def expand_record(
    obj: dict[str, Any],
    record_pos: int,
    field_map: dict[str, str] | None = None,
) -> list[Record]:
    """Convert one raw JSON object (with N parallel questions) into N Records."""
    fm = field_map or {}

    def pick(name: str) -> Any:
        if name in fm:
            return obj.get(fm[name])
        return _first_present(obj, FIELD_ALIASES[name])

    premises_nl = _coerce_list(pick("premises_nl"))
    premises_fol = _coerce_list(pick("premises_fol")) or None
    # `questions-NL` is the new minimal-schema field for the annotated training
    # JSON; it falls back to the original `questions` (which on the release
    # encodes MCQ options inline). Either way the per-question Records carry the
    # text as `question_nl`.
    questions = _coerce_list(pick("questions_nl")) or _coerce_list(pick("questions"))
    questions_fol = _coerce_list(pick("questions_fol"))
    answers = _coerce_list(pick("answers"))
    explanations = _coerce_list(pick("explanations"))
    raw_idx = pick("id")

    if not questions:
        return []
    # Pad parallel arrays defensively.
    while len(answers) < len(questions):
        answers.append("")
    while len(explanations) < len(questions):
        explanations.append("")
    while len(questions_fol) < len(questions):
        questions_fol.append("")

    out: list[Record] = []
    for q_i, q_text in enumerate(questions):
        mcq = parse_mcq_question(q_text)
        if mcq is not None:
            stem, options = mcq
        else:
            stem, options = q_text, None

        gold = answers[q_i] or None
        atype = _classify_answer(gold, options)

        if atype == AnswerType.MCQ and options and gold:
            gold = _resolve_mcq_answer(gold, options)
        elif atype == AnswerType.YES_NO_UNCERTAIN and gold:
            # Canonicalize to "Yes" / "No" / "Uncertain".
            low = gold.strip().lower().rstrip(".")
            gold = {"yes": "Yes", "no": "No", "uncertain": "Uncertain"}.get(low, gold)

        # Per-question FOL (the new annotated field). Empty string ⇒ "not annotated".
        q_fol = questions_fol[q_i].strip() if q_i < len(questions_fol) else ""
        out.append(
            Record(
                id=_record_id(raw_idx, record_pos, q_i),
                premises_nl=premises_nl,
                premises_fol=premises_fol,
                question_nl=stem,
                question_fol=q_fol or None,
                answer_type=atype,
                answer=gold,
                options=options,
                raw={"original_record_pos": record_pos, "question_pos": q_i,
                     "explanation": explanations[q_i] if q_i < len(explanations) else None,
                     "idx": raw_idx},
            )
        )
    return out


def load_records(
    path: str | Path,
    field_map: dict[str, str] | None = None,
) -> list[Record]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("data", "records", "items", "examples"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of records, got {type(data).__name__}")

    out: list[Record] = []
    for i, obj in enumerate(data):
        out.extend(expand_record(obj, record_pos=i, field_map=field_map))
    return out


def stratified_split(
    records: list[Record],
    dev_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[Record], list[Record]]:
    """Train/dev split that preserves answer-type proportions AND keeps all
    questions from one source record on the same side of the split.
    """
    import random

    rng = random.Random(seed)
    # Group by source record so dev doesn't leak premises seen in train.
    by_source: dict[Any, list[Record]] = {}
    for r in records:
        by_source.setdefault(r.raw.get("original_record_pos", r.id), []).append(r)

    sources = list(by_source.items())
    rng.shuffle(sources)

    # Stratify by the dominant answer_type within each source group.
    from collections import Counter

    def dominant_type(rs: list[Record]) -> str:
        return Counter(r.answer_type.value for r in rs).most_common(1)[0][0]

    buckets: dict[str, list[list[Record]]] = {}
    for _, rs in sources:
        buckets.setdefault(dominant_type(rs), []).append(rs)

    train, dev = [], []
    for _, groups in buckets.items():
        n_dev_groups = max(1, int(round(len(groups) * dev_frac))) if groups else 0
        for g in groups[:n_dev_groups]:
            dev.extend(g)
        for g in groups[n_dev_groups:]:
            train.extend(g)
    rng.shuffle(train)
    rng.shuffle(dev)
    return train, dev


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
