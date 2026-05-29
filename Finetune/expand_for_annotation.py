"""Expand the filtered dataset into a flat annotation-ready JSON.

Each output row has exactly 4 fields:
  premises-NL   : list[str]  — unchanged from source
  premises-FOL  : list[str]  — unchanged from source
  question-NL   : str        — one atomic question / MCQ option as a declarative statement
  question-FOL  : str        — EMPTY — fill this in manually

MCQ questions are split into one row per option.
Y/N questions become one row each.

A `_source` field is added (stripped at the end if you want a clean file) so the
stratified train/dev split can still keep both questions from the same record on
the same side of the split.

Usage:
    python expand_for_annotation.py
    python expand_for_annotation.py --input data/my.json --output data/my.expanded.json
"""

from __future__ import annotations

import io
import json
import re
import sys
import argparse
from pathlib import Path

# Force UTF-8 output on Windows so Unicode FOL symbols don't crash the terminal.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

INPUT  = Path(__file__).parent / "data" / "Logic_Based_Educational_Queries.filtered.json"
OUTPUT = Path(__file__).parent / "data" / "annotation_ready.json"

_MCQ_OPTION = re.compile(r"^\s*[\(\[]?([A-H])[\)\.\:\]\s]\s*(.+?)$", re.MULTILINE)


def parse_mcq(text: str) -> tuple[str, list[str]] | None:
    matches = list(_MCQ_OPTION.finditer(text))
    if len(matches) < 2:
        return None
    labels = [m.group(1) for m in matches]
    expected = [chr(ord("A") + i) for i in range(len(matches))]
    if labels != expected:
        return None
    stem = text[: matches[0].start()].rstrip()
    options = [m.group(2).strip() for m in matches]
    return stem, options


def option_as_statement(stem: str, option: str) -> str:
    opt = option.strip()
    if opt.endswith(".") or len(opt.split()) >= 5:
        return opt
    return f"{stem.rstrip('?.').strip()}: {opt}."


def expand(record: dict, record_idx: int) -> list[dict]:
    premises_nl  = record.get("premises-NL", [])
    premises_fol = record.get("premises-FOL", [])
    questions    = record.get("questions", [])
    # Support both old inline `questions` and already-split `questions-NL`
    questions_nl = record.get("questions-NL") or questions

    rows: list[dict] = []
    for q_i, q_text in enumerate(questions_nl):
        mcq = parse_mcq(q_text)
        if mcq is not None:
            stem, options = mcq
            for opt_i, opt in enumerate(options):
                rows.append({
                    "premises-NL":  premises_nl,
                    "premises-FOL": premises_fol,
                    "question-NL":  option_as_statement(stem, opt),
                    "question-FOL": "",
                    "_source": {"record": record_idx, "question": q_i, "option": opt_i},
                })
        else:
            rows.append({
                "premises-NL":  premises_nl,
                "premises-FOL": premises_fol,
                "question-NL":  q_text.strip(),
                "question-FOL": "",
                "_source": {"record": record_idx, "question": q_i, "option": None},
            })
    return rows


def _fix_mojibake(s: str) -> str:
    """Recover a string that was UTF-8 decoded as cp1252 then re-encoded as UTF-8.

    Example: ∀ (U+2200, UTF-8: E2 88 80) was read as cp1252 bytes E2→â 88→ˆ 80→€,
    giving the string 'âˆ€'. Encoding that back to cp1252 and decoding as UTF-8
    recovers the original ∀.
    """
    try:
        return s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def _repair(obj: object) -> object:
    if isinstance(obj, str):
        return _fix_mojibake(obj)
    if isinstance(obj, list):
        return [_repair(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _repair(v) for k, v in obj.items()}
    return obj


def _load_json(path: Path) -> list | dict:
    """Read JSON robustly: handles UTF-8 with or without BOM, cp1252, and
    double-encoded UTF-8 (mojibake where ∀ appears as âˆ€)."""
    raw_bytes = path.read_bytes()
    data = None
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            data = json.loads(raw_bytes.decode(enc))
            break
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    if data is None:
        raise ValueError(f"Cannot decode {path} as UTF-8 or cp1252")
    return _repair(data)


def main(input_path: Path, output_path: Path, keep_source: bool) -> None:
    raw = _load_json(input_path)
    if isinstance(raw, dict):
        for key in ("data", "records", "items", "examples"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break

    all_rows: list[dict] = []
    for i, record in enumerate(raw):
        all_rows.extend(expand(record, i))

    if not keep_source:
        for row in all_rows:
            row.pop("_source", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    mcq_rows = sum(1 for r in all_rows if r.get("_source", {}).get("option") is not None)
    yn_rows  = len(all_rows) - mcq_rows
    print(f"Source records : {len(raw)}")
    print(f"Expanded rows  : {len(all_rows)}  ({mcq_rows} MCQ options + {yn_rows} Y/N)")
    print(f"Output         : {output_path}")
    print(f"\nNext step: open {output_path.name} and fill in each empty \"question-FOL\" value.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand dataset for manual FOL annotation.")
    parser.add_argument("--input",  type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--drop-source", action="store_true",
                        help="Remove the _source metadata field from output rows.")
    args = parser.parse_args()
    main(args.input, args.output, keep_source=not args.drop_source)
