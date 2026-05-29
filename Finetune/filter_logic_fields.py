"""Keep only premise and question fields from the EXACT release JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FIELDS_TO_KEEP = ("premises-FOL", "premises-NL", "questions")


def filter_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row_num, record in enumerate(records, start=1):
        missing = [field for field in FIELDS_TO_KEEP if field not in record]
        if missing:
            raise KeyError(f"Record {row_num} is missing field(s): {', '.join(missing)}")

        filtered.append({field: record[field] for field in FIELDS_TO_KEEP})
    return filtered


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "data" / "Logic_Based_Educational_Queries.json"
    default_output = script_dir / "data" / "Logic_Based_Educational_Queries.filtered.json"

    parser = argparse.ArgumentParser(
        description="Create a JSON file with only premises-FOL, premises-NL, and questions."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=default_input,
        help=f"Input JSON path. Default: {default_input}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_output,
        help=f"Output JSON path. Default: {default_output}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.input.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise TypeError("Expected the input JSON to contain a list of records.")
    if not all(isinstance(record, dict) for record in data):
        raise TypeError("Expected every item in the input JSON list to be an object.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered = filter_records(data)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(filtered, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"Wrote {len(filtered)} records to {args.output}")


if __name__ == "__main__":
    main()
