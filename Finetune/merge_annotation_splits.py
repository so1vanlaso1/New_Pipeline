"""Merge annotation split JSON files into one JSON array."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PART_NUMBER_RE = re.compile(r"_part_(\d+)")


def part_sort_key(path: Path) -> tuple[int, str]:
    match = PART_NUMBER_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot find part number in filename: {path.name}")
    return int(match.group(1)), path.name


def read_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise TypeError(f"Expected {path} to contain a JSON array.")
    if not all(isinstance(item, dict) for item in data):
        raise TypeError(f"Expected every item in {path} to be a JSON object.")

    return data


def merge_files(files: list[Path]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for path in files:
        merged.extend(read_json_array(path))
    return merged


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input_dir = script_dir / "data" / "annotation_ready_splits"
    default_output = script_dir / "data" / "annotation_ready_merged.json"

    parser = argparse.ArgumentParser(
        description="Merge annotation split JSON array files into one JSON file."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help=f"Directory containing split files. Default: {default_input_dir}",
    )
    parser.add_argument(
        "--pattern",
        default="annotation_ready_part_*_FIXED.json",
        help="Glob pattern for split files. Default: annotation_ready_part_*_FIXED.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_output,
        help=f"Output JSON path. Default: {default_output}",
    )
    parser.add_argument(
        "--expected-files",
        type=int,
        default=40,
        help="Expected number of split files. Use 0 to disable this check. Default: 40",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    files = sorted(args.input_dir.glob(args.pattern), key=part_sort_key)
    if args.expected_files and len(files) != args.expected_files:
        raise ValueError(
            f"Expected {args.expected_files} files matching {args.pattern!r} in "
            f"{args.input_dir}, found {len(files)}."
        )
    if not files:
        raise FileNotFoundError(
            f"No files matching {args.pattern!r} found in {args.input_dir}."
        )

    merged = merge_files(files)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\r\n") as fh:
        json.dump(merged, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"Merged {len(files)} files and {len(merged)} objects into {args.output}")


if __name__ == "__main__":
    main()
