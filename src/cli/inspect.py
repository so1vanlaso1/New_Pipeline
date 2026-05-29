"""Inspect a data file: show top-level fields, the first few normalized records,
and an answer-type histogram. Use this the moment you drop the EXACT JSON
into data/ to confirm the loader auto-detected the right field names.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer
from rich import print
from rich.table import Table

from data.load import load_records

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def main(
    path: Path = typer.Argument(..., exists=True, help="Path to the JSON file"),
    show: int = typer.Option(3, help="How many records to display in full"),
) -> None:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        print(f"[bold]Top-level keys:[/bold] {list(raw.keys())}")
        for key in ("data", "records", "items", "examples"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break
    if isinstance(raw, list) and raw:
        print(f"[bold]Record count:[/bold] {len(raw)}")
        print(f"[bold]First-record keys:[/bold] {list(raw[0].keys())}")

    records = load_records(path)
    type_hist = Counter(r.answer_type.value for r in records)
    table = Table(title="Answer-type histogram")
    table.add_column("Type")
    table.add_column("Count", justify="right")
    for k, v in type_hist.most_common():
        table.add_row(k, str(v))
    print(table)

    for i, r in enumerate(records[:show]):
        print(f"\n[bold cyan]--- record {i} ---[/bold cyan]")
        print(f"id            : {r.id}")
        print(f"answer_type   : {r.answer_type.value}")
        print(f"answer        : {r.answer}")
        print(f"options       : {r.options}")
        print(f"question_nl   : {r.question_nl[:200]}")
        print(f"premises_nl   : {len(r.premises_nl)} items; first: {r.premises_nl[0][:200] if r.premises_nl else ''}")
        print(f"has FOL prem  : {bool(r.premises_fol)}{' (first: ' + r.premises_fol[0] + ')' if r.premises_fol else ''}")
        print(f"has FOL goal  : {bool(r.question_fol)}")


if __name__ == "__main__":
    app()
