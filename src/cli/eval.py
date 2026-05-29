"""Score predictions against ground truth and print a per-answer-type breakdown.

    python -m cli.eval --data data/exact2026/dev.json --pred artifacts/dev_predictions.json
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from data.load import load_records
from data.types import FinalAnswer
from eval.score import score, write_report

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def main(
    data: Path = typer.Option(..., exists=True),
    pred: Path = typer.Option(..., exists=True),
    out: Path = typer.Option("artifacts/eval_results.json"),
) -> None:
    records = load_records(data)
    raw_preds = json.loads(pred.read_text(encoding="utf-8"))
    predictions = {rid: FinalAnswer.model_validate(obj) for rid, obj in raw_preds.items()}

    latencies = {}
    lat_path = pred.with_suffix(".latencies.json")
    if lat_path.exists():
        latencies = json.loads(lat_path.read_text())

    report = score(records, predictions, latencies_s=latencies)

    table = Table(title="P1 accuracy")
    table.add_column("Slice")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_row("overall", str(report.overall.correct), str(report.overall.total),
                  f"{report.overall.accuracy:.3f}")
    for k, v in report.by_type.items():
        table.add_row(k, str(v.correct), str(v.total), f"{v.accuracy:.3f}")
    console.print(table)

    lat = report.latency_summary()
    if lat:
        console.print(f"\n[bold]Latency (s):[/bold] mean={lat['mean']:.2f}  "
                      f"p50={lat['p50']:.2f}  p95={lat['p95']:.2f}  "
                      f"p99={lat['p99']:.2f}  max={lat['max']:.2f}")
    if report.missing:
        console.print(f"[yellow]Missing predictions: {report.missing}[/yellow]")

    write_report(report, out)
    console.print(f"[bold green]Wrote report → {out}[/bold green]")


if __name__ == "__main__":
    app()
