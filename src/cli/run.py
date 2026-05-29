"""CLI: run the pipeline over a dataset split and emit submission JSON + per-question timing.

Example:
    python -m cli.run --data data/exact2026/dev.json --out artifacts/dev_predictions.json

Add --lora artifacts/translator-lora/ to use the fine-tuned NL→FOL adapter.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.progress import Progress

from data.load import load_records
from data.types import FinalAnswer
from pipeline import PipelineConfig, process_record
from translator.infer import TranslatorConfig, get_translator
from fallback.cot import CotConfig

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


@app.command()
def main(
    data: Path = typer.Option(..., exists=True, help="Path to dataset JSON"),
    out: Path = typer.Option(..., help="Where to write predictions JSON"),
    config: Path = typer.Option("configs/default.yaml", exists=True, help="YAML config"),
    lora: str = typer.Option(None, help="Override LoRA adapter path"),
    limit: int = typer.Option(0, help="Process only the first N records (0 = all)"),
    log_level: str = typer.Option("INFO"),
) -> None:
    logging.basicConfig(level=log_level)
    cfg = _load_config(config)

    records = load_records(data)
    if limit:
        records = records[:limit]
    console.print(f"[bold]Loaded[/bold] {len(records)} records from {data}")

    tcfg = TranslatorConfig(
        model=cfg["models"]["workhorse"],
        quantization=({"awq": "awq_marlin", "fp8": "fp8", "none": None}
                      .get(cfg["models"]["quantization"], "awq_marlin")),
        max_model_len=cfg["models"]["max_model_len"],
        gpu_memory_utilization=cfg["vllm"]["gpu_memory_utilization"],
        enable_lora=cfg["vllm"]["enable_lora"],
        max_lora_rank=cfg["vllm"]["max_lora_rank"],
        lora_path=lora or cfg["paths"].get("translator_lora"),
        k_samples=cfg["translator"]["k_samples"],
        temperature=cfg["translator"]["temperature"],
        top_p=cfg["translator"]["top_p"],
        max_new_tokens=cfg["translator"]["max_new_tokens"],
    )
    pcfg = PipelineConfig(
        wall_clock_budget_s=cfg["pipeline"]["wall_clock_budget_s"],
        solver_timeout_ms=cfg["solver"]["timeout_ms"],
        emit_unsat_core=cfg["solver"]["emit_unsat_core"],
        vote_high_threshold=cfg["vote"]["high_confidence_threshold"],
        vote_medium_threshold=cfg["vote"]["medium_confidence_threshold"],
        cot=CotConfig(
            k_samples=cfg["fallback"]["k_samples"],
            temperature=cfg["fallback"]["temperature"],
            top_p=cfg["fallback"]["top_p"],
            max_new_tokens=cfg["fallback"]["max_new_tokens"],
        ),
    )

    # If the user passed an explicit empty string for --lora, treat as "no LoRA".
    if not tcfg.lora_path or not Path(tcfg.lora_path).exists():
        if tcfg.lora_path:
            console.print(f"[yellow]LoRA path {tcfg.lora_path} not found — running base model[/yellow]")
        tcfg.lora_path = None

    console.print(f"[bold]Loading translator[/bold] ({tcfg.model}, q={tcfg.quantization})…")
    translator = get_translator(tcfg)

    predictions: dict[str, FinalAnswer] = {}
    latencies: dict[str, float] = {}

    out.parent.mkdir(parents=True, exist_ok=True)
    with Progress() as progress:
        task = progress.add_task("running", total=len(records))
        for r in records:
            final, timings = process_record(r, translator, pcfg)
            predictions[r.id] = final
            latencies[r.id] = timings.total_s
            progress.advance(task)
            progress.console.log(
                f"{r.id} → {final.answer!r} ({timings.total_s:.1f}s, "
                f"t={timings.translate_s:.1f} s={timings.solve_s:.2f} "
                f"v={timings.vote_s:.2f} cot={timings.cot_s:.1f})"
            )

    payload = {rid: pred.model_dump() for rid, pred in predictions.items()}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    (out.with_suffix(".latencies.json")).write_text(json.dumps(latencies, indent=2))
    console.print(f"[bold green]Wrote {len(predictions)} predictions → {out}[/bold green]")


if __name__ == "__main__":
    app()
