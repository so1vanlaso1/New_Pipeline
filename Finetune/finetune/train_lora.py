"""LoRA fine-tune of Qwen3.5-4B (https://huggingface.co/Qwen/Qwen3.5-4B) for the NL → Z3-Python task.

Standard TRL `SFTTrainer` path. Swap to Unsloth later for ~2× speed if needed
(the data plumbing is identical; only `from unsloth import FastLanguageModel`
changes).

Assumes the dataset has parallel NL and FOL premises and a parallel question.
If your release's FOL field is not SMT-LIB-compatible (e.g. uses Prolog or a
custom syntax), edit `_fol_to_smtlib` below — that's the only adapter point.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import typer
from rich import print

from .load import load_records
from .types import AnswerType, Record
from .fol_converter import FolParseError, convert_premises_to_z3py, parse
from .prompt import (
    SYSTEM,
    build_messages,
    mcq_option_as_statement,
)

app = typer.Typer(no_args_is_help=True, add_completion=False)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────


def _build_z3py_program(record: Record) -> tuple[str, bool] | None:
    """Convert a record's premises (+ optional goal) FOL into a Z3 Python
    training target.

    Returns `(code, has_goal_fol)` or None if the record is unusable.
    `has_goal_fol` is True when the record had a `questions-FOL` entry that the
    converter rendered into the program; False when we fell back to the
    `goal = True` placeholder. Callers can filter out placeholder rows for a
    cleaner training signal.
    """
    if not record.premises_fol:
        return None
    setup, premises, goal_expr, skipped = convert_premises_to_z3py(
        record.premises_fol, goal_fol=record.question_fol
    )
    # If too much of this record's FOL failed to parse, drop it.
    if len(skipped) > max(1, len(record.premises_fol) // 4):
        return None
    if not premises:
        return None
    code = "\n".join(setup)
    code += "\npremises = [\n"
    code += ",\n".join(f"    {p}" for p in premises)
    code += "\n]\n"
    has_goal = goal_expr is not None
    code += f"goal = {goal_expr}" if has_goal else "goal = True  # no goal FOL annotated"
    return code, has_goal


def _to_chat_example(record: Record) -> tuple[dict, bool] | None:
    """Return `(chat_row, has_goal_fol)` or None if the record is unusable."""
    result = _build_z3py_program(record)
    if result is None:
        return None
    code, has_goal = result
    messages = build_messages(record.premises_nl, record.question_nl, n_fewshot=2)
    assistant = f"<z3py>\n{code}\n</z3py>"
    messages.append({"role": "assistant", "content": assistant})
    return {"messages": messages}, has_goal


def build_chat_dataset(
    records: Iterable[Record],
    require_goal_fol: bool = False,
) -> list[dict]:
    """Build the supervised chat dataset.

    `require_goal_fol=True` drops every record that doesn't have an annotated
    `questions-FOL` value — use this once you've finished manual annotation
    so the LoRA only trains on rows with full supervision (premises AND goal).
    Default False keeps placeholder rows in, so partial annotations still
    train a useful LoRA on the premise-translation half.
    """
    out: list[dict] = []
    skipped_no_fol = 0
    skipped_parse = 0
    skipped_no_goal = 0
    with_goal = 0
    for r in records:
        if not r.premises_fol:
            skipped_no_fol += 1
            continue
        result = _to_chat_example(r)
        if result is None:
            skipped_parse += 1
            continue
        row, has_goal = result
        if require_goal_fol and not has_goal:
            skipped_no_goal += 1
            continue
        if has_goal:
            with_goal += 1
        out.append(row)
    log.info(
        "built %d training rows (%d with annotated goal FOL, %d placeholder; "
        "skipped %d for missing premise FOL, %d for parse failure, %d for missing goal FOL)",
        len(out), with_goal, len(out) - with_goal,
        skipped_no_fol, skipped_parse, skipped_no_goal,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    # Defaults target Qwen3.5-4B (https://huggingface.co/Qwen/Qwen3.5-4B) on an
    # RTX 5070 (12 GB). bf16 is supported on Blackwell; 4-bit base + LoRA fits
    # comfortably with batch=2 and grad_accum=8 (effective batch 16).
    base_model: str = "Qwen/Qwen3.5-4B"
    out_dir: str = "artifacts/translator-lora"
    epochs: float = 3.0
    lr: float = 2e-4
    batch_size: int = 2          # was 4 on a 4090; halved for the 5070's 12 GB
    grad_accum: int = 8          # bumped to preserve effective batch size
    max_seq_len: int = 2048      # was 4096; trim to keep activations + KV cache in budget
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    warmup_ratio: float = 0.03
    use_4bit: bool = True
    # Memory-safety knobs for the 5070's 12 GB. Both default to ON; flip them
    # off on a 4090/A100 if you want faster wall time.
    gradient_checkpointing: bool = True
    attn_implementation: str = "flash_attention_2"
    # When True, drop records that don't have an annotated `questions-FOL`
    # entry. Use this once you've finished manual goal annotation so the LoRA
    # only sees rows with full supervision.
    require_goal_fol: bool = False
    seed: int = 42


def train(
    train_records: list[Record],
    eval_records: list[Record] | None,
    cfg: TrainConfig,
) -> None:
    # Lazy imports so the package stays importable on a CPU-only dev box.
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant_kwargs = {}
    if cfg.use_4bit:
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Flash Attention 2 is essentially mandatory on a 12 GB card: attention
    # scores without FA2 are O(seq²) per layer and OOM the 5070 at seq=2048.
    # If your install can't load FA2 (e.g. bitsandbytes/torch/triton version
    # skew on Blackwell), set `attn_implementation="sdpa"` instead — SDPA is
    # the second-best memory-efficient backend.
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=cfg.attn_implementation,
        **quant_kwargs,
    )

    # Gradient checkpointing trades ~25% compute for a large activation-memory
    # cut. On 12 GB it's basically required; turn it off only on bigger cards.
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    train_data = Dataset.from_list(
        build_chat_dataset(train_records, require_goal_fol=cfg.require_goal_fol)
    )
    eval_data = (
        Dataset.from_list(
            build_chat_dataset(eval_records, require_goal_fol=cfg.require_goal_fol)
        )
        if eval_records else None
    )

    args = SFTConfig(
        output_dir=cfg.out_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy=("epoch" if eval_data else "no"),
        max_seq_length=cfg.max_seq_len,
        packing=False,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit",  # 8-bit Adam states halve optimizer VRAM (~200 MB saved)
        seed=cfg.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(cfg.out_dir)
    log.info("Saved LoRA adapter → %s", cfg.out_dir)


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────


@app.command()
def main(
    train_path: Path = typer.Option(..., "--train", exists=True),
    dev_path: Path = typer.Option(None, "--dev"),
    out: Path = typer.Option("artifacts/translator-lora"),
    epochs: float = typer.Option(3.0),
    lr: float = typer.Option(2e-4),
    batch_size: int = typer.Option(2),     # 5070-sized; raise on bigger GPUs
    grad_accum: int = typer.Option(8),
    max_seq_len: int = typer.Option(2048),
    lora_r: int = typer.Option(32),
    gradient_checkpointing: bool = typer.Option(
        True, "--gradient-checkpointing/--no-gradient-checkpointing",
        help="Trade ~25% compute for activation-memory savings. Required on 12 GB.",
    ),
    attn_impl: str = typer.Option(
        "flash_attention_2", "--attn",
        help="flash_attention_2 (recommended), sdpa (fallback), or eager.",
    ),
    require_goal_fol: bool = typer.Option(
        False, "--require-goal-fol/--allow-placeholder-goal",
        help="Drop records without an annotated questions-FOL. Use after full annotation.",
    ),
) -> None:
    logging.basicConfig(level="INFO")
    train_records = load_records(train_path)
    eval_records = load_records(dev_path) if dev_path else None
    print(f"[bold]train={len(train_records)}  eval={len(eval_records) if eval_records else 0}[/bold]")

    cfg = TrainConfig(
        out_dir=str(out),
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_seq_len=max_seq_len,
        lora_r=lora_r,
        gradient_checkpointing=gradient_checkpointing,
        attn_implementation=attn_impl,
        require_goal_fol=require_goal_fol,
    )
    train(train_records, eval_records, cfg)


if __name__ == "__main__":
    app()
