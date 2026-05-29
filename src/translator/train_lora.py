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

from data.load import load_records
from data.types import AnswerType, Record
from translator.fol_converter import FolParseError, convert_premises_to_z3py, parse
from translator.prompt import (
    SYSTEM,
    build_messages,
    mcq_option_as_statement,
)

app = typer.Typer(no_args_is_help=True, add_completion=False)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────


def _build_z3py_program(record: Record) -> str | None:
    """Convert a record's `premises_fol` (Unicode/Pythonic FOL) into a Z3 Python
    program suitable as a translator training target.

    The release's FOL field has premises but no per-question goal, so we
    emit a placeholder `goal = True` line — at training time the model
    learns the declaration + premise-list shape; the goal is taught by the
    few-shot prompt at inference. Records whose FOL cannot be parsed are
    skipped (returning None).
    """
    if not record.premises_fol:
        return None
    setup, premises, _goal, skipped = convert_premises_to_z3py(record.premises_fol)
    # If too much of this record's FOL failed to parse, drop it.
    if len(skipped) > max(1, len(record.premises_fol) // 4):
        return None
    if not premises:
        return None
    code = "\n".join(setup)
    code += "\npremises = [\n"
    code += ",\n".join(f"    {p}" for p in premises)
    code += "\n]\ngoal = True  # placeholder; see <goal> below"
    return code


def _to_chat_example(record: Record) -> dict | None:
    """One chat-formatted training example: 2-shot user → ideal assistant.

    Training labels supervise NL → Z3-Python program shape (declarations +
    premises list). Goal generation is left to the few-shot prompt because the
    release does not ship per-question FOL.
    """
    code = _build_z3py_program(record)
    if code is None:
        return None
    messages = build_messages(record.premises_nl, record.question_nl, n_fewshot=2)
    assistant = f"<z3py>\n{code}\n</z3py>"
    messages.append({"role": "assistant", "content": assistant})
    return {"messages": messages}


def build_chat_dataset(records: Iterable[Record]) -> list[dict]:
    out = []
    skipped_no_fol = 0
    skipped_parse = 0
    for r in records:
        if not r.premises_fol:
            skipped_no_fol += 1
            continue
        ex = _to_chat_example(r)
        if ex is None:
            skipped_parse += 1
            continue
        out.append(ex)
    log.info(
        "built %d training rows (skipped %d for missing FOL, %d for parse failure)",
        len(out), skipped_no_fol, skipped_parse,
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

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        **quant_kwargs,
    )

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    train_data = Dataset.from_list(build_chat_dataset(train_records))
    eval_data = (
        Dataset.from_list(build_chat_dataset(eval_records)) if eval_records else None
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
    lora_r: int = typer.Option(32),
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
        lora_r=lora_r,
    )
    train(train_records, eval_records, cfg)


if __name__ == "__main__":
    app()
