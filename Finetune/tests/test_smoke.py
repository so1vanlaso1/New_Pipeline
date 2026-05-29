"""Smoke tests that run on CPU — verify the env is wired up before kicking off
the GPU training run.

    pytest tests/

Costs no GPU time. If these fail, training will also fail, so run them first.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.fol_converter import convert_premises_to_z3py, parse
from finetune.load import load_records
from finetune.prompt import build_messages
from finetune.train_lora import build_chat_dataset


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "annotation_ready_merged.json"


def test_data_loader_reads_file():
    records = load_records(DATA_PATH)
    assert len(records) > 0
    print(f"loaded {len(records)} records")


def test_fol_converter_handles_real_record():
    # Record 0 from the EXACT release.
    fol = [
        "∀x (WT(x) → O(x))",
        "∀x (¬PEP8(x) → ¬WT(x))",
        "∀x (WT(x) → PEP8(x))",
    ]
    setup, prems, _goal, skipped = convert_premises_to_z3py(fol)
    assert not skipped
    assert len(prems) == 3
    assert any("DeclareSort" in line for line in setup)
    assert any("Function" in line for line in setup)


def test_training_dataset_builds():
    records = load_records(DATA_PATH)
    rows = build_chat_dataset(records[:50])
    # Should build SOMETHING from 50 records; we don't require all to succeed
    # because a few have unparseable FOL.
    assert len(rows) > 0
    assert "messages" in rows[0]
    assert rows[0]["messages"][-1]["role"] == "assistant"
    assert "<z3py>" in rows[0]["messages"][-1]["content"]


def test_prompt_template_renders():
    msgs = build_messages(["A is true.", "B is true."], "Is C true?")
    # 1 system + 2 fewshot pairs (4 msgs) + 1 final user = 6 messages
    assert len(msgs) == 6
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
