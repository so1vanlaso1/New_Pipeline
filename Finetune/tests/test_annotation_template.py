"""Verify the loader + trainer wire up correctly against the new annotation
template (the minimal `questions-NL` + `questions-FOL` schema)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.load import load_records
from finetune.train_lora import _build_z3py_program, build_chat_dataset

TEMPLATE = Path(__file__).resolve().parent.parent / "data" / "annotation_template.json"


def test_loader_picks_up_questions_fol():
    records = load_records(TEMPLATE)
    # 3 records × respective question counts = 1 + 4 + 4 = 9 expanded records
    assert len(records) == 9
    # Every record should have BOTH premises_fol and question_fol populated.
    for r in records:
        assert r.premises_fol, f"missing premises_fol for {r.id}"
        assert r.question_fol, f"missing question_fol for {r.id}"


def test_builder_uses_annotated_goal():
    records = load_records(TEMPLATE)
    result = _build_z3py_program(records[0])
    assert result is not None
    code, has_goal = result
    assert has_goal, "expected the template's first record to have annotated goal FOL"
    # The placeholder string should NOT appear; a real goal expression should.
    assert "goal = True" not in code
    assert "goal = " in code
    # First record's question: "Does it follow that all Python projects are optimized?"
    # FOL: ∀x (O(x)). After conversion: ForAll([x], O(x))
    assert "ForAll([x], O(x))" in code


def test_require_goal_fol_keeps_all_template_rows():
    records = load_records(TEMPLATE)
    rows = build_chat_dataset(records, require_goal_fol=True)
    assert len(rows) == 9, f"expected 9 fully-annotated rows, got {len(rows)}"


def test_chat_target_contains_real_goal_line():
    records = load_records(TEMPLATE)
    rows = build_chat_dataset(records)
    assistant = rows[0]["messages"][-1]["content"]
    assert "<z3py>" in assistant
    assert "</z3py>" in assistant
    assert "goal = True" not in assistant
    assert "goal = " in assistant
