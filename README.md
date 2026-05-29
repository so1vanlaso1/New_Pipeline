# EXACT 2026 Track 1 — Logic QA Pipeline

Neuro-symbolic pipeline for the **EXACT 2026 (IJCNN) — Track 1: Logic-Based Educational Queries** competition. Translates NL premises to a **Z3 Python DSL** with a LoRA-tuned [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), runs Z3 for entailment, votes across K=5 self-consistency samples, and falls back to chain-of-thought when symbolic fails.

Production target is an **RTX 5070 (12 GB GDDR7, Blackwell)** running Linux. Defaults are sized for that envelope.

Why Z3 Python DSL (not SMT-LIB): the dataset's training FOL already uses Python-style `ForAll(x, …)` / `Exists(x, …)` syntax. Matching the output format minimizes the translation distance for the LoRA. The runner safely exec's translator output in an AST-validated sandbox with allow-listed Z3 names — no `__builtins__`, no attribute access, no imports.

Validated against the real `Logic_Based_Educational_Queries.json` release:
- **99.4%** of FOL formulas parse with the converter (4443/4470)
- **94.8%** of generated Z3 Python programs exec cleanly in the sandbox
- **97.6%** of records (401/411) have all premises convertible
- 411 source records expand into 808 normalized Records: 416 Yes/No/Uncertain + 360 MCQ + 32 open-ended

Implementation plan: see `~/.claude/plans/https-ura-hcmut-edu-vn-exact-this-is-the-atomic-platypus.md`.

## Pipeline

```
NL premises + Q  ──►  Translator (Qwen3.5-4B + LoRA) ──►  K=5 Z3 Python samples
                                                          │
                                                          ▼
                                              safe-exec → Z3 entailment
                                              (MCQ returns "Unknown" if no
                                               option follows)
                                                          │
                                ┌─────────────────────────┴────────────────────────┐
                                ▼                                                  ▼
                       4–5 of 5 agree (0.95)                            split / open-ended
                       3 of 5 agree (0.70)                                         │
                                │                                                  ▼
                                ▼                                          CoT fallback (K=5)
                       symbolic answer + unsat-core                                │
                                │                                                  │
                                └────────────────────────┬─────────────────────────┘
                                                         ▼
                                          {answer, explanation, fol, cot, premises, confidence}
```

## Project layout

```
src/
  data/load.py                EXACT JSON → Record (schema-flexible field aliases,
                              expands N parallel questions per source record)
  data/types.py               Record / Translation / SolverVerdict / FinalAnswer
  translator/fol_converter.py Unicode/Pythonic FOL → Z3 Python DSL (for training labels)
  translator/prompt.py        Few-shot NL→Z3 Python prompt
  translator/parse.py         Robust <z3py> block extraction
  translator/infer.py         vLLM client + LoRA-aware translator
  translator/train_lora.py    TRL SFTTrainer LoRA on NL→Z3-Python
  solver/z3_runner.py         AST-validated safe exec → Z3 → Yes/No/Uncertain/MCQ/Unknown
  vote.py                     K-of-N majority vote with conservative confidence
  fallback/cot.py             vLLM CoT fallback with self-consistency (supports Unknown)
  explain.py                  Unsat-core / CoT → FinalAnswer
  pipeline.py                 End-to-end orchestration with per-stage timing
  eval/score.py               P1 accuracy + per-answer-type breakdown
  cli/{run,eval,inspect}.py
tests/                        Pytest smoke + end-to-end exec tests (no GPU needed)
configs/default.yaml          All knobs live here
data/                         (gitignored) drop EXACT JSON here
artifacts/                    (gitignored) model adapters + predictions + reports
```

## Quick setup (WSL2 + Linux + RTX 5070)

vLLM and Unsloth are Linux-only, so you want WSL2 for development.

### 1. Install WSL2 + Ubuntu (one time)

```powershell
# In an elevated PowerShell on Windows:
wsl --install -d Ubuntu-22.04
# Reboot, then finish Ubuntu setup (username/password).
```

### 2. Set up the Linux env

```bash
# Inside the WSL2 Ubuntu shell:
sudo apt update && sudo apt install -y python3.11 python3.11-venv git build-essential

# Verify CUDA is visible (Windows nvidia driver passes through to WSL):
nvidia-smi

cd "/mnt/c/Users/hi/New folder (2)"
python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel
pip install -e ".[gpu,dev]"
# Optional Unsloth speedup:
# pip install "unsloth[cu121-torch24] @ git+https://github.com/unslothai/unsloth.git"
```

### 3. Drop the training data

Put the EXACT JSON files under `data/exact2026/`:

```
data/exact2026/train.json
data/exact2026/dev.json     (or use --dev-frac to split off train)
data/exact2026/test.json    (when released)
```

Confirm the loader auto-detects the right fields:

```bash
python -m cli.inspect data/exact2026/train.json
```

If `answer_type` shows `open_ended` for everything (or any other field looks wrong), the release uses different field names than the aliases in `src/data/load.py:FIELD_ALIASES` — add the missing alias there, or pass a `field_map` override.

## Run the pipeline

### Baseline (no LoRA, few-shot translator)

```bash
python -m cli.run --data data/exact2026/dev.json --out artifacts/dev_baseline.json
python -m cli.eval --data data/exact2026/dev.json --pred artifacts/dev_baseline.json
```

### Fine-tune the translator (Day 2)

```bash
python -m translator.train_lora --train data/exact2026/train.json --dev data/exact2026/dev.json \
  --out artifacts/translator-lora --epochs 3 --batch-size 2 --grad-accum 8 --lora-r 32
```

Defaults above (`--batch-size 2 --grad-accum 8`, effective batch 16) target the **5070's 12 GB**. On a 4090 (24 GB) you can raise to `--batch-size 4 --grad-accum 4` for the same effective batch and faster wall time. Wall time on a 5070 for ~720 rows of NL→Z3-Python on Qwen3.5-4B + 4-bit base + LoRA r=32 is on the order of 1–2 hours; on a 4090, ~half that.

### Run with the fine-tuned adapter

```bash
python -m cli.run --data data/exact2026/dev.json --out artifacts/dev_lora.json \
  --lora artifacts/translator-lora
python -m cli.eval --data data/exact2026/dev.json --pred artifacts/dev_lora.json
```

## Configuration

All knobs live in `configs/default.yaml`. The ones worth tuning first:

| Knob | Default | Notes |
| --- | --- | --- |
| `translator.k_samples` | 5 | Drop to 3 if you're tight on the 60s budget. Don't drop to 1. |
| `translator.temperature` | 0.3 | Higher gives more diversity for the vote, but worse SMT-LIB. |
| `vote.high_confidence_threshold` | 4 | Vote count for 0.95 confidence (out of K). |
| `vote.medium_confidence_threshold` | 3 | Vote count for 0.70 confidence; below this triggers CoT fallback. |
| `solver.timeout_ms` | 5000 | Per Z3 call; tighten if Z3 hangs are eating budget. |
| `fallback.use_reasoner_variant` | false | A dedicated reasoner variant does not fit alongside the workhorse on a 5070's 12 GB. On a 4090 you can load a second model and flip this on. |
| `pipeline.wall_clock_budget_s` | 55 | Per-record wall clock; pipeline will skip CoT if budget is too low. |

## Testing

```bash
pytest tests/ -v
```

The smoke tests cover the Z3 wrapper, the LLM-output parser, and the voter — all of which run on CPU and require no model download. Run them after any change to those modules.

## Troubleshooting

- **`No module named vllm`** — you're running outside the GPU venv. Activate `.venv` inside WSL.
- **`CUDA out of memory` during inference** — on a 5070 (12 GB) drop `vllm.gpu_memory_utilization` to 0.75, shrink `max_model_len` to 3072, or switch `models.quantization` from `fp8` to `awq` (~½ the VRAM).
- **vLLM doesn't see the 5070 (`sm_120`)** — Blackwell needs a recent vLLM build. `pip install --upgrade "vllm>=0.7"` and a matching PyTorch nightly with CUDA 12.6 wheels. If the AWQ Marlin kernel is unavailable for Blackwell, fall back to `models.quantization: fp8`.
- **Z3 returns `unknown` on most queries** — translator is producing huge formulas. Inspect with `--limit 3` and check the SMT-LIB; the few-shot prompt may need more examples covering the offending pattern.
- **High `parse_error` rate** — LLM is ignoring the `<smtlib>/<goal>` tag format. Either raise the few-shot example count (`n_fewshot=3`) or LoRA-fine-tune (the trainer enforces the format in the assistant target).
- **Loader returns 0 records / wrong fields** — re-run `python -m cli.inspect <path>` and update `FIELD_ALIASES` in `src/data/load.py`.

## Day-by-day build order (matches the plan)

| Day | Goal |
| --- | --- |
| 0 (½) | Wire data, pipeline, scorer. Run pure-CoT baseline as floor. |
| 1 | Few-shot translator + Z3 + vote. Compare to Day-0 floor — the gap shows whether translation is the bottleneck. |
| 2 | LoRA fine-tune the translator. Swap into pipeline and re-evaluate. |
| 3 | Tune K, thresholds, fallback. Inspect failures by answer type. |
