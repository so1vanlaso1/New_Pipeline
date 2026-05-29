#!/usr/bin/env bash
# One-shot environment setup on the training machine.
#
# Assumes Ubuntu 22.04+/Debian on a Linux box with NVIDIA driver + CUDA already
# installed (verify with `nvidia-smi` before running). RTX 5070 is Blackwell
# (sm_120) and needs CUDA 12.6+ wheels — that's what we install below.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh

set -euo pipefail

# 1. Sanity-check the GPU.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found — install the NVIDIA driver before running this." >&2
    exit 1
fi
echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# 2. System Python: prefer 3.11. Fall back to whichever python3 is on PATH.
PYTHON_BIN=$(command -v python3.11 || command -v python3)
echo "Using $PYTHON_BIN ($($PYTHON_BIN --version))"

# 3. Create venv if absent.
if [ ! -d .venv ]; then
    "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 4. Upgrade pip + install torch first (with the right CUDA wheels for Blackwell).
pip install --upgrade pip wheel
# CUDA 12.6 wheels — required for sm_120. If you're on CUDA 12.1 (older
# driver), drop the --index-url and pip will pull the default cu121 wheels.
pip install --index-url https://download.pytorch.org/whl/cu126 "torch>=2.5"

# 5. Install everything else.
pip install -r requirements.txt

# 6. Flash Attention 2 — installed separately because the wheel build needs to
# see the torch install we just did. On 12 GB this is essentially required.
# If this fails on Blackwell (sm_120) due to a kernel-build issue, the trainer
# falls back gracefully when launched with `--attn sdpa`.
pip install flash-attn --no-build-isolation || echo "[warn] flash-attn install failed — train with --attn sdpa instead"

# 7. Quick import check.
python -c "
import torch
print(f'torch    {torch.__version__}')
print(f'CUDA     {torch.version.cuda}')
print(f'GPU      {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"(none)\"}')
print(f'Capabil. sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]} ' if torch.cuda.is_available() else '')
import transformers, peft, trl, datasets, bitsandbytes
print(f'transformers {transformers.__version__}  peft {peft.__version__}  trl {trl.__version__}')
print(f'datasets {datasets.__version__}  bitsandbytes {bitsandbytes.__version__}')
try:
    import flash_attn
    print(f'flash-attn {flash_attn.__version__}  OK')
except ImportError:
    print('flash-attn  NOT INSTALLED — pass --attn sdpa to train.sh')
"

echo
echo "Environment ready. Run ./train.sh to start fine-tuning."
