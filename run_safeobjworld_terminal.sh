#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SafeObjWorld terminal launcher with fresh CUDA-compatible env
# ============================================================

PROJECT_ROOT="$HOME/SafeObjWorld"
DATA_ROOT="$PROJECT_ROOT/data/DAVIS2017"
RUN_ROOT="$PROJECT_ROOT/runs_musthave"
CODE_DIR="$PROJECT_ROOT/code"
VENV_DIR="$PROJECT_ROOT/.venv_safeobjworld"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "SafeObjWorld terminal launcher"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "RUN_ROOT=$RUN_ROOT"
echo "VENV_DIR=$VENV_DIR"
echo "============================================================"

# ------------------------------------------------------------
# 0. Optional: kill old broken CPU/GPU jobs from previous run
# ------------------------------------------------------------
if [ "${KILL_OLD:-0}" = "1" ]; then
  echo "[KILL_OLD=1] Killing previous SafeObjWorld jobs..."
  pkill -f "safeobjworld_launcher.py" || true
  pkill -f "safeobjworld_run.py" || true
fi

# ------------------------------------------------------------
# 1. Basic dataset check
# ------------------------------------------------------------
if [ ! -d "$DATA_ROOT/DAVIS/JPEGImages/480p" ]; then
  echo "ERROR: missing $DATA_ROOT/DAVIS/JPEGImages/480p"
  exit 1
fi

if [ ! -d "$DATA_ROOT/DAVIS/Annotations/480p" ]; then
  echo "ERROR: missing $DATA_ROOT/DAVIS/Annotations/480p"
  exit 1
fi

if [ ! -f "$DATA_ROOT/DAVIS/ImageSets/2017/train.txt" ]; then
  echo "ERROR: missing $DATA_ROOT/DAVIS/ImageSets/2017/train.txt"
  exit 1
fi

if [ ! -f "$DATA_ROOT/DAVIS/ImageSets/2017/val.txt" ]; then
  echo "ERROR: missing $DATA_ROOT/DAVIS/ImageSets/2017/val.txt"
  exit 1
fi

echo "[OK] DAVIS2017 structure found."

# ------------------------------------------------------------
# 2. Create fresh venv
# ------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "[create env] $VENV_DIR"

  if command -v python3.10 >/dev/null 2>&1; then
    python3.10 -m venv "$VENV_DIR"
  elif command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv "$VENV_DIR"
  else
    python3 -m venv "$VENV_DIR"
  fi
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------
# 3. Install CUDA-compatible PyTorch
# ------------------------------------------------------------
# Driver reports CUDA 12.4 in your previous logs, so cu121 wheel is safe.
# This avoids the current broken torch build that expects a newer driver.
echo "[install] torch cu121"
pip install --upgrade --force-reinstall \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

echo "[install] project deps"
pip install --upgrade \
  numpy pandas pillow opencv-python tqdm scikit-learn matplotlib

# ------------------------------------------------------------
# 4. Patch runtime safety for CPU fallback and old oneDNN issue
# ------------------------------------------------------------
# If CUDA ever fails and torch goes CPU, this prevents the oneDNN primitive crash.
python - <<'PY'
from pathlib import Path

p = Path.home() / "SafeObjWorld" / "code" / "safeobjworld_run.py"
txt = p.read_text()

needle = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
patch = """import torch
try:
    # Prevent CPU oneDNN/MKLDNN primitive crashes if CUDA is unavailable.
    torch.backends.mkldnn.enabled = False
except Exception:
    pass
import torch.nn as nn
import torch.nn.functional as F
"""

if needle in txt and "torch.backends.mkldnn.enabled = False" not in txt:
    txt = txt.replace(needle, patch)
    p.write_text(txt)
    print("[patch] inserted torch.backends.mkldnn.enabled = False")
else:
    print("[patch] no change needed")
PY

# ------------------------------------------------------------
# 5. CUDA sanity check
# ------------------------------------------------------------
echo "============================================================"
echo "CUDA sanity check"
echo "============================================================"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
        x = torch.randn(4, 3, 192, 192, device=f"cuda:{i}")
        conv = torch.nn.Conv2d(3, 16, 3, padding=1).to(f"cuda:{i}")
        y = conv(x)
        torch.cuda.synchronize(i)
        print("conv ok:", tuple(y.shape))
else:
    raise SystemExit("CUDA is still not available. Check nvidia-smi and installed torch.")
PY

echo "============================================================"
echo "nvidia-smi"
echo "============================================================"
nvidia-smi || true

# ------------------------------------------------------------
# 6. Clean broken partial runs if requested
# ------------------------------------------------------------
if [ "${CLEAN_BROKEN:-0}" = "1" ]; then
  echo "[CLEAN_BROKEN=1] Removing partial model/eval outputs..."
  rm -rf "$RUN_ROOT/models"
  rm -rf "$RUN_ROOT/eval"
  rm -rf "$RUN_ROOT/RESULTS"
  rm -rf "$RUN_ROOT/logs"
fi

mkdir -p "$RUN_ROOT/logs"

# ------------------------------------------------------------
# 7. Launch dynamic scheduler
# ------------------------------------------------------------
# Must-have plan:
#   baselines: copy_last, linear_centroid, flow_warp
#   learned: convlstm, safeobj_full, safeobj_det, safeobj_no_geom, safeobj_no_stress
#   stresses: clean, occlusion, blur, frame_drop
#   H: 5 by default
#
# For first debug run, set QUICK=1.
# For full run, leave QUICK=0.
# ------------------------------------------------------------

if [ "${QUICK:-0}" = "1" ]; then
  IMAGE_SIZE=160
  BATCH_SIZE=8
  EPOCHS=2
  MAX_TRAIN_SAMPLES=512
  MAX_VAL_SAMPLES=256
  MAX_JOBS_PER_GPU=1
  MIN_FREE_MB=7000
  echo "[QUICK=1] Running smoke-test settings."
else
  IMAGE_SIZE=192
  BATCH_SIZE=8
  EPOCHS=8
  MAX_TRAIN_SAMPLES=0
  MAX_VAL_SAMPLES=0
  MAX_JOBS_PER_GPU=2
  MIN_FREE_MB=9000
fi

echo "============================================================"
echo "Launching SafeObjWorld jobs"
echo "IMAGE_SIZE=$IMAGE_SIZE"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "EPOCHS=$EPOCHS"
echo "MAX_TRAIN_SAMPLES=$MAX_TRAIN_SAMPLES"
echo "MAX_VAL_SAMPLES=$MAX_VAL_SAMPLES"
echo "MAX_JOBS_PER_GPU=$MAX_JOBS_PER_GPU"
echo "MIN_FREE_MB=$MIN_FREE_MB"
echo "============================================================"

python "$CODE_DIR/safeobjworld_launcher.py" \
  --project_root "$PROJECT_ROOT" \
  --dataset_root "$DATA_ROOT" \
  --run_root "$RUN_ROOT" \
  --image_size "$IMAGE_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --prefix_len 3 \
  --max_h 5 \
  --run_h10 0 \
  --max_train_samples "$MAX_TRAIN_SAMPLES" \
  --max_val_samples "$MAX_VAL_SAMPLES" \
  --num_workers 4 \
  --max_jobs_per_gpu "$MAX_JOBS_PER_GPU" \
  --min_free_mb "$MIN_FREE_MB" \
  --poll_seconds 20 \
  --start_gap_seconds 12 \
  --amp

echo "============================================================"
echo "DONE"
echo "Main table:"
echo "$RUN_ROOT/RESULTS/table_main_reliability.csv"
echo
echo "All summaries:"
echo "$RUN_ROOT/RESULTS/all_summaries.csv"
echo
echo "Logs:"
echo "$RUN_ROOT/logs"
echo "============================================================"
