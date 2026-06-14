#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$HOME/SafeObjWorld"
VENV_DIR="$PROJECT_ROOT/.venv_safeobjworld"
DATA_ROOT="$PROJECT_ROOT/data/DAVIS2017"

source "$VENV_DIR/bin/activate"

if [ "${KILL_OLD:-0}" = "1" ]; then
  pkill -f "safeobjworld_v2.py" || true
  pkill -f "launch_safeobjworld_v2.py" || true
fi

if [ "${QUICK:-0}" = "1" ]; then
  RUN_ROOT="$PROJECT_ROOT/runs_v2_smoke"
  IMAGE_SIZE=160
  BATCH_SIZE=8
  EPOCHS=2
  MAX_TRAIN_SAMPLES=512
  MAX_VAL_SAMPLES=256
  MAX_JOBS_PER_GPU=1
  MIN_FREE_MB=7000
  RUN_H10=0
  OPTIONAL_STRESS=0
else
  RUN_ROOT="$PROJECT_ROOT/runs_v2_full"
  IMAGE_SIZE="${IMAGE_SIZE:-192}"
  BATCH_SIZE="${BATCH_SIZE:-8}"
  EPOCHS="${EPOCHS:-10}"
  MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
  MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"
  MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-2}"
  MIN_FREE_MB="${MIN_FREE_MB:-9000}"
  RUN_H10="${RUN_H10:-0}"
  OPTIONAL_STRESS="${OPTIONAL_STRESS:-0}"
fi

if [ "${CLEAN:-0}" = "1" ]; then
  echo "[CLEAN=1] removing $RUN_ROOT"
  rm -rf "$RUN_ROOT"
fi

mkdir -p "$RUN_ROOT/logs"

echo "============================================================"
echo "SafeObjWorld V2 launch"
echo "RUN_ROOT=$RUN_ROOT"
echo "IMAGE_SIZE=$IMAGE_SIZE"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "EPOCHS=$EPOCHS"
echo "MAX_TRAIN_SAMPLES=$MAX_TRAIN_SAMPLES"
echo "MAX_VAL_SAMPLES=$MAX_VAL_SAMPLES"
echo "MAX_JOBS_PER_GPU=$MAX_JOBS_PER_GPU"
echo "MIN_FREE_MB=$MIN_FREE_MB"
echo "RUN_H10=$RUN_H10"
echo "OPTIONAL_STRESS=$OPTIONAL_STRESS"
echo "============================================================"

python "$PROJECT_ROOT/code/launch_safeobjworld_v2.py" \
  --project_root "$PROJECT_ROOT" \
  --dataset_root "$DATA_ROOT" \
  --run_root "$RUN_ROOT" \
  --image_size "$IMAGE_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --prefix_len 3 \
  --max_h 5 \
  --run_h10 "$RUN_H10" \
  --run_optional_stress "$OPTIONAL_STRESS" \
  --max_train_samples "$MAX_TRAIN_SAMPLES" \
  --max_val_samples "$MAX_VAL_SAMPLES" \
  --num_workers 4 \
  --max_jobs_per_gpu "$MAX_JOBS_PER_GPU" \
  --min_free_mb "$MIN_FREE_MB" \
  --poll_seconds 20 \
  --start_gap_seconds 10 \
  --amp

echo "============================================================"
echo "FINISHED"
echo "Main results:"
echo "$RUN_ROOT/RESULTS"
echo "============================================================"
