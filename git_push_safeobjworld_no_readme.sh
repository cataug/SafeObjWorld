#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Push SafeObjWorld to GitHub via SSH
# WITHOUT README
#
# Keeps:
#   code, scripts, logs, CSV tables, figures, configs, notebooks
#
# Excludes:
#   datasets, venvs, caches, checkpoints, model weights, archives,
#   heavy files > 95 MB
# ============================================================

PROJECT_ROOT="$HOME/SafeObjWorld"
REMOTE_SSH="git@github.com:cataug/SafeObjWorld.git"
BRANCH="main"
MAX_FILE_MB=95

cd "$PROJECT_ROOT"

echo "============================================================"
echo "SafeObjWorld Git push WITHOUT README"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "REMOTE=$REMOTE_SSH"
echo "BRANCH=$BRANCH"
echo "MAX_FILE_MB=$MAX_FILE_MB"
echo "============================================================"

# ------------------------------------------------------------
# 0. Init git
# ------------------------------------------------------------
if [ ! -d ".git" ]; then
  git init
fi

git checkout -B "$BRANCH"

git config user.name "cataug"
git config user.email "cataug@users.noreply.github.com"

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE_SSH"
else
  git remote add origin "$REMOTE_SSH"
fi

echo
echo "[git] remote:"
git remote -v

# ------------------------------------------------------------
# 1. SSH check
# ------------------------------------------------------------
echo
echo "[ssh] GitHub SSH check:"
ssh -T git@github.com || true

# ------------------------------------------------------------
# 2. Pull remote if exists
# ------------------------------------------------------------
echo
echo "[git] checking remote branch..."

if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  echo "[git] remote branch exists, pulling..."
  git pull --rebase --autostash origin "$BRANCH" || {
    echo "ERROR: pull/rebase failed. Resolve conflicts, then rerun."
    exit 1
  }
else
  echo "[git] remote branch does not exist or repo is empty."
fi

# ------------------------------------------------------------
# 3. .gitignore
# ------------------------------------------------------------
echo
echo "[gitignore] writing .gitignore"

cat > .gitignore <<'EOF'
# ============================================================
# SafeObjWorld ignore rules
# ============================================================

# ---- no README for first push ----
README.md
README.*

# ---- datasets / raw data ----
/data/
/datasets/
/dataset/
/DAVIS/
/DAVIS2017/
/ECCV_DATASETS*/
/*DAVIS*trainval*
*DAVIS-2017*

# ---- archives ----
*.zip
*.tar
*.tar.gz
*.tgz
*.rar
*.7z

# ---- Python environments ----
.venv/
.venv*/
venv/
env/
ENV/
__pycache__/
*.pyc
*.pyo
*.pyd
.ipynb_checkpoints/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# ---- ML weights/checkpoints/heavy model files ----
*.pt
*.pth
*.ckpt
*.safetensors
*.bin
*.onnx
*.engine
*.pkl
*.pickle
*.joblib
*.h5
*.hdf5

# ---- HuggingFace / model caches ----
hf_cache/
huggingface/
.cache/
cache/
models_cache/
model_cache/
wandb/
mlruns/

# ---- temporary files ----
*.tmp
*.temp
*.bak
*.swp
*.swo
.DS_Store
Thumbs.db

# ---- giant raw arrays ----
*.npy
*.npz

# ---- explicitly keep lightweight useful outputs ----
!*.csv
!*.tsv
!*.txt
!*.md
!*.json
!*.yaml
!*.yml
!*.py
!*.sh
!*.ipynb
!*.tex
!*.png
!*.jpg
!*.jpeg
!*.pdf
!*.svg
!*.log

# but still no README
README.md
README.*
EOF

# ------------------------------------------------------------
# 4. Dynamic exclude huge files
# ------------------------------------------------------------
echo
echo "[exclude] scanning files > ${MAX_FILE_MB} MB"

mkdir -p .git/info
: > .git/info/exclude
: > git_excluded_heavy_files.txt

find . -type f -size +"${MAX_FILE_MB}"M \
  -not -path "./.git/*" \
  -print0 | while IFS= read -r -d '' f; do
    rel="${f#./}"
    echo "/$rel" >> .git/info/exclude
    size_h=$(du -h "$f" | awk '{print $1}')
    echo "$size_h  $rel" >> git_excluded_heavy_files.txt
  done

if [ -s git_excluded_heavy_files.txt ]; then
  echo "[exclude] heavy files excluded:"
  cat git_excluded_heavy_files.txt
else
  echo "[exclude] no files > ${MAX_FILE_MB} MB"
fi

# ------------------------------------------------------------
# 5. Project manifest
# ------------------------------------------------------------
echo
echo "[manifest] creating git_project_manifest.txt"

{
  echo "SafeObjWorld Git manifest"
  echo "Generated: $(date)"
  echo
  echo "Project root: $PROJECT_ROOT"
  echo
  echo "Code/scripts:"
  find . -type f \( -name "*.py" -o -name "*.sh" -o -name "*.ipynb" -o -name "*.tex" \) \
    -not -path "./data/*" \
    -not -path "./.git/*" \
    -not -path "./.venv*/*" \
    | sort
  echo
  echo "CSV/result tables:"
  find . -type f \( -name "*.csv" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" \) \
    -not -path "./data/*" \
    -not -path "./.git/*" \
    -not -path "./.venv*/*" \
    | sort
  echo
  echo "Figures:"
  find . -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.pdf" -o -name "*.svg" \) \
    -not -path "./data/*" \
    -not -path "./.git/*" \
    -not -path "./.venv*/*" \
    | sort
  echo
  echo "Logs:"
  find . -type f -name "*.log" \
    -not -path "./data/*" \
    -not -path "./.git/*" \
    -not -path "./.venv*/*" \
    | sort
  echo
  echo "Excluded heavy files:"
  cat git_excluded_heavy_files.txt 2>/dev/null || true
} > git_project_manifest.txt

# ------------------------------------------------------------
# 6. Stage
# ------------------------------------------------------------
echo
echo "[git] staging"

git add -A

# ------------------------------------------------------------
# 7. Safety unstage
# ------------------------------------------------------------
echo
echo "[safety] unstaging README, datasets, checkpoints, archives, huge files"

# Always unstage README for this first push.
git reset -q -- README.md README.* 2>/dev/null || true

git diff --cached --name-only -z | while IFS= read -r -d '' f; do
  # forbidden directories
  case "$f" in
    data/*|datasets/*|dataset/*|DAVIS/*|DAVIS2017/*|.venv*/*|venv/*|env/*|hf_cache/*|cache/*|.cache/*)
      echo "[unstage forbidden dir] $f"
      git reset -q -- "$f" || true
      continue
      ;;
  esac

  # README
  case "$f" in
    README.md|README.*)
      echo "[unstage README] $f"
      git reset -q -- "$f" || true
      continue
      ;;
  esac

  # forbidden heavy extensions
  case "$f" in
    *.pt|*.pth|*.ckpt|*.safetensors|*.bin|*.onnx|*.engine|*.zip|*.tar|*.tar.gz|*.tgz|*.rar|*.7z|*.npy|*.npz|*.pkl|*.pickle|*.joblib|*.h5|*.hdf5)
      echo "[unstage forbidden ext] $f"
      git reset -q -- "$f" || true
      continue
      ;;
  esac

  # size check
  if [ -f "$f" ]; then
    size_mb=$(du -m "$f" | awk '{print $1}')
    if [ "$size_mb" -gt "$MAX_FILE_MB" ]; then
      echo "[unstage huge ${size_mb}MB] $f"
      git reset -q -- "$f" || true
      echo "${size_mb}MB  $f" >> git_excluded_heavy_files.txt
    fi
  fi
done

# Restage manifests and ignore rules, but NOT README.
git add .gitignore git_project_manifest.txt git_excluded_heavy_files.txt || true
git reset -q -- README.md README.* 2>/dev/null || true

# ------------------------------------------------------------
# 8. Show staged summary
# ------------------------------------------------------------
echo
echo "============================================================"
echo "[git] staged files:"
echo "============================================================"
git status --short

echo
echo "============================================================"
echo "[git] largest staged files:"
echo "============================================================"
git diff --cached --name-only | while read -r f; do
  if [ -f "$f" ]; then
    du -h "$f"
  fi
done | sort -hr | head -50 || true

echo
echo "============================================================"
echo "[git] check README is not staged:"
echo "============================================================"
git diff --cached --name-only | grep -i '^README' && {
  echo "ERROR: README is still staged."
  exit 1
} || echo "OK: README not staged."

echo
echo "============================================================"
echo "[git] check forbidden staged files:"
echo "============================================================"
bad=$(
  git diff --cached --name-only | grep -E '(^data/|^datasets/|^dataset/|^\.venv|\.pt$|\.pth$|\.ckpt$|\.safetensors$|\.bin$|\.zip$|\.tar$|\.npy$|\.npz$)' || true
)

if [ -n "$bad" ]; then
  echo "ERROR: forbidden files are still staged:"
  echo "$bad"
  exit 1
else
  echo "OK: no forbidden staged files."
fi

# ------------------------------------------------------------
# 9. Commit
# ------------------------------------------------------------
echo
echo "[git] commit"

if git diff --cached --quiet; then
  echo "[git] nothing to commit"
else
  git commit -m "Add SafeObjWorld experiments, scripts, figures, logs, and tables"
fi

# ------------------------------------------------------------
# 10. Push
# ------------------------------------------------------------
echo
echo "[git] push"

git push -u origin "$BRANCH"

echo
echo "============================================================"
echo "DONE"
echo "Pushed to:"
echo "  $REMOTE_SSH"
echo
echo "Tracked file count:"
git ls-files | wc -l
echo
echo "Largest tracked files:"
git ls-files | while read -r f; do
  if [ -f "$f" ]; then
    du -h "$f"
  fi
done | sort -hr | head -30 || true
echo "============================================================"
