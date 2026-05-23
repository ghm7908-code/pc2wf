#!/usr/bin/env bash
#SBATCH --job-name=pc2wf-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=GEOG-HPC-GPU
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Avoid `set -u` here because conda package activate/deactivate hooks
# may read backup vars that are unset.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}}"
DATA_ROOT="${DATA_ROOT:-/geogfs1/groups/hkurs/u3666068mgh/Tallin}"
PATCH_SIZE="${PATCH_SIZE:-32}"
MINI_BATCH="${MINI_BATCH:-512}"
NMS_TH="${NMS_TH:-0.01}"
LINE_POS_TH="${LINE_POS_TH:-0.01}"
LINE_NEG_TH="${LINE_NEG_TH:-0.05}"
LOSS_WEIGHT_PATCH="${LOSS_WEIGHT_PATCH:-1.0}"
LOSS_WEIGHT_VERTEX="${LOSS_WEIGHT_VERTEX:-50.0}"
LOSS_WEIGHT_LINE="${LOSS_WEIGHT_LINE:-1.0}"
EPOCHS="${EPOCHS:-20}"
BACKBONE_LR="${BACKBONE_LR:-5e-5}"
HEAD_LR="${HEAD_LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
PATCH_DROPOUT="${PATCH_DROPOUT:-0.2}"
VERTEX_DROPOUT="${VERTEX_DROPOUT:-0.1}"
LINE_DROPOUT="${LINE_DROPOUT:-0.3}"
LR_DECAY_FACTOR="${LR_DECAY_FACTOR:-0.5}"
LR_DECAY_PATIENCE="${LR_DECAY_PATIENCE:-2}"
MIN_LR="${MIN_LR:-1e-5}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-6}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-1e-4}"
SIGMA="${SIGMA:-0.01}"
CLIP="${CLIP:-0.01}"
CONDA_ENV="${CONDA_ENV:-/geogfs1/groups/hkurs/u3666068mgh/conda_envs/pc2wf-h100-final}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  if [ -n "$CONDA_ENV" ]; then
    conda activate "$CONDA_ENV"
  fi
fi

PATCH_ROOT="${DATA_ROOT}/patches_${PATCH_SIZE}_noise_sigma${SIGMA}clip${CLIP}"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "PATCH_ROOT=$PATCH_ROOT"
echo "CONDA_ENV=$CONDA_ENV"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "PATCH_SIZE=$PATCH_SIZE MINI_BATCH=$MINI_BATCH SIGMA=$SIGMA CLIP=$CLIP"
echo "EPOCHS=$EPOCHS BACKBONE_LR=$BACKBONE_LR HEAD_LR=$HEAD_LR WEIGHT_DECAY=$WEIGHT_DECAY"
echo "PATCH_DROPOUT=$PATCH_DROPOUT VERTEX_DROPOUT=$VERTEX_DROPOUT LINE_DROPOUT=$LINE_DROPOUT"
echo "LR_DECAY_FACTOR=$LR_DECAY_FACTOR LR_DECAY_PATIENCE=$LR_DECAY_PATIENCE MIN_LR=$MIN_LR"
echo "EARLY_STOP_PATIENCE=$EARLY_STOP_PATIENCE EARLY_STOP_MIN_DELTA=$EARLY_STOP_MIN_DELTA"

export OMP_NUM_THREADS
export PYTHONNOUSERSITE
export CUDA_HOME="${CONDA_PREFIX:-}"
if [ -n "${CUDA_HOME}" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
fi

if [ ! -d "$PATCH_ROOT/train" ]; then
  echo "Training patch directory not found: $PATCH_ROOT/train" >&2
  exit 1
fi

python - <<'PY'
import os
import torch
import MinkowskiEngine as ME

print("Preflight torch:", torch.__version__)
print("Preflight torch.cuda:", torch.version.cuda)
print("Preflight cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Preflight device:", torch.cuda.get_device_name(0))
    print("Preflight capability:", torch.cuda.get_device_capability(0))
else:
    raise SystemExit("CUDA is not available in the training environment.")

print("Preflight MinkowskiEngine:", getattr(ME, "__version__", "unknown"))
print("Preflight MinkowskiEngine file:", ME.__file__)
print("Preflight CONDA_PREFIX:", os.environ.get("CONDA_PREFIX", ""))
PY

python "$PROJECT_ROOT/main.py" \
  -d "$DATA_ROOT" \
  -p "$PATCH_SIZE" \
  -b "$MINI_BATCH" \
  -nt "$NMS_TH" \
  -lpt "$LINE_POS_TH" \
  -lnt "$LINE_NEG_TH" \
  -lwP "$LOSS_WEIGHT_PATCH" \
  -lwV "$LOSS_WEIGHT_VERTEX" \
  -lwL "$LOSS_WEIGHT_LINE" \
  -s "$SIGMA" \
  -c "$CLIP" \
  -e "$EPOCHS" \
  --backbone_lr "$BACKBONE_LR" \
  --head_lr "$HEAD_LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --patch_dropout "$PATCH_DROPOUT" \
  --vertex_dropout "$VERTEX_DROPOUT" \
  --line_dropout "$LINE_DROPOUT" \
  --lr_decay_factor "$LR_DECAY_FACTOR" \
  --lr_decay_patience "$LR_DECAY_PATIENCE" \
  --min_lr "$MIN_LR" \
  --early_stop_patience "$EARLY_STOP_PATIENCE" \
  --early_stop_min_delta "$EARLY_STOP_MIN_DELTA"
