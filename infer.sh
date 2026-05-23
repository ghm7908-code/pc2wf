#!/usr/bin/env bash
#SBATCH --job-name=pc2wf-infer
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err
#SBATCH --partition=GEOG-HPC-GPU
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Inference + OBJ export pipeline for PC2WF on HPC.
# It is designed to avoid GLIBCXX mismatch by preferring conda runtime libs.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}}"

DATA_ROOT="${DATA_ROOT:-/geogfs1/groups/hkurs/u3666068mgh/Tallin}"
PATCH_SIZE="${PATCH_SIZE:-32}"
SIGMA="${SIGMA:-0.01}"
CLIP="${CLIP:-0.01}"
SPLIT="${SPLIT:-test}"
CONDA_ENV="${CONDA_ENV:-/geogfs1/groups/hkurs/u3666068mgh/conda_envs/pc2wf-h100-final}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/checkpoint_sigma${SIGMA}clip${CLIP}}"
OVERWRITE="${OVERWRITE:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
EVAL="${EVAL:-0}"
EVAL_VERTEX_TH="${EVAL_VERTEX_TH:-0.03}"
EVAL_EDGE_TH="${EVAL_EDGE_TH:-0.05}"
EVAL_REPORT="${EVAL_REPORT:-$PROJECT_ROOT/visualize/eval_patch${PATCH_SIZE}sigma${SIGMA}clip${CLIP}_${SPLIT}.json}"
THRESHOLD_PROFILE="${THRESHOLD_PROFILE:-default}"

RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/visualize/run_test_result/patch${PATCH_SIZE}sigma${SIGMA}clip${CLIP}_${SPLIT}}"
OBJ_DIR="${OBJ_DIR:-$PROJECT_ROOT/visualize/visualize_line/patch${PATCH_SIZE}sigma${SIGMA}clip${CLIP}_${SPLIT}}"

# Prediction thresholds (run_test_line.py)
PATCH_PROB_TH="${PATCH_PROB_TH:-}"
LINE_PROB_TH="${LINE_PROB_TH:-}"
VERTEX_NMS_TH="${VERTEX_NMS_TH:-}"
LINE_DIST_TH="${LINE_DIST_TH:-}"

# Post-process thresholds (visualize_line.py)
VERTEX_PROB_TH="${VERTEX_PROB_TH:-}"
VIS_VERTEX_NMS_TH="${VIS_VERTEX_NMS_TH:-}"
VIS_LINE_PROB_TH="${VIS_LINE_PROB_TH:-}"
VIS_LINE_LEN_TH="${VIS_LINE_LEN_TH:-}"
VIS_LINE_NMS_TH="${VIS_LINE_NMS_TH:-}"
MERGE_TH="${MERGE_TH:-}"

case "$THRESHOLD_PROFILE" in
  strict)
    PATCH_PROB_TH="${PATCH_PROB_TH:-0.65}"
    LINE_PROB_TH="${LINE_PROB_TH:-0.55}"
    VERTEX_NMS_TH="${VERTEX_NMS_TH:-0.01}"
    LINE_DIST_TH="${LINE_DIST_TH:-0.03}"
    VERTEX_PROB_TH="${VERTEX_PROB_TH:-0.65}"
    VIS_VERTEX_NMS_TH="${VIS_VERTEX_NMS_TH:-0.01}"
    VIS_LINE_PROB_TH="${VIS_LINE_PROB_TH:-0.60}"
    VIS_LINE_LEN_TH="${VIS_LINE_LEN_TH:-0.03}"
    VIS_LINE_NMS_TH="${VIS_LINE_NMS_TH:-0.05}"
    MERGE_TH="${MERGE_TH:-0.02}"
    ;;
  default)
    PATCH_PROB_TH="${PATCH_PROB_TH:-0.5}"
    LINE_PROB_TH="${LINE_PROB_TH:-0.5}"
    VERTEX_NMS_TH="${VERTEX_NMS_TH:-0.01}"
    LINE_DIST_TH="${LINE_DIST_TH:-0.03}"
    VERTEX_PROB_TH="${VERTEX_PROB_TH:-0.5}"
    VIS_VERTEX_NMS_TH="${VIS_VERTEX_NMS_TH:-0.01}"
    VIS_LINE_PROB_TH="${VIS_LINE_PROB_TH:-0.5}"
    VIS_LINE_LEN_TH="${VIS_LINE_LEN_TH:-0.02}"
    VIS_LINE_NMS_TH="${VIS_LINE_NMS_TH:-0.05}"
    MERGE_TH="${MERGE_TH:-0.02}"
    ;;
  *)
    echo "Unknown THRESHOLD_PROFILE: $THRESHOLD_PROFILE" >&2
    echo "Supported values: default, strict" >&2
    exit 1
    ;;
esac

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  if [ -n "$CONDA_ENV" ]; then
    conda activate "$CONDA_ENV"
  fi
fi

export CUDA_HOME="${CONDA_PREFIX:-}"
if [ -n "$CUDA_HOME" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
  # Prefer conda C++ runtime to avoid GLIBCXX mismatch with MinkowskiEngine backend.
  if [ -f "$CUDA_HOME/lib/libstdc++.so.6" ]; then
    export LD_PRELOAD="$CUDA_HOME/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
fi

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "CONDA_ENV=$CONDA_ENV"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "PATCH_SIZE=$PATCH_SIZE SIGMA=$SIGMA CLIP=$CLIP SPLIT=$SPLIT"
echo "RESULT_DIR=$RESULT_DIR"
echo "OBJ_DIR=$OBJ_DIR"
echo "THRESHOLD_PROFILE=$THRESHOLD_PROFILE"
echo "PATCH_PROB_TH=$PATCH_PROB_TH LINE_PROB_TH=$LINE_PROB_TH VERTEX_NMS_TH=$VERTEX_NMS_TH LINE_DIST_TH=$LINE_DIST_TH"
echo "VERTEX_PROB_TH=$VERTEX_PROB_TH VIS_VERTEX_NMS_TH=$VIS_VERTEX_NMS_TH VIS_LINE_PROB_TH=$VIS_LINE_PROB_TH VIS_LINE_LEN_TH=$VIS_LINE_LEN_TH VIS_LINE_NMS_TH=$VIS_LINE_NMS_TH MERGE_TH=$MERGE_TH"
echo "EVAL=$EVAL EVAL_VERTEX_TH=$EVAL_VERTEX_TH EVAL_EDGE_TH=$EVAL_EDGE_TH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "LD_PRELOAD=${LD_PRELOAD:-}"

export OMP_NUM_THREADS
export PYTHONNOUSERSITE

PATCH_ROOT="${DATA_ROOT}/patches_${PATCH_SIZE}_noise_sigma${SIGMA}clip${CLIP}"

if [ ! -d "$PATCH_ROOT/$SPLIT" ]; then
  echo "Inference split directory not found: $PATCH_ROOT/$SPLIT" >&2
  exit 1
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "Checkpoint directory not found: $CHECKPOINT_DIR" >&2
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
print("Preflight MinkowskiEngine:", getattr(ME, "__version__", "unknown"))
print("Preflight MinkowskiEngine file:", ME.__file__)
print("Preflight CONDA_PREFIX:", os.environ.get("CONDA_PREFIX", ""))
PY

RUN_TEST_CMD=(
  python "$PROJECT_ROOT/visualize/run_test_line.py"
  --data_root "$DATA_ROOT"
  --patch_size "$PATCH_SIZE"
  --sigma "$SIGMA"
  --clip "$CLIP"
  --split "$SPLIT"
  --checkpoint_dir "$CHECKPOINT_DIR"
  --save_dir "$RESULT_DIR"
  --patch_prob_th "$PATCH_PROB_TH"
  --line_prob_th "$LINE_PROB_TH"
  --vertex_nms_th "$VERTEX_NMS_TH"
  --line_dist_th "$LINE_DIST_TH"
)

if [ "$OVERWRITE" = "1" ]; then
  RUN_TEST_CMD+=(--overwrite)
fi

echo "Running prediction..."
"${RUN_TEST_CMD[@]}"

echo "Running OBJ export..."
python "$PROJECT_ROOT/visualize/visualize_line.py" \
  --patch_size "$PATCH_SIZE" \
  --sigma "$SIGMA" \
  --clip "$CLIP" \
  --result_dir "$RESULT_DIR" \
  --save_dir "$OBJ_DIR" \
  --vertex_prob_th "$VERTEX_PROB_TH" \
  --vertex_nms_th "$VIS_VERTEX_NMS_TH" \
  --line_prob_th "$VIS_LINE_PROB_TH" \
  --line_len_th "$VIS_LINE_LEN_TH" \
  --line_nms_th "$VIS_LINE_NMS_TH" \
  --merge_th "$MERGE_TH"

obj_count="$(find "$OBJ_DIR" -maxdepth 1 -type f -name '*_pred.obj' | wc -l | tr -d ' ')"
echo "Done. Exported OBJ count: $obj_count"
echo "OBJ output dir: $OBJ_DIR"

if [ "$EVAL" = "1" ]; then
  echo "Running wireframe evaluation..."
  python "$PROJECT_ROOT/visualize/evaluate_wireframe.py" \
    --data_root "$DATA_ROOT" \
    --split "$SPLIT" \
    --obj_dir "$OBJ_DIR" \
    --vertex_th "$EVAL_VERTEX_TH" \
    --edge_th "$EVAL_EDGE_TH" \
    --report_path "$EVAL_REPORT"
fi
