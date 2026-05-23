#!/usr/bin/env bash
#SBATCH --job-name=pc2wf-gen
#SBATCH --output=logs/gen_%j.out
#SBATCH --error=logs/gen_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Avoid `set -u` because conda package activate/deactivate hooks may read
# backup vars that were never defined in the current shell.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}}"
DATA_ROOT="${DATA_ROOT:-/geogfs1/groups/hkurs/u3666068mgh/Tallin}"
SIGMA="${SIGMA:-0.01}"
CLIP="${CLIP:-0.01}"
PATCH_SIZE="${PATCH_SIZE:-50}"
NUM_WORKERS="${NUM_WORKERS:-1}"
REBUILD="${REBUILD:-1}"
SKIP_NOISE="${SKIP_NOISE:-0}"
GRAPH_RADIUS="${GRAPH_RADIUS:-0.05}"
QUANTIZATION_SIZE="${QUANTIZATION_SIZE:-0.03}"
PATCH_VERTEX_THRESHOLD="${PATCH_VERTEX_THRESHOLD:-0.01}"
LINE_DIST_THRESHOLD="${LINE_DIST_THRESHOLD:-0.03}"
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

export OMP_NUM_THREADS
export PYTHONNOUSERSITE
export CUDA_HOME="${CONDA_PREFIX:-}"
if [ -n "$CUDA_HOME" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
  # Prefer conda's C++ runtime to satisfy MinkowskiEngine's libstdc++ requirement.
  if [ -f "$CUDA_HOME/lib/libstdc++.so.6" ]; then
    export LD_PRELOAD="$CUDA_HOME/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
fi

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "CONDA_ENV=$CONDA_ENV"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "SIGMA=$SIGMA CLIP=$CLIP PATCH_SIZE=$PATCH_SIZE NUM_WORKERS=$NUM_WORKERS"
echo "REBUILD=$REBUILD SKIP_NOISE=$SKIP_NOISE"
echo "GRAPH_RADIUS=$GRAPH_RADIUS QUANTIZATION_SIZE=$QUANTIZATION_SIZE PATCH_VERTEX_THRESHOLD=$PATCH_VERTEX_THRESHOLD LINE_DIST_THRESHOLD=$LINE_DIST_THRESHOLD"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "LD_PRELOAD=${LD_PRELOAD:-}"

python - <<'PY'
import os
import MinkowskiEngine as ME

print("Preflight MinkowskiEngine:", getattr(ME, "__version__", "unknown"))
print("Preflight MinkowskiEngine file:", ME.__file__)
print("Preflight CONDA_PREFIX:", os.environ.get("CONDA_PREFIX", ""))
PY

NOISE_CMD=(
  python "$PROJECT_ROOT/gen_data/noise_addnoise.py"
  --data_root "$DATA_ROOT"
  --sigma "$SIGMA"
  --clip "$CLIP"
  --num_workers "$NUM_WORKERS"
)

PATCH_CMD=(
  python "$PROJECT_ROOT/gen_data/noise_gen_patch_straight.py"
  --data_root "$DATA_ROOT"
  --patch_size "$PATCH_SIZE"
  --sigma "$SIGMA"
  --clip "$CLIP"
  --num_workers "$NUM_WORKERS"
  --graph_radius "$GRAPH_RADIUS"
  --quantization_size "$QUANTIZATION_SIZE"
  --patch_vertex_threshold "$PATCH_VERTEX_THRESHOLD"
  --line_dist_threshold "$LINE_DIST_THRESHOLD"
)

if [ "$REBUILD" = "1" ]; then
  PATCH_CMD+=(--rebuild)
fi

if [ "$SKIP_NOISE" != "1" ]; then
  if [ "$REBUILD" = "1" ]; then
    NOISE_CMD+=(--rebuild)
  fi
  echo "Running noise generation..."
  "${NOISE_CMD[@]}"
else
  echo "Skipping noise generation and reusing existing noisy point clouds."
fi

echo "Running patch generation..."
"${PATCH_CMD[@]}"
