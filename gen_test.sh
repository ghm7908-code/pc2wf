#!/usr/bin/env bash
#SBATCH --job-name=pc2wf-gentest
#SBATCH --output=logs/gentest_%j.out
#SBATCH --error=logs/gentest_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Regenerate test patches filtered by test_list.
# Only processes the test split — does NOT touch train/validation.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}}"

DATA_ROOT="${DATA_ROOT:-/geogfs1/groups/hkurs/u3666068mgh/Tallin}"
TEST_LIST="${TEST_LIST:-$DATA_ROOT/test/test_list.txt}"
PATCH_SIZE="${PATCH_SIZE:-32}"
SIGMA="${SIGMA:-0.01}"
CLIP="${CLIP:-0.01}"
NUM_WORKERS="${NUM_WORKERS:-4}"
REBUILD="${REBUILD:-1}"
SKIP_NOISE="${SKIP_NOISE:-0}"
GRAPH_RADIUS="${GRAPH_RADIUS:-0.05}"
QUANTIZATION_SIZE="${QUANTIZATION_SIZE:-0.03}"
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

echo "============================================"
echo "Regenerating test patches (filtered by test_list)"
echo "============================================"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "TEST_LIST=$TEST_LIST"
echo "PATCH_SIZE=$PATCH_SIZE SIGMA=$SIGMA CLIP=$CLIP"
echo "NUM_WORKERS=$NUM_WORKERS REBUILD=$REBUILD SKIP_NOISE=$SKIP_NOISE"

if [ ! -f "$TEST_LIST" ]; then
  echo "ERROR: test_list not found at $TEST_LIST" >&2
  exit 1
fi

TEST_LIST_COUNT=$(wc -l < "$TEST_LIST" | tr -d ' ')
echo "test_list contains $TEST_LIST_COUNT names"

# ── Step 1: Add noise ──────────────────────────────────────────
NOISE_DIR="$DATA_ROOT/noise_sigma${SIGMA}clip${CLIP}"

if [ "$SKIP_NOISE" != "1" ]; then
  NOISE_FLAGS=(--data_root "$DATA_ROOT" --sigma "$SIGMA" --clip "$CLIP" --num_workers "$NUM_WORKERS" --split test --names_file "$TEST_LIST")

  # Manually remove only the test noise subdirectory (--rebuild would wipe ALL splits)
  if [ "$REBUILD" = "1" ] && [ -d "$NOISE_DIR/test" ]; then
    echo "Removing old test noise directory: $NOISE_DIR/test"
    rm -rf "$NOISE_DIR/test"
  fi

  echo ""
  echo "Step 1/2: Adding noise to test clouds..."
  echo "  Output: $NOISE_DIR/test/"
  python "$PROJECT_ROOT/gen_data/noise_addnoise.py" "${NOISE_FLAGS[@]}"
else
  echo ""
  echo "Step 1/2: Skipping noise generation (SKIP_NOISE=1)"
  echo "  Expecting noisy xyz files at: $NOISE_DIR/test/xyz/"
  if [ ! -d "$NOISE_DIR/test/xyz" ]; then
    echo "ERROR: Noise directory not found: $NOISE_DIR/test/xyz" >&2
    exit 1
  fi
fi

# ── Step 2: Generate patches ────────────────────────────────────
PATCH_DIR="$DATA_ROOT/patches_${PATCH_SIZE}_noise_sigma${SIGMA}clip${CLIP}"

PATCH_FLAGS=(
  --data_root "$DATA_ROOT"
  --patch_size "$PATCH_SIZE"
  --sigma "$SIGMA"
  --clip "$CLIP"
  --num_workers "$NUM_WORKERS"
  --graph_radius "$GRAPH_RADIUS"
  --quantization_size "$QUANTIZATION_SIZE"
  --split test
  --names_file "$TEST_LIST"
)

if [ "$REBUILD" = "1" ]; then
  # Only rebuild the test subdirectory, not the entire patches dir
  if [ -d "$PATCH_DIR/test" ]; then
    echo "Removing old test patch directory: $PATCH_DIR/test"
    rm -rf "$PATCH_DIR/test"
  fi
fi

echo ""
echo "Step 2/2: Generating patches for test clouds..."
echo "  Output: $PATCH_DIR/test/"

python "$PROJECT_ROOT/gen_data/noise_gen_patch_straight.py" "${PATCH_FLAGS[@]}"

# ── Verify ─────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Verification"
echo "============================================"

DOWN_COUNT=$(find "$PATCH_DIR/test" -maxdepth 1 -name '*.down' | wc -l | tr -d ' ')
PATCH_INDEX_COUNT=$(find "$PATCH_DIR/test" -maxdepth 1 -name '*.patch_index' | wc -l | tr -d ' ')
echo "Generated .down files:    $DOWN_COUNT"
echo "Generated .patch_index:   $PATCH_INDEX_COUNT"

MISSING=$(comm -23 <(sort "$TEST_LIST") <(find "$PATCH_DIR/test" -maxdepth 1 -name '*.down' -exec basename {} .down \; | sort) | wc -l | tr -d ' ')
if [ "$MISSING" -gt 0 ]; then
  echo "WARNING: $MISSING test_list names have NO corresponding .down file!"
  echo "Missing names:"
  comm -23 <(sort "$TEST_LIST") <(find "$PATCH_DIR/test" -maxdepth 1 -name '*.down' -exec basename {} .down \; | sort) | head -20
  if [ "$MISSING" -gt 20 ]; then
    echo "  ... and $(($MISSING - 20)) more"
  fi
else
  echo "All test_list names have corresponding .down files."
fi

echo ""
echo "Done. Test patches ready at: $PATCH_DIR/test/"
echo "Ready to run: sbatch infer.sh"
