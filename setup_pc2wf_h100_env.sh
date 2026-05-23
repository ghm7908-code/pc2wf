#!/usr/bin/env bash
# Avoid set -u here because conda package activate/deactivate hooks
# (notably cuda-nvcc) may read backup vars that are unset.
set -eo pipefail

GROUP_ROOT="${GROUP_ROOT:-/geogfs1/groups/hkurs/u3666068mgh}"
ENV_ROOT="${ENV_ROOT:-$GROUP_ROOT/conda_envs}"
PKGS_DIR="${PKGS_DIR:-$GROUP_ROOT/conda_pkgs}"
TMP_DIR="${TMP_DIR:-$GROUP_ROOT/tmp}"
ENV_NAME="${ENV_NAME:-pc2wf-h100-final}"
ENV_PATH="${ENV_PATH:-$ENV_ROOT/$ENV_NAME}"
ME_SRC_DIR="${ME_SRC_DIR:-$GROUP_ROOT/MinkowskiEngine-src-final}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.1.2}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.16.2}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.1.2}"
CUDA_VERSION="${CUDA_VERSION:-11.8}"
GCC_VERSION="${GCC_VERSION:-11}"
OMP_THREADS="${OMP_THREADS:-8}"
MAX_JOBS_VALUE="${MAX_JOBS_VALUE:-8}"
RECREATE="${RECREATE:-0}"

mkdir -p "$PKGS_DIR" "$ENV_ROOT" "$TMP_DIR"

export CONDA_PKGS_DIRS="$PKGS_DIR"
export CONDA_ENVS_PATH="$ENV_ROOT"
export TMPDIR="$TMP_DIR"

if [ "$RECREATE" = "1" ] && [ -e "$ENV_PATH" ]; then
  echo "Removing existing env at $ENV_PATH"
  rm -rf "$ENV_PATH"
fi

if [ ! -e "$ENV_PATH" ]; then
  conda create -p "$ENV_PATH" "python=$PYTHON_VERSION" -y
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_PATH"

conda install -y \
  "cuda-toolkit=$CUDA_VERSION" \
  openblas-devel \
  ninja \
  cmake \
  -c nvidia -c anaconda -c conda-forge

conda install -y -c conda-forge \
  "gcc_linux-64=$GCC_VERSION" \
  "gxx_linux-64=$GCC_VERSION"

python -m pip install -U pip setuptools wheel
python -m pip install numpy scipy tqdm
python -m pip install \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION" \
  --index-url "https://download.pytorch.org/whl/cu118"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
else:
    raise SystemExit("PyTorch is not seeing the GPU. Stop here.")
if torch.version.cuda is None:
    raise SystemExit("PyTorch CPU build detected. Stop here.")
PY

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export C_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="9.0"
export OMP_NUM_THREADS="$OMP_THREADS"
export MAX_JOBS="$MAX_JOBS_VALUE"

echo "Using nvcc from: $(which nvcc)"
nvcc --version
"$CC" --version
"$CXX" --version

rm -rf "$ME_SRC_DIR"
git clone https://github.com/NVIDIA/MinkowskiEngine.git "$ME_SRC_DIR"
cd "$ME_SRC_DIR"
python setup.py install --blas_include_dirs="$CONDA_PREFIX/include" --blas=openblas

python - <<'PY'
import torch
import MinkowskiEngine as ME
print("torch:", torch.__version__)
print("torch.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("MinkowskiEngine:", getattr(ME, "__version__", "unknown"))
print("MinkowskiEngine file:", ME.__file__)
PY

echo
echo "Environment ready at: $ENV_PATH"
echo "Use it in train.sh with:"
echo "CONDA_ENV=$ENV_PATH PATCH_SIZE=32 sbatch train.sh"
