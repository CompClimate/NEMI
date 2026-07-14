#!/usr/bin/env bash
set -euo pipefail

# ======================================================================
# Rebuilds the RAPIDS cuML + PyTorch env on NRP JupyterHub.
# Everything (cache + temp) stays on /home to avoid ephemeral eviction.
# Run inside tmux:  tmux new -s build; bash build_cuml_env_nrp.sh
# ======================================================================

# ---- config ----------------------------------------------------------
ENV_PREFIX="$HOME/conda_envs/main_cuml"
PKGS_DIR="$HOME/conda_envs/.pkgs"          # cache on HOME (same fs as env -> hardlinks)
TMP_DIR="$HOME/conda_tmp"                   # temp on HOME (off ephemeral overlay)
PY_VER="3.12"
DBOF_REF="cutouts_data_v2"                  # change to 'main' once merged
NEMI_REPO="$HOME/git/NEMI"                   # local NEMI clone (editable install)
NEMI_URL="https://github.com/CompClimate/NEMI.git"
NEMI_REF="GPU-workflow"
TARGET_TRIPLE="targets/x86_64-linux"        # where conda puts CUDA headers

SOLVER="conda"; command -v mamba >/dev/null 2>&1 && SOLVER="mamba"
echo ">> using solver: $SOLVER"

# ---- network resilience (repodata fetches have timed out here) --------
conda config --set remote_connect_timeout_secs 60
conda config --set remote_read_timeout_secs 300
conda config --set remote_max_retries 8
conda config --set remote_backoff_factor 2

# ---- 1. clean up any failed run --------------------------------------
echo ">> cleaning previous attempts"
rm -rf "$ENV_PREFIX" "$PKGS_DIR" "$TMP_DIR" /tmp/conda_pkgs /tmp/conda_tmp 2>/dev/null || true
rm -rf /opt/conda/pkgs/* 2>/dev/null || true
conda clean --packages --tarballs -y || true       # keep repodata index cached across retries
pip cache purge 2>/dev/null || true
rm -rf "$HOME/.cache/pip" 2>/dev/null || true

# ---- 2. direct cache + temp to HOME (never ephemeral) ----------------
mkdir -p "$PKGS_DIR" "$TMP_DIR"
export CONDA_PKGS_DIRS="$PKGS_DIR"
export TMPDIR="$TMP_DIR"
echo ">> free space before build:"; df -h "$HOME" | tail -1

# ---- 3. create the env -----------------------------------------------
# Pins learned the hard way:
#   numpy<2.5        -> numba (via cuml) ceiling
#   scikit-learn=1.5 -> cuml 25.06 sklearn-compat shim
#   umap-learn<0.5.7 -> 0.5.7+ needs sklearn 1.6 (installed in pip step below)
#   cuda-*-dev/cccl/nvrtc -> headers cupy needs to JIT-compile kernels
echo ">> creating env at $ENV_PREFIX"
"$SOLVER" create -p "$ENV_PREFIX" -y \
    -c rapidsai -c conda-forge -c nvidia \
    "python=$PY_VER" "cuda-version=12.*" \
    cuml=25.06 cupy \
    cuda-cudart-dev cuda-cccl cuda-nvrtc \
    pytorch-gpu torchvision \
    "numpy<2.5" "scikit-learn=1.5.*" \
    zarr s3fs xarray dask ipykernel matplotlib

# reclaim downloaded tarballs immediately
conda clean --all -y
rm -rf "$PKGS_DIR"

# ---- 4. activate + add pip-only project deps -------------------------
echo ">> installing pip-only deps"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# dbof: --no-deps so it can't pull pip torch/numpy and re-tangle the CUDA stack
pip install --no-deps --no-cache-dir \
    "dbof-in-native-grid @ git+https://github.com/Sea-Meets-the-Stars/llc4320-native-grid-preprocessing.git@${DBOF_REF}"

# umap-learn<0.5.7 for NEMI: 0.5.7+ calls sklearn 1.6's ensure_all_finite, but
# cuML pins scikit-learn to 1.5.x here.  (Also brings tqdm/pynndescent for NEMI.)
pip install --no-cache-dir einops timm==0.3.2 xmitgcm torchinfo "umap-learn<0.5.7"

# NEMI (editable) for GPU cuML UMAP + clustering.
if [ ! -f "$NEMI_REPO/pyproject.toml" ]; then
    echo ">> cloning NEMI to $NEMI_REPO"
    git clone --branch "$NEMI_REF" "$NEMI_URL" "$NEMI_REPO"
fi
# Base install only (no [gpu] extra): its cupy-cuda12x would clash with conda's
# cupy, and umap-learn is already pinned above.  Pulls pyyaml for the NEMI CLI.
pip install --no-cache-dir -e "$NEMI_REPO"

# ---- 5. Jupyter kernel + cupy CUDA_PATH safeguard --------------------
# The kernel launches without `conda activate`, so cupy can't find CUDA on
# its own. sitecustomize sets CUDA_PATH at interpreter startup, pointed at
# the TARGETS dir whose include/ actually holds cuda_runtime.h.
echo ">> registering kernel + sitecustomize"
python -m ipykernel install --user --name main_cuml --display-name "Python (cuML + torch)"

cat > "$ENV_PREFIX/lib/python${PY_VER}/site-packages/sitecustomize.py" <<EOF
import os
os.environ["CUDA_PATH"] = "$ENV_PREFIX/$TARGET_TRIPLE"
EOF

# ---- 6. verify -------------------------------------------------------
echo ">> verifying"
python - <<'PY'
import os, numpy, torch, cupy, cuml, sklearn, umap, nemi
os.environ.setdefault("CUDA_PATH", "")  # sitecustomize normally sets this
print("python       ", __import__("sys").version.split()[0])
print("numpy        ", numpy.__version__)
print("scikit-learn ", sklearn.__version__)
print("umap-learn   ", umap.__version__)
print("nemi         ", getattr(nemi, "__version__", "ok"))
print("torch cuda   ", torch.cuda.is_available())
print("cupy         ", cupy.__version__, "| jit:", int((cupy.arange(5)*2).sum()))  # 20
print("cuml         ", cuml.__version__)
PY

echo ">> DONE. free space after build:"; df -h "$HOME" | tail -1
