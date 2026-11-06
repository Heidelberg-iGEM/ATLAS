#!/usr/bin/env bash
#
# GVSAM - installation.
#
#   ./setup_env.sh /path/to/project
#
# Creates the folder layout, clones CellSAM and builds one conda environment.
# Unlike the in silico pipeline, everything here runs in a single environment:
# CellSAM, torch and the image readers do not conflict.
#
# A CUDA-capable NVIDIA GPU is needed for training. Inference on a handful of
# images also works on CPU, just slowly.

set -euo pipefail

ROOT="${1:-$PWD}"
PKG="${PKG:-conda}"
ENV_NAME="${ENV_NAME:-gvsam}"
TOOLS="$ROOT/tools"

echo "Installing into $ROOT"

# --- 1  Folders --------------------------------------------------------------
# 00_pipeline  the scripts, config.yaml and visualize.ipynb
# 01_raw       raw microscopy images
# 02_masks_gv  hand-drawn instance masks, one per image
# 03_splits    train/val/test TSVs and the augmented copies
# 04_runs      one folder per training run: checkpoint, losses, results
mkdir -p "$ROOT"/{00_pipeline,01_raw,02_masks_gv,03_splits,04_runs} "$TOOLS"

# --- 2  Environment ----------------------------------------------------------
if ! $PKG env list | grep -q "$ENV_NAME"; then
    $PKG create -y -n "$ENV_NAME" python=3.11
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"

    # Torch with CUDA. Adjust the index URL to the CUDA version of your card.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    # Image IO and annotation handling
    pip install numpy pandas tifffile scikit-image scipy matplotlib tqdm pyyaml
    pip install bioio bioio-czi          # .czi microscope files
    pip install jupyter nbconvert        # for the visualisation stage

    conda deactivate
else
    echo "environment $ENV_NAME already exists, skipped"
fi

# --- 3  CellSAM --------------------------------------------------------------
# GVSAM is a fine-tuned CellSAM, so the package has to be importable and the
# base checkpoint has to be on disk.
if [ ! -d "$TOOLS/cellSAM" ]; then
    git clone https://github.com/vanvalenlab/cellSAM.git "$TOOLS/cellSAM"
    [ -z "${CELLSAM_REF:-}" ] || git -C "$TOOLS/cellSAM" checkout "$CELLSAM_REF"
    (
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"
        pip install -e "$TOOLS/cellSAM"
    )
else
    echo "cellSAM already cloned, skipped"
fi

# The base checkpoint is downloaded on first use into ~/.deepcell/models/.
# Trigger that here so the first training run does not stall on a download.
CONDA_BASE=$(conda info --base)
"$CONDA_BASE/envs/$ENV_NAME/bin/python" - <<'PY' || echo "  download it manually before the first run"
from cellSAM.model import get_model
get_model(None)
print("base checkpoint ready")
PY

# --- 4  Annotation tool (optional, local machine) -----------------------------
# napari is only needed where the masks are drawn, not on a headless compute server:
#   pip install "napari[all]"

cat <<EOF

Done. Put this into config.yaml:

env:
  python: $CONDA_BASE/envs/$ENV_NAME/bin/python

models:
  cellsam_ckpt: ~/.deepcell/models/cellsam_v1.2/cellsam_general.pt

Check before the first run:

  $CONDA_BASE/envs/$ENV_NAME/bin/python -c "import cellSAM, torch; print(torch.cuda.is_available())"

Then copy the scripts to $ROOT/00_pipeline/, the images to $ROOT/01_raw/,
the masks to $ROOT/02_masks_gv/, and start with:

  cd $ROOT/00_pipeline && ./run_pipeline.sh
EOF
