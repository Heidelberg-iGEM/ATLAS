#!/usr/bin/env bash
#
# GvpC evolution - Installation.
#
#   ./setup_envs.sh /path/to/project
#
# Creates the folder structure, clones the three external tools and builds four
# separate Python environments. At the end it prints the paths that have to be
# entered in config.yaml under envs:.
#
# WHY FOUR ENVIRONMENTS
# LigandMPNN, AlphaFold2, Boltz and PyRosetta have dependencies that are
# mutually exclusive (different JAX, Torch and NumPy versions).
# A shared environment cannot be built. That is why
# run_pipeline.sh runs every stage with its own interpreter.
#
# HARDWARE
# A CUDA-capable NVIDIA GPU is required (AlphaFold2 and Boltz). Disk space
# about 20 GB for environments and weights, of which 5.3 GB alone for the
# AlphaFold2 parameters. The PyRosetta step runs on CPU and benefits from
# many cores.
#
# The script is idempotent: existing environments and repos are
# skipped, so it can be rerun after a failure.

set -euo pipefail

ROOT="${1:-$PWD}"
CUDA="${CUDA:-12.4}"          # must match the graphics card
PKG="${PKG:-conda}"           # conda or mamba
TOOLS="$ROOT/tools"

# SETUP_ONLY="boltz foundry bindcraft rosetta" limits the build to some components
# (used by the repository-level install.sh). Empty means all four.
want() { [ -z "${SETUP_ONLY:-}" ] || [[ " $SETUP_ONLY " == *" $1 "* ]]; }

echo "Installing to $ROOT  (CUDA $CUDA, $PKG)"

# --- 1  Folder structure -------------------------------------------------------
# 00_pipeline   the scripts of this pipeline
# 02_input      template, precomputed MSA, generated Boltz inputs
# 03_output     raw output of the tools
# 04_loops      the tables: one round per subfolder
mkdir -p "$ROOT"/{00_pipeline,02_input/templates,03_output,04_loops} "$TOOLS"

if want boltz; then
# --- 2  Boltz ----------------------------------------------------------------
# Structure prediction with template and pocket constraint.
# The evaluation stages that need gemmi also run in this environment.
if ! $PKG env list | grep -q "evolution_boltz"; then
    $PKG create -y -n evolution_boltz python=3.11
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate evolution_boltz
    pip install "boltz[cuda]" -U
    pip install gemmi numpy pyyaml
    conda deactivate
else
    echo "evolution_boltz already exists, skipped"
fi
fi

if want foundry; then
# --- 3  LigandMPNN and RFdiffusion3 (RosettaCommons foundry) ------------------
# Installed from PyPI (pinned, see third_party.lock) with uv, not with conda.
# $TOOLS/foundry doubles as the foundry workspace (input_jsons/, scripts/,
# output/) that the latch pipeline expects; the environment lives in .venv.
FOUNDRY_VERSION="${FOUNDRY_VERSION:-0.2.0}"
mkdir -p "$TOOLS/foundry"
if [ ! -x "$TOOLS/foundry/.venv/bin/python" ]; then
    (
        cd "$TOOLS/foundry"
        command -v uv >/dev/null || pip install uv
        uv python install 3.12
        uv venv --python 3.12
        # shellcheck disable=SC1091
        source .venv/bin/activate
        uv pip install "rc-foundry[all]==$FOUNDRY_VERSION" matplotlib scipy pyyaml numpy
        foundry install rfd3        # RFdiffusion3 checkpoint -> ~/.foundry/checkpoints
    )
else
    echo "foundry environment already exists, skipped"
fi

# Model weights. The pipeline runs with ligand_mpnn; the other two are
# only needed if mpnn.model in config.yaml is switched.
# is_legacy_weights is True in 10_mpnn_design.py because these are the weights
# from the original repositories.
mkdir -p "$TOOLS/mpnn_weights"
for w in ligandmpnn_v_32_010_25.pt proteinmpnn_v_48_020.pt solublempnn_v_48_020.pt; do
    [ -f "$TOOLS/mpnn_weights/$w" ] || \
        wget -q -P "$TOOLS/mpnn_weights" "https://files.ipd.uw.edu/pub/ligandmpnn/$w"
done
fi

if want bindcraft; then
# --- 4  AlphaFold2 via BindCraft -------------------------------------------
# BindCraft brings ColabDesign and the AlphaFold2 weights, and exactly those
# two are needed by 21_af2_score.py. The bundled installer builds its own
# conda environment.
# --cuda must match the graphics card. If it is left empty, the installer
# may choose the wrong version, and the error is only noticed on the first
# GPU run.
if [ ! -d "$TOOLS/BindCraft" ]; then
    git clone https://github.com/martinpacesa/BindCraft "$TOOLS/BindCraft"
    [ -z "${BINDCRAFT_REF:-}" ] || git -C "$TOOLS/BindCraft" checkout "$BINDCRAFT_REF"
    (
        cd "$TOOLS/BindCraft"
        bash install_bindcraft.sh --cuda "$CUDA" --pkg_manager "$PKG"
    )
else
    echo "BindCraft already exists, skipped"
fi
fi

if want rosetta; then
# --- 5  PyRosetta ------------------------------------------------------------
# Needs a licence; free for academic use, see
# https://www.pyrosetta.org/downloads
if ! $PKG env list | grep -q "evolution_rosetta"; then
    $PKG create -y -n evolution_rosetta python=3.11
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate evolution_rosetta
    pip install pyrosetta \
        --find-links https://west.rosettacommons.org/pyrosetta/quarterly/release
    conda deactivate
else
    echo "evolution_rosetta already exists, skipped"
fi
fi

# --- 6  The paths for config.yaml -------------------------------------------
CONDA_BASE=$(conda info --base)
BINDCRAFT_PY=$(ls -d "$TOOLS"/BindCraft/.venv/bin/python 2>/dev/null \
               || echo "$CONDA_BASE/envs/BindCraft/bin/python")

cat <<EOF

Done. Enter this block in config.yaml under envs:

envs:
  mpnn_py:    $TOOLS/foundry/.venv/bin/python
  boltz_py:   $CONDA_BASE/envs/evolution_boltz/bin/python
  boltz_bin:  $CONDA_BASE/envs/evolution_boltz/bin/boltz
  af2_py:     $BINDCRAFT_PY
  ros_py:     $CONDA_BASE/envs/evolution_rosetta/bin/python
  af2_params: $TOOLS/BindCraft

Check two things before the first run:

  ls $TOOLS/BindCraft/params/ | head
      af2_params is the PARENT folder of params/, not params/ itself -
      ColabDesign appends '/params' itself. With the wrong value it searches
      params/params/, and that is only noticed after the GPU waiting time.

  $CONDA_BASE/envs/evolution_boltz/bin/python -c "import gemmi, numpy; print('ok')"
      The evaluation stages run in this environment and need both.

Then put the scripts into $ROOT/00_pipeline/, the template into
$ROOT/02_input/templates/, fill in config.yaml and start:

  cd $ROOT/00_pipeline && ./run_pipeline.sh
EOF
