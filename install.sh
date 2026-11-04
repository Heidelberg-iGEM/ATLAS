#!/usr/bin/env bash
#
# Installer for the Drylab pipelines.
#
#   ./install.sh [--dry-run] [--cuda 12.4] [PART ...]
#
#   PART   all (default) | evolution | gvpa | latch | gvsam | gas
#
#   evolution  GvpC in silico evolution    envs: foundry (uv), evolution_boltz, BindCraft, evolution_rosetta
#   gvpa   GvpA binder design              envs: foundry (uv), evolution_boltz, BindCraft
#   latch  biosensor latch design          envs: foundry (uv), flexcraft (conda)
#   gvsam  GV cluster detection            env:  gvsam (conda)
#   gas    collapse-pressure analysis      env:  gas (conda)
#
# What it does
#   - fetches third-party code at the versions pinned in third_party.lock into third_party/
#   - builds the conda / uv environments (each part uses its own, because the tools
#     need conflicting JAX / Torch / NumPy versions)
#   - downloads model weights
#   - creates a project folder per part under work/ with the repository's scripts
#     linked in and a ready-to-edit config
#   - writes drylab.env, which drylab.sh reads to find every interpreter
#
# It is idempotent: whatever exists is skipped, so it can be rerun after a failure.
# --dry-run prints every command without executing it.
#
# Needs: git, wget, conda or mamba (PKG=mamba), and a CUDA-capable NVIDIA GPU for the
# structure-prediction stages. PyRosetta needs a licence (free for academic use).

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$REPO/third_party.lock"

TP="$REPO/third_party"
WORK="$REPO/work"
PKG="${PKG:-conda}"
CUDA="${CUDA:-$CUDA_VERSION}"
DRY=0
PARTS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1 ;;
        --cuda) shift; CUDA="$1" ;;
        -h|--help) sed -n 2,30p "$0"; exit 0 ;;
        all|evolution|gvpa|latch|gvsam|gas) PARTS+=("$1") ;;
        *) echo "unknown argument: $1 (see --help)"; exit 1 ;;
    esac
    shift
done
[ ${#PARTS[@]} -gt 0 ] || PARTS=(all)
[[ " ${PARTS[*]} " == *" all "* ]] && PARTS=(evolution gvpa latch gvsam gas)
want() { [[ " ${PARTS[*]} " == *" $1 "* ]]; }

run() { echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }
say() { printf '\n=== %s\n' "$*"; }

# --- preflight ---------------------------------------------------------------
missing=0
for c in git wget "$PKG"; do
    command -v "$c" >/dev/null 2>&1 || { echo "missing command: $c"; missing=1; }
done
if [ "$missing" = 1 ] && [ "$DRY" = 0 ]; then
    echo "Install the missing commands first (conda: https://github.com/conda-forge/miniforge)."
    exit 1
fi
command -v nvidia-smi >/dev/null 2>&1 || echo "note: no nvidia-smi found; GPU stages will not run on this machine."

if command -v "$PKG" >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
else
    CONDA_BASE="$HOME/miniforge3"      # only reached with --dry-run
fi
export PKG CUDA FOUNDRY_VERSION BINDCRAFT_REF CELLSAM_REF
[ "$DRY" = 1 ] || mkdir -p "$TP" "$WORK"
ENVFILE="$REPO/drylab.env"

# --- helpers -----------------------------------------------------------------
# clone <url> <dir> <ref>
clone() {
    if [ -d "$2/.git" ]; then echo "$2 exists, skipped"; return; fi
    run git clone "$1" "$2"
    run git -C "$2" checkout "$3"
}

# Link the repository's files into a project folder (edits stay in the repository).
link_into() {   # link_into <dest dir> <src files...>
    local dest="$1"; shift
    run mkdir -p "$dest"
    for f in "$@"; do run ln -sfn "$f" "$dest/$(basename "$f")"; done
}

# Copy an example config once, with the paths of this installation filled in.
make_config() { # make_config <example> <target> <sed expressions...>
    local ex="$1" target="$2"; shift 2
    if [ -f "$target" ]; then echo "$target exists, kept"; return; fi
    echo "+ create $target from $(basename "$ex")"
    [ "$DRY" = 1 ] || sed "$@" "$ex" > "$target"
}

base_envs() {   # base_envs <components...>   (boltz foundry bindcraft rosetta)
    say "Environments: $*"
    run mkdir -p "$WORK/evolution"
    run ln -sfn "$TP" "$WORK/evolution/tools"      # one tools/ folder shared by evolution and gvpa
    run env SETUP_ONLY="$*" "$REPO/gvpc-evolution/setup_envs.sh" "$WORK/evolution"
    # The legacy LigandMPNN weights are fetched by setup_envs.sh into third_party/mpnn_weights;
    # foundry looks in ~/.foundry/checkpoints.
    if [[ " $* " == *" foundry "* ]]; then
        run mkdir -p "$HOME/.foundry/checkpoints"
        for w in ligandmpnn_v_32_010_25.pt proteinmpnn_v_48_020.pt solublempnn_v_48_020.pt; do
            [ -e "$HOME/.foundry/checkpoints/$w" ] || run ln -s "$TP/mpnn_weights/$w" "$HOME/.foundry/checkpoints/$w"
        done
    fi
}

# --- parts ---------------------------------------------------------------------
if want evolution; then
    base_envs boltz foundry bindcraft rosetta
    say "GvpC evolution project folder: $WORK/evolution"
    link_into "$WORK/evolution/00_pipeline" \
        "$REPO"/gvpc-evolution/*.py "$REPO"/gvpc-evolution/run_pipeline.sh \
        "$REPO"/gvpc-evolution/config.example.yaml
    # Template (Serratia GvpC repeat 3-4 on four GvpA subunits) and the precomputed GvpA target MSA
    for f in "$REPO"/gvpc-evolution/inputs/SerGvpCR3_*; do run ln -sfn "$f" "$WORK/evolution/02_input/templates/$(basename "$f")"; done
    run ln -sfn "$REPO/gvpc-evolution/inputs/gvpa_target.csv" "$WORK/evolution/02_input/gvpa_target.csv"
    make_config "$REPO/gvpc-evolution/config.example.yaml" "$WORK/evolution/00_pipeline/config.yaml" \
        -e "s|~/evolution_project/tools|$TP|g" -e "s|~/evolution_project|$WORK/evolution|g" -e "s|~/miniforge3|$CONDA_BASE|g"
fi

if want gvpa; then
    base_envs boltz foundry bindcraft
    say "GvpA binder design package: $REPO/gvpa-binder-design"
    # The package is used in place (common.py resolves configs/ and inputs/ next to scripts/).
    run mkdir -p "$REPO/gvpa-binder-design"/{configs,inputs,logs}
fi

if want latch; then
    base_envs foundry
    say "FlexCraft environment (conda env flexcraft)"
    clone "$FLEXCRAFT_REPO" "$TP/flexcraft" "$FLEXCRAFT_REF"
    if [ ! -x "$CONDA_BASE/envs/flexcraft/bin/python" ]; then
        run "$PKG" create -y -n flexcraft python=3.12
    else
        echo "conda env flexcraft exists, skipped"
    fi
    FPIP="$CONDA_BASE/envs/flexcraft/bin/pip"
    run "$FPIP" install -e "$TP/flexcraft" "jax[cuda12]" gemmi numpy pyyaml
    # PyRosetta is imported by score_gfp_fusion_partners.py (interface metrics). Licence required.
    run "$FPIP" install pyrosetta --find-links https://west.rosettacommons.org/pyrosetta/quarterly/release
    if [ ! -d "$TP/flexcraft/params/af" ]; then
        run bash -c "cd '$TP/flexcraft' && bash download_params.sh"     # AlphaFold2, salad and MPNN parameters
    fi
    say "Latch pipeline files go where the FlexCraft layout expects them"
    PIPE="$TP/flexcraft/flexcraft/pipelines/dual_binder_pipeline"
    run mkdir -p "$PIPE/scripts" "$PIPE/inputs" "$TP/flexcraft/filters"
    for f in "$REPO"/biosensor-latch/scripts/*.py; do run ln -sfn "$f" "$PIPE/scripts/$(basename "$f")"; done
    for f in "$REPO"/biosensor-latch/filters/*.json; do run ln -sfn "$f" "$TP/flexcraft/filters/$(basename "$f")"; done
    run mkdir -p "$TP/foundry/input_jsons/example" "$TP/foundry/input_pdbs" "$TP/foundry/scripts"
    # The example YAMLs name the motif PDB by absolute path; point them at this installation.
    for f in "$REPO"/biosensor-latch/inputs/*.yaml; do
        make_config "$f" "$TP/foundry/input_jsons/example/$(basename "$f")" \
            -e "s|/home/hd/hd_hd/hd_ps334/workspace/foundry|$TP/foundry|g"
    done
    # Patched ProteinMPNN driver (--redesign-terminus, --fixed-end-sequence) and the motif PDBs of the examples
    run ln -sfn "$REPO/biosensor-latch/foundry/batch_partial_mpnn.py" "$TP/foundry/scripts/batch_partial_mpnn.py"
    for f in "$REPO"/biosensor-latch/inputs/pdbs/*.pdb; do run ln -sfn "$f" "$TP/foundry/input_pdbs/$(basename "$f")"; done
    run mkdir -p "$HOME/.foundry/checkpoints"
fi

if want gvsam; then
    say "GVSAM"
    run env CELLSAM_REF="$CELLSAM_REF" PKG="$PKG" "$REPO/gvsam/setup_env.sh" "$WORK/gvsam"
    link_into "$WORK/gvsam/00_pipeline" \
        "$REPO"/gvsam/*.py "$REPO"/gvsam/run_pipeline.sh "$REPO"/gvsam/visualize.ipynb
    make_config "$REPO/gvsam/config.yaml" "$WORK/gvsam/00_pipeline/config.yaml" \
        -e "s|root: /path/to/project|root: $WORK/gvsam|" \
        -e "s|python: ~/.conda/envs/gvsam/bin/python|python: $CONDA_BASE/envs/gvsam/bin/python|"
fi

if want gas; then
    say "GAS (conda env gas)"
    if [ ! -x "$CONDA_BASE/envs/gas/bin/python" ]; then
        run "$PKG" create -y -n gas python=3.11
        run "$CONDA_BASE/envs/gas/bin/pip" install numpy pandas scipy matplotlib jupyter
    else
        echo "conda env gas exists, skipped"
    fi
fi

# --- drylab.env ----------------------------------------------------------------
say "Writing $ENVFILE"
if [ "$DRY" = 0 ]; then
    cat > "$ENVFILE" <<EOF
# Written by install.sh. Read by drylab.sh and gvpa-binder-design/env.sh.
CONDA_BASE=$CONDA_BASE
TP=$TP
WORK=$WORK
# evolution
EVOLUTION_PROJECT=$WORK/evolution
# gvpa (the guide's env.sh variables)
FOUNDRY_ENV=$TP/foundry/.venv
BINDCRAFT_ENV=$CONDA_BASE/envs/BindCraft
BOLTZ_BIN=$CONDA_BASE/envs/evolution_boltz/bin/boltz
BINDCRAFT_DIR=$TP/BindCraft
RFD3_CKPT=\$HOME/.foundry/checkpoints/rfd3_latest.ckpt
MPNN_CKPT=$TP/mpnn_weights/solublempnn_v_48_020.pt
AF2_PARAMS_DIR=$TP/BindCraft
GVPA_PACKAGE=$REPO/gvpa-binder-design
# latch
FLEXCRAFT_DIR=$TP/flexcraft
FLEXCRAFT_ENV=$CONDA_BASE/envs/flexcraft
FOUNDRY_WORKSPACE=$TP/foundry
# gvsam
GVSAM_PROJECT=$WORK/gvsam
# gas
GAS_ENV=$CONDA_BASE/envs/gas
EOF
else
    echo "(dry run: not written)"
fi

cat <<EOF

Done. Next steps:
  ./drylab.sh doctor              check that everything the selected parts need is in place
  ./drylab.sh help                list the commands of every part
  README.md                       tutorial for each part
EOF
