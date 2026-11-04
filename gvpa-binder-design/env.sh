#!/usr/bin/env bash
# Site settings of the GvpA binder design package (paths and GPU variables).
#
#   source env.sh
#
# The defaults come from drylab.env, which install.sh writes at the repository
# root. Override any variable by exporting it before sourcing this file.
#
#   FOUNDRY_ENV      environment of the driver scripts, rfd3 and MPNN
#   BINDCRAFT_ENV    environment for AF2 and PyRosetta
#   BOLTZ_BIN        Boltz-2 executable
#   BINDCRAFT_DIR    BindCraft clone (for DAlphaBall.gcc)
#   RFD3_CKPT        RFdiffusion3 checkpoint
#   MPNN_CKPT        SolubleMPNN checkpoint (solublempnn_v_48_020.pt)
#   AF2_PARAMS_DIR   folder that CONTAINS params/
#   RUN_ROOT         output location (default: ./runs)
#   CAMPAIGN_CONFIG  alternative campaign config (default: configs/gvpa_serratia.yaml)

# Resolve symlinks (work/gvpa/env.sh points here) to find drylab.env at the repository root.
_src="${BASH_SOURCE[0]}"
while [ -L "$_src" ]; do _src="$(readlink "$_src")"; done
_repo="$(cd "$(dirname "$_src")/.." && pwd -P)"
if [ -f "$_repo/drylab.env" ]; then
    # Variables that are already exported win over drylab.env.
    while IFS='=' read -r _k _v; do
        case "$_k" in ''|'#'*) continue ;; esac
        [ -n "${!_k:-}" ] || export "$_k=$(eval echo "$_v")"
    done < "$_repo/drylab.env"
fi

: "${FOUNDRY_ENV:=$HOME/.conda/envs/foundry}"
: "${BINDCRAFT_ENV:=$HOME/.conda/envs/BindCraft}"
: "${BOLTZ_BIN:=$HOME/.conda/envs/boltz/bin/boltz}"
: "${BINDCRAFT_DIR:=$HOME/BindCraft}"
: "${RFD3_CKPT:=$HOME/checkpoints/rfd3_latest.ckpt}"
: "${MPNN_CKPT:=$HOME/checkpoints/solublempnn_v_48_020.pt}"
: "${AF2_PARAMS_DIR:=$HOME/checkpoints/alphafold_params}"
: "${RUN_ROOT:=$PWD/runs}"
export FOUNDRY_ENV BINDCRAFT_ENV BOLTZ_BIN BINDCRAFT_DIR RFD3_CKPT MPNN_CKPT AF2_PARAMS_DIR RUN_ROOT
[ -z "${CAMPAIGN_CONFIG:-}" ] || export CAMPAIGN_CONFIG

# Without the first variable JAX reserves most of the GPU, cuSolver gets no
# workspace and AF2 silently returns NaN.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_UNIFIED_MEMORY=0
# NumPy in the BindCraft environment needs its own libstdc++.
export LD_LIBRARY_PATH="$BINDCRAFT_ENV/lib:${LD_LIBRARY_PATH:-}"

# Interpreters
export PY="$FOUNDRY_ENV/bin/python"            # driver scripts, rfd3, MPNN, ranking, report
export PY_BINDCRAFT="$BINDCRAFT_ENV/bin/python"  # AlphaFold2 (stages 04, 05, 09) and PyRosetta (06)
