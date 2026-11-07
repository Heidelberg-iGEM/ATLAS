#!/usr/bin/env bash
#
# One entry point for every part of the Drylab pipelines.
#
#   ./atlas.sh help
#   ./atlas.sh doctor [PART]
#   ./atlas.sh evolution calibrate | sharp [FROM] | resume N | selftest | py SCRIPT [ARGS]
#   ./atlas.sh gvpa   COMMAND [ARGS]       (./atlas.sh gvpa help)
#   ./atlas.sh latch  run [ARGS] | stage SCRIPT [ARGS] | check
#   ./atlas.sh gvsam  run [FROM] | stage SCRIPT [ARGS]
#   ./atlas.sh gas    notebook
#
# It reads atlas.env (written by install.sh) to find every interpreter, changes
# into the right folder and calls the scripts of this repository. It adds no logic
# of its own: every command prints the line it runs.

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
ENVFILE="$REPO/atlas.env"

usage() { sed -n 2,14p "$0" | sed 's/^# \{0,1\}//'; }

if [ -f "$ENVFILE" ]; then
    while IFS='=' read -r k v; do
        case "$k" in ''|'#'*) continue ;; esac
        [ -n "${!k:-}" ] || export "$k=$(eval echo "$v")"
    done < "$ENVFILE"
fi

die() { echo "error: $*" >&2; exit 1; }
need_env() { [ -f "$ENVFILE" ] || die "atlas.env not found. Run ./install.sh first."; }
show() { echo "+ $*" >&2; "$@"; }

# Read one interpreter path (key: value) from a config.yaml
yaml_value() { local v; v="$(sed -n "s/^[[:space:]]*$2:[[:space:]]*//p" "$1" | head -n1)"; echo "${v/#\~/$HOME}"; }

# ---------------------------------------------------------------------------
# evolution
# ---------------------------------------------------------------------------
cmd_evolution() {
    need_env
    local sub="${1:-help}"; shift || true
    local d="${EVOLUTION_PROJECT:?}/00_pipeline"
    [ -d "$d" ] || die "$d not found. Run ./install.sh evolution."
    cd "$d"
    case "$sub" in
        calibrate) [ -f config.yaml ] || die "config.yaml missing"; show ./run_pipeline.sh ;;
        sharp)     show ./run_pipeline.sh sharp "${1:-1}" ;;
        resume)    show ./run_pipeline.sh "${MODE:-sharp}" "${1:?stage number}" ;;
        selftest)  show python3 lib_registry.py ;;
        py)
            local s="${1:?script}"; shift
            case "$s" in
                cells_after_filter.py|check_controls.py|lib_registry.py) show python3 "$s" "$@" ;;
                10_*|11_*|12_*|15_*|40_*) show "$(yaml_value config.yaml mpnn_py)" "$s" "$@" ;;
                20_*|21_*)                show "$(yaml_value config.yaml af2_py)" "$s" "$@" ;;
                05_*|30_*|31_*|32_*|33_*) show "$(yaml_value config.yaml boltz_py)" "$s" "$@" ;;
                60_*)                     show "$(yaml_value config.yaml ros_py)" "$s" "$@" ;;
                *) die "unknown script $s" ;;
            esac ;;
        help|*)    cat <<EOF
evolution commands (run in $d):
  calibrate         ./run_pipeline.sh               calibration run, nothing selected
  sharp [FROM]      ./run_pipeline.sh sharp [FROM]  gate applied, order.csv written
  resume N          ./run_pipeline.sh \$MODE N       resume at stage N (MODE=open|sharp, default sharp)
  selftest          python lib_registry.py          stdlib-only self-test
  py SCRIPT [ARGS]  run one stage script with the interpreter of its environment
Before the first run: edit $d/config.yaml (the REPLACE entries) and put the template
into $EVOLUTION_PROJECT/02_input/templates/.
EOF
        ;;
    esac
}

# ---------------------------------------------------------------------------
# gvpa
# ---------------------------------------------------------------------------
cmd_gvpa() {
    need_env
    local sub="${1:-help}"; shift || true
    local pkg="${GVPA_PACKAGE:?}"
    [ -d "$pkg" ] || die "$pkg not found."
    cd "$pkg"
    mkdir -p logs
    # shellcheck disable=SC1091
    source ./env.sh
    local script="" interp="$PY"
    case "$sub" in
        make-inputs) script=make_inputs ;;            # 01 inputs for RFdiffusion3
        collect)     script=collect_backbones ;;      # 02 geometry triage
        mpnn)        script=run_mpnn ;;               # 03 sequence design
        manifest)    script=make_af2_manifest ;;      # 04/05 AF2 manifests
        af2)         script=af2_score; interp="$PY_BINDCRAFT" ;;        # 04/05
        rosetta)     script=rosetta_metrics; interp="$PY_BINDCRAFT" ;;  # 06
        boltz)       script=run_boltz ;;              # 07
        rank)        script=merge_rank ;;             # 08
        diversity)   script=diversity ;;              # 08
        order-list)  script=make_order_list ;;        # 08
        report)      script=report ;;                 # 08
        null)        script=null_control ;;           # 09
        boltz-check) script=boltz_score_binder ;;     # standalone Boltz check of one sequence
        rfd3)        gvpa_rfd3 "$@"; return ;;        # 01 backbone generation
        help|*)      cat <<EOF
gvpa commands (run in $pkg after sourcing env.sh):
  01  make-inputs                 expand the campaign config into RFD3 inputs and jobs.tsv
      rfd3 [--gpus N] [--cells REGEX]   run rfd3 once per cell of jobs.tsv
  02  collect                     convert backbones to PDB and triage them
  03  mpnn                        design sequences (SolubleMPNN)
  04  manifest --out ...          AF2 manifest, then: af2 --manifest ... --out ... --mode gate
  05  manifest --gate-scores ...  then: af2 --mode full
  06  rosetta --scores ... --out ...
  07  boltz --accepted ... --out ...
  08  rank | diversity | order-list | report
  09  null make ... ; af2 --mode full ... ; null report --null ...
  extra: boltz-check --binder-seq ... --template ... --pocket-residues ...
Every command passes its arguments on to the script (ATLAS Guide, chapter "Pipeline
Stages", or add --help). It reads configs/gvpa_serratia.yaml, configs/filters_*.json and
inputs/gvpa_ser_5mer.cif in $pkg.
EOF
            return ;;
    esac
    show "$interp" "scripts/$script.py" "$@"
}

# Stage 01: one rfd3 call per cell, as described in the guide (200 timesteps, batch size 8).
gvpa_rfd3() {
    local gpus=1 filter="."
    while [ $# -gt 0 ]; do
        case "$1" in
            --gpus) shift; gpus="$1" ;;
            --cells) shift; filter="$1" ;;
            *) die "unknown option $1" ;;
        esac; shift
    done
    local run
    run="$("$PY" -c '
import sys; sys.path.insert(0, "scripts")
from pathlib import Path
import common
cfg = common.load_config()
print(Path(cfg["run_root"]) / cfg["campaign"])
')" || die "could not read the campaign config"
    [ -f "$run/jobs.tsv" ] || die "$run/jobs.tsv not found; run './atlas.sh gvpa make-inputs' first"
    local n=0 cell spec step gamma nb
    while IFS=$'\t' read -r cell spec step gamma nb; do
        [ "$cell" = "cell" ] && continue
        [[ "$cell" =~ $filter ]] || continue
        local gpu=$((n % gpus)); n=$((n + 1))
        echo "+ [gpu $gpu] rfd3 design $cell" >&2
        CUDA_VISIBLE_DEVICES=$gpu "$FOUNDRY_ENV/bin/rfd3" design \
            "out_dir=$run/backbones/$cell" "inputs=$spec" "n_batches=$nb" \
            diffusion_batch_size=8 "ckpt_path=$RFD3_CKPT" \
            inference_sampler.num_timesteps=200 \
            "inference_sampler.step_scale=$step" "inference_sampler.gamma_0=$gamma" \
            > "logs/rfd3_$cell.log" 2>&1 &
        if (( n % gpus == 0 )); then wait; fi
    done < "$run/jobs.tsv"
    wait
}

# ---------------------------------------------------------------------------
# latch
# ---------------------------------------------------------------------------
cmd_latch() {
    need_env
    local sub="${1:-help}"; shift || true
    case "$sub" in
        check)
            # GPU-free wiring check; needs neither FlexCraft nor weights
            show python3 "$REPO/biosensor-latch/scripts/test_pipeline_wiring.py"; return ;;
    esac
    local fc="${FLEXCRAFT_DIR:?}"
    [ -d "$fc" ] || die "$fc not found. Run ./install.sh latch."
    local py="$FLEXCRAFT_ENV/bin/python"
    local pkg=flexcraft.pipelines.dual_binder_pipeline.scripts
    case "$sub" in
        run)
            cd "$fc"
            show "$py" -u -m "$pkg.run_cage_pipeline" \
                --foundry_dir "$FOUNDRY_WORKSPACE" \
                --rfd3_executable "$FOUNDRY_ENV/bin/rfd3" \
                --foundry_python "$FOUNDRY_ENV/bin/python" \
                --mpnn_checkpoint "$HOME/.foundry/checkpoints/proteinmpnn_v_48_020.pt" \
                "$@" ;;
        stage)
            cd "$fc"
            local s="${1:?script (e.g. score_gfp_fusion_monomer)}"; shift
            show "$py" -u -m "$pkg.$s" "$@" ;;
        help|*) cat <<EOF
latch commands:
  run [ARGS]         the orchestrator (run_cage_pipeline); foundry_dir, rfd3, foundry python and the
                     ProteinMPNN checkpoint are filled in, everything else is passed on
  stage SCRIPT ARGS  one stage script, e.g. score_gfp_fusion_monomer
  check              GPU-free wiring check of the orchestrator
The patched MPNN driver is linked into $FOUNDRY_WORKSPACE/scripts by install.sh. Flags: ./atlas.sh latch run --help and biosensor-latch/README.md.
EOF
            ;;
    esac
}

# ---------------------------------------------------------------------------
# gvsam
# ---------------------------------------------------------------------------
cmd_gvsam() {
    need_env
    local sub="${1:-help}"; shift || true
    local d="${GVSAM_PROJECT:?}/00_pipeline"
    [ -d "$d" ] || die "$d not found. Run ./install.sh gvsam."
    cd "$d"
    case "$sub" in
        run)   show ./run_pipeline.sh "${1:-2}" ;;
        stage) local s="${1:?script}"; shift
               show "$(yaml_value config.yaml python)" "$s" "$@" ;;
        help|*) cat <<EOF
gvsam commands (run in $d):
  run [FROM]         ./run_pipeline.sh [FROM]   split/augment, train, evaluate, gallery
  stage SCRIPT ARGS  one script, e.g. 01_train_cellfinder.py, or predict.py IMAGES --outdir DIR (shipped weights)
Put images in $GVSAM_PROJECT/01_raw and masks in $GVSAM_PROJECT/02_masks_gv first.
EOF
            ;;
    esac
}

# ---------------------------------------------------------------------------
# gas
# ---------------------------------------------------------------------------
cmd_gas() {
    need_env
    case "${1:-help}" in
        notebook) cd "$REPO/gas" && show "$GAS_ENV/bin/jupyter" lab gas-walkthrough.ipynb ;;
        *) cat <<EOF
gas commands:
  notebook           open gas/gas-walkthrough.ipynb in Jupyter
The notebook expects the gvv package, labbook/ and data/ next to it (not part of this
repository yet, README "Known gaps").
EOF
            ;;
    esac
}

# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
ok=0; bad=0
chk() {   # chk <label> <test flag> <path>
    if [ "$2" "$3" ]; then printf '  ok       %s\n' "$1"; ok=$((ok+1))
    else printf '  MISSING  %s  (%s)\n' "$1" "$3"; bad=$((bad+1)); fi
}

cmd_doctor() {
    need_env
    local p="${1:-all}"
    if [ "$p" = all ] || [ "$p" = evolution ]; then
        echo "evolution"
        chk "foundry env"           -x "$TP/foundry/.venv/bin/python"
        chk "boltz env"             -x "$CONDA_BASE/envs/evolution_boltz/bin/python"
        chk "PyRosetta env"         -x "$CONDA_BASE/envs/evolution_rosetta/bin/python"
        chk "BindCraft clone"       -d "$TP/BindCraft"
        chk "AF2 parameters"        -d "$TP/BindCraft/params"
        chk "MPNN weights"          -f "$TP/mpnn_weights/ligandmpnn_v_32_010_25.pt"
        chk "config.yaml"           -f "$EVOLUTION_PROJECT/00_pipeline/config.yaml"
        chk "template.cif"          -f "$EVOLUTION_PROJECT/02_input/templates/SerGvpCR3_4GvpA_trimmed_repacked.cif"
        chk "33_boltz_register.py"  -f "$REPO/gvpc-evolution/33_boltz_register.py"
    fi
    if [ "$p" = all ] || [ "$p" = gvpa ]; then
        echo "gvpa"
        chk "foundry env + rfd3"    -x "$FOUNDRY_ENV/bin/rfd3"
        chk "RFD3 checkpoint"       -f "$RFD3_CKPT"
        chk "SolubleMPNN checkpoint" -f "$MPNN_CKPT"
        chk "BindCraft env"         -x "$BINDCRAFT_ENV/bin/python"
        chk "DAlphaBall.gcc"        -f "$BINDCRAFT_DIR/functions/DAlphaBall.gcc"
        chk "boltz"                 -x "$BOLTZ_BIN"
        chk "campaign config" -f "$GVPA_PACKAGE/configs/gvpa_serratia.yaml"
        chk "filters_strict.json" -f "$GVPA_PACKAGE/configs/filters_strict.json"
        chk "target gvpa_ser_5mer.cif" -f "$GVPA_PACKAGE/inputs/gvpa_ser_5mer.cif"
    fi
    if [ "$p" = all ] || [ "$p" = latch ]; then
        echo "latch"
        chk "flexcraft env"         -x "$FLEXCRAFT_ENV/bin/python"
        chk "flexcraft clone"       -d "$FLEXCRAFT_DIR/flexcraft"
        chk "AF2 parameters"        -d "$FLEXCRAFT_DIR/params/af"
        chk "batch_partial_mpnn.py"  -f "$FOUNDRY_WORKSPACE/scripts/batch_partial_mpnn.py"
    fi
    if [ "$p" = all ] || [ "$p" = gvsam ]; then
        echo "gvsam"
        chk "gvsam env"             -x "$CONDA_BASE/envs/gvsam/bin/python"
        chk "CellSAM clone"         -d "$TP/cellSAM"
        chk "config.yaml"           -f "$GVSAM_PROJECT/00_pipeline/config.yaml"
    fi
    if [ "$p" = all ] || [ "$p" = gas ]; then
        echo "gas"
        chk "gas env"               -x "$GAS_ENV/bin/python"
        chk "gvv package (repository gap)" -d "$REPO/gas/gvv"
    fi
    echo; echo "$ok ok, $bad missing"
    [ "$bad" = 0 ]
}

case "${1:-help}" in
    evolution) shift; cmd_evolution "$@" ;;
    gvpa)   shift; cmd_gvpa "$@" ;;
    latch)  shift; cmd_latch "$@" ;;
    gvsam)  shift; cmd_gvsam "$@" ;;
    gas)    shift; cmd_gas "$@" ;;
    doctor) shift; cmd_doctor "$@" ;;
    help|-h|--help) usage ;;
    *) usage; exit 1 ;;
esac
