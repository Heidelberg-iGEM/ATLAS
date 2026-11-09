#!/usr/bin/env bash
#
# One entry point for every part of the Drylab pipelines.
#
#   atlas help
#   atlas doctor [PART]
#   atlas evolution [--config F] [--sharp] [--from N]   one round (more: atlas evolution help)
#   atlas gvpa      [--config F] [--filters F] [--gpus N] [--shards N] [--from 1-9]   whole campaign
#   atlas latch     --rfd3_yaml F --gfp_binder_pdb F ...   (= run; more: atlas latch help)
#   atlas gvsam     IMAGES --outdir DIR                    (= predict; train: atlas gvsam run)
#   atlas gas                                              opens the notebook
#   atlas pas                                              opens the PAS measurement GUI
#
# "atlas" is the symlink install.sh puts into ~/.local/bin (./atlas.sh works too).
#
# It reads atlas.env (written by install.sh) to find every interpreter, changes
# into the right folder and calls the scripts of this repository. It adds no logic
# of its own: every command prints the line it runs.

set -euo pipefail

SELF="${BASH_SOURCE[0]}"          # follow the ~/.local/bin/atlas symlink back to the repository
while [ -L "$SELF" ]; do
    t="$(readlink "$SELF")"
    case "$t" in /*) SELF="$t" ;; *) SELF="$(dirname "$SELF")/$t" ;; esac
done
REPO="$(cd "$(dirname "$SELF")" && pwd)"
CALLER_PWD="$PWD"
abspath() { case "$1" in /*) echo "$1" ;; *) echo "$CALLER_PWD/$1" ;; esac; }
ENVFILE="$REPO/atlas.env"

usage() { sed -n 2,14p "$SELF" | sed 's/^# \{0,1\}//'; }

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
        --*)       # atlas evolution [--config F] [--sharp] [--from N]
            set -- "$sub" "$@"
            local mode=open from=1
            while [ $# -gt 0 ]; do
                case "$1" in
                    --config) shift; export CONFIG="$(abspath "${1:?}")" ;;
                    --sharp)  mode=sharp ;;
                    --from)   shift; from="${1:?}" ;;
                    *) die "unknown option $1 (atlas evolution help)" ;;
                esac; shift
            done
            [ -f "${CONFIG:-config.yaml}" ] || die "${CONFIG:-config.yaml} missing"
            show ./run_pipeline.sh "$mode" "$from" ;;
        calibrate) [ -f config.yaml ] || die "config.yaml missing"; show ./run_pipeline.sh ;;
        sharp)     show ./run_pipeline.sh sharp "${1:-1}" ;;
        resume)    show ./run_pipeline.sh "${MODE:-sharp}" "${1:?stage number}" ;;
        selftest)  show python3 lib_registry.py ;;
        check)     show python3 lib_checks.py "$(abspath "${1:?template .cif}")" ;;
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
evolution: one round in one go
  atlas evolution [--config F] [--sharp] [--from N]
    --config   config.yaml of another target/round (default $d/config.yaml)
    --sharp    apply the gate and write order.csv (default: calibration run, nothing selected)
    --from     resume at stage N
Single commands (run in $d):
  calibrate         ./run_pipeline.sh               calibration run, nothing selected
  sharp [FROM]      ./run_pipeline.sh sharp [FROM]  gate applied, order.csv written
  resume N          ./run_pipeline.sh \$MODE N       resume at stage N (MODE=open|sharp, default sharp)
  check CIF         python lib_checks.py CIF        check a template without a GPU
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
        run|--*)     [ "$sub" = run ] || set -- "$sub" "$@"; gvpa_run "$@"; return ;;   # whole campaign
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
gvpa: the whole campaign in one go, each stage prints the command it runs
  atlas gvpa [--config campaign.yaml] [--filters filters.json] [--gpus N] [--shards N] [--from 1-9]
    --config   campaign file (target, hotspots, ...; default configs/gvpa_serratia.yaml)
    --filters  acceptance filters (default configs/filters_strict.json)
    --gpus     GPUs for RFdiffusion3     --shards  parallel CPU shards for Rosetta (stage 06)
    --from     resume at a stage; the order is 1 2 3 4 5 6 8 7 9 (see the table below)
Single stages (run in $pkg after sourcing env.sh):
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
    run="$(gvpa_run_dir)" || die "could not read the campaign config"
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

# The whole campaign, stage by stage in the order of the README table (8 before 7:
# the Boltz run needs accepted.csv, then the ranking is repeated with its scores).
gvpa_run() {
    local gpus=1 shards=1 from=1 filters="configs/filters_strict.json"
    while [ $# -gt 0 ]; do
        case "$1" in
            --config)  shift; export CAMPAIGN_CONFIG="$(abspath "${1:?}")" ;;
            --filters) shift; filters="$(abspath "${1:?}")" ;;
            --gpus)    shift; gpus="${1:?}" ;;
            --shards)  shift; shards="${1:?}" ;;
            --from)    shift; from="${1:?}" ;;
            *) die "unknown option $1 (atlas gvpa help)" ;;
        esac; shift
    done
    case "$from" in [1-9]) ;; *) die "--from must be a stage number 1-9" ;; esac
    local run; run="$(gvpa_run_dir)" || die "could not read the campaign config"
    local g="$run/af2_gate" f="$run/af2_full" n="$run/null" started=0 s m i
    mkdir -p "$g" "$f" "$n" "$run/boltz" "$run/rosetta"
    for s in 1 2 3 4 5 6 8 7 9; do
        [ "$s" = "$from" ] && started=1
        [ "$started" = 1 ] || continue
        echo "=== stage 0$s" >&2
        case "$s" in
            1) show "$PY" scripts/make_inputs.py; gvpa_rfd3 --gpus "$gpus" ;;
            2) show "$PY" scripts/collect_backbones.py ;;
            3) show "$PY" scripts/run_mpnn.py ;;
            4) show "$PY" scripts/make_af2_manifest.py --out "$g/manifest.csv"
               show "$PY_BINDCRAFT" scripts/af2_score.py --manifest "$g/manifest.csv" --out "$g/scores.csv" --mode gate ;;
            5) show "$PY" scripts/make_af2_manifest.py --out "$f/manifest.csv" --gate-scores "$g/scores.csv"
               show "$PY_BINDCRAFT" scripts/af2_score.py --manifest "$f/manifest.csv" --out "$f/scores.csv" \
                   --mode full --save-pdb-dir "$f/pdb" ;;
            6) for m in 0 1; do
                   local pids="" outs=""
                   for ((i = 0; i < shards; i++)); do
                       local o="$run/rosetta.m$m.csv"
                       [ "$shards" = 1 ] || o="$run/rosetta/m$m.shard$i.csv"
                       outs="$outs $o"
                       show "$PY_BINDCRAFT" scripts/rosetta_metrics.py --scores "$f/scores.csv" --out "$o" \
                           --model "$m" --n-shards "$shards" --shard "$i" --resume &
                       pids="$pids $!"
                   done
                   for i in $pids; do wait "$i"; done
                   [ "$shards" = 1 ] || "$PY" -c '
import csv, sys
rows = [r for f in sys.argv[2:] for r in csv.DictReader(open(f))]
w = csv.DictWriter(open(sys.argv[1], "w", newline=""), list(dict.fromkeys(k for r in rows for k in r)))
w.writeheader(); w.writerows(rows)' "$run/rosetta.m$m.csv" $outs
               done ;;
            8) gvpa_rank "$filters" ;;
            7) show "$PY" scripts/run_boltz.py --accepted "$run/accepted.csv" --out "$run/boltz/scores.csv"
               gvpa_rank "$filters" --boltz "$run/boltz/scores.csv" ;;
            9) show "$PY" scripts/null_control.py make
               show "$PY_BINDCRAFT" scripts/af2_score.py --manifest "$n/manifest.csv" --out "$n/scores.csv" --mode full
               show "$PY" scripts/null_control.py report --null "$n/scores.csv" ;;
        esac
    done
}

gvpa_rank() {   # gvpa_rank <filters> [extra merge_rank args]
    local filters="$1"; shift
    show "$PY" scripts/merge_rank.py --filters "$filters" "$@"
    show "$PY" scripts/diversity.py
    show "$PY" scripts/make_order_list.py
    show "$PY" scripts/report.py
}

gvpa_run_dir() {
    "$PY" -c '
import sys; sys.path.insert(0, "scripts")
from pathlib import Path
import common
cfg = common.load_config()
print(Path(cfg["run_root"]) / cfg["campaign"])
'
}

# ---------------------------------------------------------------------------
# latch
# ---------------------------------------------------------------------------
cmd_latch() {
    need_env
    local sub="${1:-help}"; shift || true
    case "$sub" in
        --*) set -- "$sub" "$@"; sub=run ;;     # atlas latch --rfd3_yaml ... = atlas latch run ...
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
latch: atlas latch --rfd3_yaml latch.yaml --gfp_binder_pdb ... (same as run)
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
    [ ! -d "$(abspath "$sub")" ] || { set -- "$sub" "$@"; sub=predict; }   # atlas gvsam IMAGES --outdir DIR
    local d="${GVSAM_PROJECT:?}/00_pipeline"
    [ -d "$d" ] || die "$d not found. Run ./install.sh gvsam."
    cd "$d"
    case "$sub" in
        run)   show ./run_pipeline.sh "${1:-2}" ;;
        predict) # absolute paths: the script runs from the project folder
               local imgs; imgs="$(abspath "${1:?images folder}")"; shift
               local a=() x; for x in "$@"; do case "$x" in -*|[0-9]*) a+=("$x") ;; *) a+=("$(abspath "$x")") ;; esac; done
               show "$(yaml_value config.yaml python)" predict.py "$imgs" ${a[@]+"${a[@]}"} ;;
        stage) local s="${1:?script}"; shift
               show "$(yaml_value config.yaml python)" "$s" "$@" ;;
        help|*) cat <<EOF
gvsam: atlas gvsam IMAGES --outdir DIR   (= predict)
commands (run in $d):
  predict IMAGES --outdir DIR   segment images with the shipped weights (relative paths ok)
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
    case "${1:-notebook}" in
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
# pas
# ---------------------------------------------------------------------------
cmd_pas() {
    need_env
    case "${1:-gui}" in
        gui) cd "$REPO/pas-gui" && show "$PAS_ENV/bin/python" pas_gui.py ;;
        *) cat <<EOF
pas commands:
  gui                run pas-gui/pas_gui.py (default): ramp, log and evaluate a PAS measurement
Needs the Arduino Nano (flash pas-gui/sweep_up_nano first) and the photometer on USB.
Setup, hardware and a run step by step: pas-gui/README.md
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
    if [ "$p" = all ] || [ "$p" = pas ]; then
        echo "pas"
        chk "pas env"               -x "$PAS_ENV/bin/python"
        if [ -x "$PAS_ENV/bin/python" ] && "$PAS_ENV/bin/python" -c "import tkinter, numpy, scipy, matplotlib" 2>/dev/null; then
            printf '  ok       %s\n' "tkinter, numpy, scipy, matplotlib import"; ok=$((ok+1))
        else printf '  MISSING  %s\n' "tkinter, numpy, scipy, matplotlib import"; bad=$((bad+1)); fi
        chk "pas_gui.py"            -f "$REPO/pas-gui/pas_gui.py"
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
    pas)    shift; cmd_pas "$@" ;;
    doctor) shift; cmd_doctor "$@" ;;
    help|-h|--help) usage ;;
    *) usage; exit 1 ;;
esac
