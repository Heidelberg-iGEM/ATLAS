#!/usr/bin/env bash
#
# GvpC evolution, one round of in silico evolution of a binding interface.
#
#   ./run_pipeline.sh              calibration run, selects nothing
#   ./run_pipeline.sh sharp        gate applies, order.csv is written
#   ./run_pipeline.sh sharp 8      resume from stage 8
#
# Everything is set in config.yaml. This script reads the values and passes
# them to the stages; it has no parameters of its own.
#
#    1 LigandMPNN          5 Boltz                 9 Gate and selection
#    2 Helix palette       6 Boltz ranking        10 PyRosetta
#    3 Controls            7 Boltz register       11 Second join
#    4 Composition filter  8 AlphaFold2
#
# The stages run in four conda environments because their dependencies are
# mutually exclusive. Each stage prints its command before running, so any of
# them can be repeated by hand.
#
# AlphaFold2 runs before Boltz: it separates better and costs less per sequence,
# so Boltz only sees the survivors. PyRosetta runs last because it needs the
# Boltz models and a row list from the gate.

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-open}"       # open | sharp
FROM="${2:-1}"
CONFIG="${CONFIG:-config.yaml}"
PY="${PY:-python3}"

[ -f "$CONFIG" ] || { echo "config.yaml not found: $CONFIG"; exit 1; }

# lib_registry needs no external packages, so any python3 can read the config
# before an environment is active.
cfg() {
    "$PY" - "$CONFIG" "$1" "${2-}" <<'PYEOF'
import sys
import lib_registry as reg
value = reg.load_config(sys.argv[1])
try:
    for key in sys.argv[2].split("."):
        value = value[key]
except (KeyError, TypeError):
    value = sys.argv[3]
if value is None:
    value = ""
if isinstance(value, bool):
    value = "true" if value else "false"
print(",".join(str(x) for x in value) if isinstance(value, list) else value)
PYEOF
}

head() { printf '\n=== %2s/11  %s\n' "$1" "$2"; }


# --- values from the config --------------------------------------------------
ROOT=$(cfg project.root)
LOOP=$(cfg project.loop)

MPNN_PY=$(cfg envs.mpnn_py);   AF2_PY=$(cfg envs.af2_py)
BOLTZ_PY=$(cfg envs.boltz_py); ROS_PY=$(cfg envs.ros_py)

BINDER=$(cfg template.binder_chain)
TARGETS=$(cfg template.target_chains)
SITE=$(cfg epitope.site | tr ',' '-')
ANCHORS=$(cfg zones.anchors none)

TIERS=$(cfg mpnn.tiers);       N_PER_CELL=$(cfg mpnn.n_per_cell)
TEMP=$(cfg mpnn.temperature);  SEED0=$(cfg mpnn.seed0)
MPNN_MODEL=$(cfg mpnn.model ligand_mpnn)

LEVELS=$(cfg helicity.levels)

N_SCRAMBLES=$(cfg controls.n_scrambles)
CTRL_SEED=$(cfg controls.seed)
NORMAL_HELIX=$(cfg controls.normal_helix true)
N_ORDER_SCRAMBLES=$(cfg controls.n_order_scrambles 0)
CTRL_EXTRA=$(cfg controls.extra)

ROS_ROWS=$(cfg rosetta.rows order)
ROS_JOBS=$(cfg rosetta.jobs 8)

LOOPDIR="$ROOT/04_loops/$LOOP"
BOLTZ_OUT="$ROOT/03_output/boltz_output/${LOOP}_constrained"

mkdir -p "$LOOPDIR" "$ROOT/02_input/boltz_input" \
         "$ROOT/03_output"/{MPNN_output,af2_output,boltz_output,archive}

# Report missing environments now, not two GPU hours in.
for p in "$MPNN_PY" "$AF2_PY" "$BOLTZ_PY"; do
    [ -x "${p/#\~/$HOME}" ] || { echo "Interpreter not executable: $p
Check the envs: block in config.yaml."; exit 1; }
done

echo "GvpC evolution  round $LOOP  ($MODE, from stage $FROM)"
echo "  $ROOT"
echo "  binder $BINDER against $TARGETS, epitope $SITE"
echo "  $TIERS x $N_PER_CELL sequences at T=$TEMP, palettes $LEVELS"


# --- 1  LigandMPNN -----------------------------------------------------------
# Redesigns only the positions listed in zones.tiers. Everything else stays
# native, so the solvent-facing side cannot be rebuilt along with the interface.
# Writes registry.csv: from here on every sequence has an ID.
# The stage checks the template before loading anything.
if [ "$FROM" -le 1 ]; then
    head 1 "LigandMPNN"
    "$MPNN_PY" 10_mpnn_design.py --config "$CONFIG" --loop "$LOOP" \
        --tiers "$TIERS" --n-per-cell "$N_PER_CELL" \
        --temperature "$TEMP" --seed0 "$SEED0" --model-type "$MPNN_MODEL"
fi


# --- 2  Helix palette --------------------------------------------------------
# Replaces the solvent-exposed positions from a lookup table, keeping the charge
# class of the native residue. A second MPNN run would reach for alanine, which
# is what we want less of on the outside.
# --with-t0 appends the wild type as a row, so it runs in the same prediction
# job as the designs.
if [ "$FROM" -le 2 ]; then
    head 2 "Helix palette"
    "$PY" 11_mpnn_helicity.py --config "$CONFIG" --loop "$LOOP" \
        --levels "$LEVELS" --with-t0
fi


# --- 3  Controls -------------------------------------------------------------
# Scrambles keep the composition of the wild type and destroy the order, so they
# test whether a score reads binding or amino acid statistics. The EAAAK helix
# breaks the composition filter on purpose.
if [ "$FROM" -le 3 ]; then
    head 3 "Controls"
    ARGS=(--config "$CONFIG" --loop "$LOOP"
          --n-scrambles "$N_SCRAMBLES" --seed "$CTRL_SEED"
          --n-order-scrambles "$N_ORDER_SCRAMBLES")
    [ "$NORMAL_HELIX" = "true" ] || ARGS+=(--no-normal-helix)
    [ -z "$CTRL_EXTRA" ] || ARGS+=(--extra "$ROOT/$CTRL_EXTRA")
    "$PY" 12_controls.py "${ARGS[@]}"
fi


# --- 4  Composition filter ---------------------------------------------------
# Cheap, and before the GPU. Also deduplicates against earlier rounds.
# Worth a look afterwards, while it can still be fixed:
#   python cells_after_filter.py --config config.yaml --loop $LOOP
if [ "$FROM" -le 4 ]; then
    head 4 "Composition filter and deduplication"
    "$PY" 15_filter_composition.py --config "$CONFIG" --loop "$LOOP"
fi


# --- 5  Boltz input ----------------------------------------------------------
# One YAML per sequence in a freshly created directory, because boltz predict
# takes a directory and runs everything it finds there.
if [ "$FROM" -le 5 ]; then
    head 5 "Boltz input"
    "$BOLTZ_PY" 30_boltz_yaml.py --config "$CONFIG" --loop "$LOOP" --mode constrained
fi


# --- 6  Boltz ----------------------------------------------------------------
# Five diffusion samples. bz_shape_rmsd is a gate criterion and only exists as
# the spread across seeds.
if [ "$FROM" -le 6 ]; then
    head 6 "Boltz"
    "$BOLTZ_PY" 31_boltz_predict.py --config "$CONFIG" --loop "$LOOP" --mode constrained
fi

PRED=$(ls -d "$BOLTZ_OUT"/*/predictions 2>/dev/null | head -n1 || true)
if [ "$FROM" -le 10 ] && [ -z "$PRED" ]; then
    echo "No Boltz predictions under $BOLTZ_OUT"; exit 1
fi


# --- 7  Boltz ranking --------------------------------------------------------
# bz_shape_rmsd (pose reproducible), bz_site_frac (right surface) and bz_bsa,
# the buried area, which is the ranking axis of the selection.
# The PASS/FAIL column this prints is orientation only; the decision is made in
# stage 9 from the CSV. --prefix is required, stages 7 and 8 both emit a column
# called site_frac with different counting.
if [ "$FROM" -le 7 ]; then
    head 7 "Boltz ranking"
    "$BOLTZ_PY" 32_boltz_rank.py "$PRED" \
        --binder "$BINDER" --targets "$TARGETS" --site "$SITE" \
        --prefix bz_ --id-col seq_id --csv "$LOOPDIR/scores_boltz.csv"
fi


# --- 8  Boltz register -------------------------------------------------------
# The column this exists for is reg_chains_hit_seeds: in how many seeds all
# target chains are contacted. That is a gate criterion (gate.min_chain_seeds).
# --anchors only affects a diagnostic column.
if [ "$FROM" -le 8 ]; then
    head 8 "Boltz register"
    "$BOLTZ_PY" 33_boltz_register.py "$PRED" \
        --binder "$BINDER" --targets "$TARGETS" --site "$SITE" \
        --anchors "$ANCHORS" --per-seed \
        --prefix reg_ --id-col seq_id --csv "$LOOPDIR/scores_register.csv"
fi


# --- 9  AlphaFold2 -----------------------------------------------------------
# Initial guess: the binder backbone is given in the designed pose, sequence and
# side chains erased. Produces af_i_pae_m1.
# 21_af2_score.py is launched by this stage and only has to sit alongside.
if [ "$FROM" -le 9 ]; then
    head 9 "AlphaFold2 initial guess"
    "$AF2_PY" 20_af2_run.py --config "$CONFIG" --loop "$LOOP"
fi


# --- 10  Gate and selection --------------------------------------------------
# Joins registry.csv and all scores_*.csv into master.csv. The gate is a column,
# not a stage: nothing is deleted, every row gets gate_pass and gate_fail.
# With --open nothing is excluded and nothing is selected; that run is for
# reading the distributions and setting the thresholds.
if [ "$MODE" != "sharp" ]; then
    head 10 "Gate, calibration run"
    "$PY" 40_select_gate.py --config "$CONFIG" --loop "$LOOP" --open
    cat <<EOF

Calibration run finished. Nothing was selected and nothing ordered.

Next:
  1. Read the distributions above.
  2. Check the controls:
       python check_controls.py --table $LOOPDIR/master.csv
     A threshold is only useful if the wild type passes it and the controls fail
     it. If controls score better than the wild type, no threshold on that
     metric helps.
  3. If you have lab measurements, put them in 04_loops/experimental.csv with a
     seq_sha8 or binder_seq column. They are attached automatically next run.
  4. Enter the values under gate: in config.yaml, then
       ./run_pipeline.sh sharp

Table: $LOOPDIR/master.csv
EOF
    exit 0
fi

head 10 "Gate, live"
"$PY" 40_select_gate.py --config "$CONFIG" --loop "$LOOP"


# --- 11  PyRosetta -----------------------------------------------------------
# FastRelax on the Boltz models, then interface energetics. Nothing downstream
# depends on this; no ros_ column is a gate criterion.
# --rows order scores the order list because it contains the controls, and a
# single interface energy means nothing without the series.
if [ "$FROM" -le 11 ]; then
    head 11 "PyRosetta"
    "$ROS_PY" 60_rosetta_score.py --config "$CONFIG" --loop "$LOOP" \
        --models-dir "$PRED" --rows "$ROS_ROWS" --jobs "$ROS_JOBS"
fi


# --- second join -------------------------------------------------------------
# Seconds. Pulls scores_rosetta.csv into master.csv, gate.csv and order.csv.
# The selection does not change, only columns are added.
"$PY" 40_select_gate.py --config "$CONFIG" --loop "$LOOP"

cat <<EOF

Round $LOOP finished.

  $LOOPDIR/master.csv    all sequences, all columns
  $LOOPDIR/gate.csv      plus gate_pass and gate_fail
  $LOOPDIR/selected.csv  the selected designs
  $LOOPDIR/order.csv     what goes to the lab: designs, controls and the
                         deliberately included failures

After the wet lab: add the measurements to 04_loops/experimental.csv, re-check
the thresholds with check_controls.py, and run the next round.
EOF
