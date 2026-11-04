#!/usr/bin/env bash
#
# GVSAM, one training run.
#
#   ./run_pipeline.sh              all stages
#   ./run_pipeline.sh 3            resume from stage 3
#
# NOTE: reconstructed from the ATLAS Guide and the stage scripts. The original
# driver was not available when this repository was assembled; the stage
# arguments below are taken from the scripts' own argparse definitions.
#
# Everything is set in config.yaml.
#
#    2 Split and augment       5 Train decoder (only if decoder.enabled)
#    3 Train CellFinder        6 Evaluate decoder (only if decoder.enabled)
#    4 Evaluate CellFinder     7 Gallery (visualize.ipynb)
#
# Stage 1 (boxes from masks) is part of augment.py and has no separate command.

set -euo pipefail
cd "$(dirname "$0")"

FROM="${1:-2}"
CONFIG="${CONFIG:-config.yaml}"
[ -f "$CONFIG" ] || { echo "config.yaml not found: $CONFIG"; exit 1; }

# The interpreter comes from config.yaml (env.python). Reading that one value
# needs no YAML parser.
PY=$(sed -n 's/^[[:space:]]*python:[[:space:]]*//p' "$CONFIG" | head -n1)
PY="${PY/#\~/$HOME}"
PY="${GVSAM_PY:-$PY}"
[ -x "$PY" ] || { echo "Interpreter not executable: $PY (env.python in $CONFIG)"; exit 1; }

cfg() {
    "$PY" - "$CONFIG" "$1" "${2-}" <<'PYEOF'
import os, sys, yaml
value = yaml.safe_load(open(sys.argv[1]))
try:
    for key in sys.argv[2].split("."):
        value = value[key]
except (KeyError, TypeError):
    value = sys.argv[3]
if value is None:
    value = ""
if isinstance(value, bool):
    value = "true" if value else "false"
if isinstance(value, list):
    value = " ".join(str(x) for x in value)
if isinstance(value, str):
    value = os.path.expanduser(value)
print(value)
PYEOF
}

head() { printf '\n=== %s  %s\n' "$1" "$2"; }

ROOT=$(cfg project.root)
RUN=$(cfg project.run)
IMG=$ROOT/$(cfg data.image_dir)
MASK=$ROOT/$(cfg data.mask_dir)
SPLIT=$ROOT/03_splits
OUT=$ROOT/04_runs/$RUN
CKPT=$(cfg models.cellsam_ckpt)
mkdir -p "$SPLIT" "$OUT"

# --- 2  Split and augment ----------------------------------------------------
if [ "$FROM" -le 2 ]; then
    head 2 "Split and augment"
    ARGS=(--image_dir "$IMG" --mask_dir "$MASK" --out_dir "$SPLIT"
          --image_ext "$(cfg data.image_ext .tif)" --mask_suffix "$(cfg data.mask_suffix _mask)"
          --val_frac "$(cfg split.val_frac 0.2)" --test_frac "$(cfg split.test_frac 0.1)"
          --seed "$(cfg split.seed 42)")
    ANGLES=$(cfg augment.angles "90 270"); FLIPS=$(cfg augment.flips "")
    # shellcheck disable=SC2086
    [ -z "$ANGLES" ] || ARGS+=(--angles $ANGLES)
    # shellcheck disable=SC2086
    [ -z "$FLIPS" ] || ARGS+=(--flips $FLIPS)
    NEG=$(cfg data.negatives "")
    [ -z "$NEG" ] || { [ -f "$ROOT/$NEG" ] && ARGS+=(--negatives "$ROOT/$NEG"); } || true
    "$PY" augment.py "${ARGS[@]}"
fi

# --- 3  Train CellFinder -----------------------------------------------------
if [ "$FROM" -le 3 ]; then
    head 3 "Train CellFinder"
    "$PY" 01_train_cellfinder.py --split_dir "$SPLIT" --outdir "$OUT" \
        --init_cellsam_ckpt "$CKPT" \
        --epochs "$(cfg cellfinder.epochs 50)" --batch_size "$(cfg cellfinder.batch_size 3)" \
        --lr "$(cfg cellfinder.lr 1e-4)" --lr_backbone "$(cfg cellfinder.lr_backbone 1e-5)" \
        --weight_decay "$(cfg cellfinder.weight_decay 1e-4)" \
        --freeze_backbone_epochs "$(cfg cellfinder.freeze_backbone_epochs 0)" \
        --sampler_weight "$(cfg augment.sampler_weight 0.3)" \
        --num_workers "$(cfg cellfinder.num_workers 8)"
fi

# --- 4  Evaluate CellFinder against base CellSAM -----------------------------
if [ "$FROM" -le 4 ]; then
    head 4 "Evaluate CellFinder"
    "$PY" 02_eval_cellfinder.py --bboxes_test "$SPLIT/test_bboxes.tsv" \
        --new_ckpt "$OUT/cellfinder.pt" --base_ckpt "$CKPT" --outdir "$OUT" \
        --score_thr "$(cfg eval.score_thr 0.2)" --match "$(cfg eval.match center)" \
        --center_tol "$(cfg eval.center_tol 2)" --area_ratio "$(cfg eval.area_ratio 4.0)" \
        --iou_thr "$(cfg eval.iou_thr 0.3)" --border_margin "$(cfg eval.border_margin 1)"
fi

# --- 5, 6  Mask decoder (optional) -------------------------------------------
if [ "$(cfg decoder.enabled false)" = "true" ]; then
    if [ "$FROM" -le 5 ]; then
        head 5 "Train decoder"
        "$PY" 03_train_decoder.py --split_dir "$SPLIT" --outdir "$OUT" \
            --init_cellsam_ckpt "$CKPT" --cellfinder_ckpt "$OUT/cellfinder.pt" \
            --epochs "$(cfg decoder.epochs 50)" --batch_size "$(cfg decoder.batch_size 2)" \
            --lr_decoder "$(cfg decoder.lr_decoder 1e-5)" --lr_neck "$(cfg decoder.lr_neck 5e-6)" \
            --weight_decay "$(cfg decoder.weight_decay 1e-4)" \
            --max_instances_per_image "$(cfg decoder.max_instances_per_image 200)" \
            --sampler_weight "$(cfg augment.sampler_weight 0.3)" \
            --num_workers "$(cfg cellfinder.num_workers 8)"
    fi
    if [ "$FROM" -le 6 ]; then
        head 6 "Evaluate decoder"
        "$PY" 04_eval_decoder.py --split_dir "$SPLIT" --outdir "$OUT" \
            --base_ckpt "$CKPT" --new_ckpt "$OUT/decoder.pt" \
            --bbox_threshold "$(cfg decoder.bbox_threshold 0.4)" \
            --inst_iou_thr "$(cfg decoder.inst_iou_thr 0.5)" \
            --border_margin "$(cfg eval.border_margin 1)"
    fi
fi

# --- 7  Gallery --------------------------------------------------------------
if [ "$FROM" -le 7 ]; then
    head 7 "Gallery"
    NB=$(cfg visualize.notebook visualize.ipynb)
    GVSAM_CONFIG="$PWD/$CONFIG" GVSAM_RUN_DIR="$OUT" \
        "$PY" -m jupyter nbconvert --to notebook --execute "$NB" \
        --output "$OUT/visualize_executed.ipynb"
fi

echo
echo "Run $RUN finished: $OUT"
