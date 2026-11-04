#!/usr/bin/env python
"""Build the AF2 manifest, optionally subsetting to what survived the gate.

    # pass 1: everything MPNN produced
    python scripts/make_af2_manifest.py --out <run>/af2_gate/manifest.csv

    # pass 2: only sequences that cleared the pass-1 thresholds
    python scripts/make_af2_manifest.py --out <run>/af2_full/manifest.csv \
        --gate-scores <run>/af2_gate/scores.csv

Rows are sorted by binder_len then backbone: AF2's free-binder model is bound
to a single length and its complex model to a single backbone, so scoring in
this order minimises how often either has to be rebuilt.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def gate_survivors(path, cfg):
    """ids passing the pass-1 thresholds on ANY scored model."""
    g = cfg["af2"]["gate"]
    keep = set()
    with open(path) as fh:
        for r in csv.DictReader(fh):
            ms = sorted({k.split("_")[0] for k in r if k.startswith("m") and "_" in k
                         and k.split("_")[0][1:].isdigit()})
            for m in ms:
                try:
                    ok = (float(r[f"{m}_i_pae"]) <= g["i_pae_max"]
                          and float(r[f"{m}_i_ptm"]) >= g["i_ptm_min"]
                          and float(r[f"{m}_plddt"]) >= g["plddt_min"])
                except (KeyError, ValueError, TypeError):
                    continue
                if ok:
                    keep.add(r["id"])
                    break
    return keep


def select_diverse(rows, gate_path, max_rows, max_per_bb, rank_by):
    """Cap the gate survivors without collapsing the campaign onto one arm.

    Ranking the whole pool by i_pae and taking the top N would hand almost all
    the budget to whichever arm and length bin happens to score best, which is
    exactly the geometric variety the campaign exists to produce. Instead:

      1. keep at most `max_per_bb` sequences per backbone -- eight MPNN
         sequences on one backbone are far more alike than eight backbones, so
         this is the cheapest diversity per row spent;
      2. round-robin across (arm, length_bin) strata, taking each stratum's
         best remaining row each pass, until the budget is spent.

    Strata that run out simply drop out, so their budget flows to the rest and
    nothing is wasted. Rows not selected stay eligible for a later tranche.
    """
    best = {}
    with open(gate_path) as fh:
        for r in csv.DictReader(fh):
            ms = sorted({k.split("_")[0] for k in r
                         if k.startswith("m") and "_" in k and k.split("_")[0][1:].isdigit()})
            vals = []
            for m in ms:
                try:
                    vals.append(float(r[f"{m}_{rank_by}"]))
                except (KeyError, ValueError, TypeError):
                    pass
            if vals:
                best[r["id"]] = min(vals)          # i_pae: lower is better
    rank = lambda r: best.get(r["id"], float("inf"))

    if max_per_bb:
        bybb = {}
        for r in rows:
            bybb.setdefault(r["design"], []).append(r)
        rows = [r for v in bybb.values() for r in sorted(v, key=rank)[:max_per_bb]]

    if not max_rows or len(rows) <= max_rows:
        return rows, len(rows)

    strata = {}
    for r in rows:
        strata.setdefault((r.get("arm", "?"), r.get("length_bin", "?")), []).append(r)
    for v in strata.values():
        v.sort(key=rank)
    out, keys = [], sorted(strata)
    while len(out) < max_rows and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(out) < max_rows:
                out.append(strata[k].pop(0))
    return out, len(strata)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate-scores", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--sequences", default=None)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="cap the gate survivors at N rows (0 = no cap). "
                         "Selection is stratified, not top-N: see select_diverse.")
    ap.add_argument("--max-seqs-per-backbone", type=int, default=0,
                    help="keep at most N sequences per backbone (0 = all)")
    ap.add_argument("--rank-by", default="i_pae",
                    help="gate metric used to rank within a stratum")
    ap.add_argument("--exclude-ids", default=None,
                    help="glob of CSVs whose `id` column is REMOVED from the pool. "
                         "Used to score a second tranche without redoing the first.")
    a = ap.parse_args()

    cfg = common.load_config(a.config)
    root = Path(cfg["run_root"]) / cfg["campaign"]

    files = [a.sequences] if a.sequences else (
        sorted(glob.glob(str(root / "sequences.shard*.csv"))) or [str(root / "sequences.csv")])
    rows = []
    for f in files:
        with open(f) as fh:
            rows.extend(list(csv.DictReader(fh)))
    if not rows:
        raise SystemExit(f"no sequences found ({files})")

    if a.gate_scores:
        keep = gate_survivors(a.gate_scores, cfg)
        before = len(rows)
        rows = [r for r in rows if r["id"] in keep]
        print(f"gate: {len(rows)}/{before} sequences survived "
              f"({len(rows)/before:.1%})" if before else "gate: no rows")

    if a.exclude_ids:
        drop = set()
        for f in sorted(glob.glob(a.exclude_ids)):
            with open(f) as fh:
                drop |= {r["id"] for r in csv.DictReader(fh)}
        before = len(rows)
        rows = [r for r in rows if r["id"] not in drop]
        print(f"exclude: dropped {before - len(rows)} rows already covered "
              f"({len(drop)} ids from {a.exclude_ids})")

    if a.gate_scores and (a.max_rows or a.max_seqs_per_backbone):
        before = len(rows)
        rows, nstrata = select_diverse(rows, a.gate_scores, a.max_rows,
                                       a.max_seqs_per_backbone, a.rank_by)
        print(f"cap: {len(rows)}/{before} rows kept "
              f"(<={a.max_seqs_per_backbone or 'all'}/backbone, "
              f"round-robin over {nstrata} arm x length strata)")

    rows.sort(key=lambda r: (int(r["binder_len"]), r["design"], r["id"]))

    # Shard by BACKBONE, in contiguous blocks of the length-sorted list -- not
    # by row. af2_score groups by backbone so one template prep + XLA compile
    # is amortised over that backbone's 8 sequences; an interleaved
    # rows[shard::n] split hands EVERY shard EVERY backbone with only 2
    # sequences each, so the ~34 s setup is paid 4x over and amortised 4x
    # worse. Measured on the production gate: 0.058 predictions/s/GPU, a ~30 h
    # projection for 25k sequences.
    #
    # Contiguous blocks also keep each shard inside a narrow length range, so
    # it compiles ~30 shapes instead of all 119.
    designs = list(dict.fromkeys(r["design"] for r in rows))
    n = len(designs)
    lo = a.shard * n // a.n_shards
    hi = (a.shard + 1) * n // a.n_shards
    keep = set(designs[lo:hi])
    mine = [r for r in rows if r["design"] in keep]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "design", "cell", "arm", "length_bin", "binder_len", "seq", "pdb"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(mine)
    nd = len({r["design"] for r in mine})
    nl = len({r["binder_len"] for r in mine})
    print(f"shard {a.shard}/{a.n_shards}: {len(mine)} rows, {nd} backbones "
          f"({len(mine)/max(nd,1):.1f} seqs each), {nl} distinct lengths -> {out}")


if __name__ == "__main__":
    main()
