#!/usr/bin/env python
"""Composition-matched scramble null. The campaign is not reportable without it.

A metric that scores a shuffled sequence as highly as a real design is not
measuring design quality, and this group has already been burned by exactly
that: on an earlier target, Boltz ipTM, ligand-ipTM and the affinity head all
gave z ~ 0 against scrambles, while AF2 initial-guess separated real from
scramble at AUC 0.99. Which metrics discriminate is a property of the TARGET,
not a settled fact, so it is re-measured here.

Each scramble keeps the design's exact amino-acid composition and its length,
and is threaded onto the SAME backbone through the SAME AF2 path. The only
thing that differs is residue order, so any separation is attributable to
sequence design rather than to composition, size or backbone.

Two steps, with an AF2 run in between:

    # 1. build the scrambled manifest
    python scripts/null_control.py make --out <run>/null/manifest.csv

    # 2. score it -- same scorer, same settings as the real designs
    python scripts/af2_score.py --manifest <run>/null/manifest.csv \
        --out <run>/null/scores.csv --mode full --save-pdb-dir <run>/null/pdb

    # 3. compare
    python scripts/null_control.py report --null <run>/null/scores.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

METRICS = ["plddt", "i_ptm", "i_pae", "ptm", "scrmsd_complex", "hotspot_rmsd",
           "binder_alone_plddt", "binder_rmsd"]
LOWER_IS_BETTER = {"i_pae", "scrmsd_complex", "hotspot_rmsd", "binder_rmsd", "pae"}


def scramble(seq, rng):
    """Shuffle interior residues, keeping composition and both termini fixed."""
    body = list(seq[1:-1])
    for _ in range(64):
        rng.shuffle(body)
        cand = seq[0] + "".join(body) + seq[-1]
        if cand != seq:
            return cand
    return seq[0] + "".join(body) + seq[-1]


def auc(pos, neg):
    """Mann-Whitney AUC: P(a real design scores above a scramble)."""
    pos = np.asarray([v for v in pos if v == v], float)
    neg = np.asarray([v for v in neg if v == v], float)
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)) + 1.0
    # average ranks for ties
    order = np.argsort(allv)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
        i = j + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cmd_make(cfg, root, a):
    src = Path(a.source) if a.source else (
        root / ("accepted.csv" if a.only_accepted else "merged.csv"))
    if not src.exists():
        src = root / "merged.csv"
    with open(src) as fh:
        rows = list(csv.DictReader(fh))
    if a.only_accepted:
        rows = [r for r in rows if r.get("accepted") == "1"]
    if not rows:
        raise SystemExit(f"no designs in {src}")

    rng = random.Random(cfg["null_control"]["seed"])
    n_per = cfg["null_control"]["n_per_accepted"]
    cap = int(cfg["null_control"].get("max_designs") or 0)
    if cap and len(rows) > cap:
        # Even subsample across the score range, not the top N: a null drawn
        # only from the best designs overstates separation.
        step = len(rows) / cap
        rows = [rows[int(i * step)] for i in range(cap)]
        print(f"subsampled to {len(rows)} designs (null_control.max_designs)")
    out_rows = []
    for r in rows:
        for k in range(n_per):
            out_rows.append({
                "id": f"{r['id']}__null{k}", "design": r["design"],
                "cell": r.get("cell", ""), "arm": r.get("arm", ""),
                "length_bin": r.get("length_bin", ""),
                "binder_len": r["binder_len"], "seq": scramble(r["seq"], rng),
                "pdb": r["pdb"], "real_id": r["id"],
            })
    out = Path(a.out or root / "null" / "manifest.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    print(f"{len(out_rows)} scrambles from {len(rows)} designs (source: {src.name})")
    print(f"wrote {out}")
    print("\nnow score it with af2_score.py --mode full, then run `null_control.py report`")


def cmd_report(cfg, root, a):
    real = {}
    for f in sorted(glob.glob(str(root / "af2_full" / "scores*.csv"))):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                real[r["id"]] = r
    with open(a.null) as fh:
        null = list(csv.DictReader(fh))
    if not real or not null:
        raise SystemExit("need both real af2_full scores and null scores")

    models = cfg["af2"]["full_models"]

    def series(rows, key):
        out = []
        for r in rows:
            vs = [float(r[f"m{m}_{key}"]) for m in models
                  if r.get(f"m{m}_{key}") not in (None, "")
                  and r[f"m{m}_{key}"] == r[f"m{m}_{key}"]]
            vs = [v for v in vs if v == v]
            if vs:
                out.append(float(np.mean(vs)))
        return out

    real_rows = list(real.values())
    print(f"real designs : {len(real_rows)}")
    print(f"scrambles    : {len(null)}\n")
    print(f"{'metric':<22}{'real mean':>11}{'null mean':>11}{'z':>8}{'AUC':>8}  verdict")
    print("-" * 76)
    for k in METRICS:
        rv, nv = series(real_rows, k), series(null, k)
        if len(rv) < 3 or len(nv) < 3:
            print(f"{k:<22}{'-':>11}{'-':>11}{'-':>8}{'-':>8}  insufficient data")
            continue
        rm, nm, sd = np.mean(rv), np.mean(nv), np.std(nv)
        z = (rm - nm) / sd if sd > 0 else float("nan")
        A = auc(rv, nv)
        if k in LOWER_IS_BETTER:
            z = -z
            A = 1 - A
        verdict = ("DISCRIMINATES" if A >= 0.80 else
                   "weak" if A >= 0.65 else "NON-DISCRIMINATIVE")
        print(f"{k:<22}{rm:>11.4f}{nm:>11.4f}{z:>8.2f}{A:>8.3f}  {verdict}")
    print("\nAUC is P(real design beats a scramble), sign-corrected so higher is")
    print("always better. A metric near 0.5 is measuring composition, not design;")
    print("do not gate or rank on it, whatever the literature says.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("--out", default=None)
    m.add_argument("--source", default=None)
    # Default OFF. The null is most informative over EVERY fully-scored design,
    # not just the accepted ones: it asks whether the metrics separate real
    # designs from scrambles at all, and restricting it to designs that already
    # passed those same metrics both shrinks n and biases the comparison.
    m.add_argument("--only-accepted", action="store_true", default=False,
                   help="restrict to accepted designs (default: all scored)")
    r = sub.add_parser("report"); r.add_argument("--null", required=True)
    for p in (m, r):
        p.add_argument("--config", default=None)
    a = ap.parse_args()

    cfg = common.load_config(a.config)
    root = Path(cfg["run_root"]) / cfg["campaign"]
    (cmd_make if a.cmd == "make" else cmd_report)(cfg, root, a)


if __name__ == "__main__":
    main()
