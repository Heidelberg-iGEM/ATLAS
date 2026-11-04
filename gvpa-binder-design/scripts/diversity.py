#!/usr/bin/env python
"""Pick a geometrically diverse shortlist from the accepted designs.

Ranking by score alone produces an order list of near-duplicates: the top
designs tend to come from a handful of backbones, approach the epitope from the
same direction and bury the same residues. Ordering 48 versions of one binder
tests one hypothesis 48 times.

Diversity is enforced on three independent axes, in decreasing order of how
much they matter for an experiment:

  epitope footprint  Jaccard overlap of the target residues actually contacted.
                     Two binders burying the same residues are the same
                     experiment regardless of their folds.
  approach direction angle between the epitope-centroid -> binder-centroid
                     vectors. Separates a binder lying along the rib from one
                     standing off it.
  sequence identity  guards against MPNN emitting near-identical sequences on
                     different backbones.

Plus a hard cap on sequences per backbone, since one backbone yields eight
sequences that dock identically.

Generation-side diversity (arm x length bin x sampler setting) is handled in
make_inputs.py; this is the selection-side half.

    python scripts/diversity.py [--n 48] [--out <run>/shortlist.csv]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def seq_identity(a, b):
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] == b[i]) / max(len(a), len(b))


def features(cfg, root):
    """design -> (contact set, approach unit vector)."""
    st = common.read_structure(common.resolve(cfg, cfg["target"]["cif"]))
    tips = []
    for arm in cfg["arms"].values():
        for hs in arm["hotspots"]:
            ch, rs = hs[0], int(hs[1:])
            m = (st.chain == ch) & (st.resseq == rs)
            if m.any():
                tips.append(st.xyz[m].mean(0))
    epicentre = np.mean(tips, axis=0)

    out = {}
    with open(root / "backbones.csv") as fh:
        for r in csv.DictReader(fh):
            if r.get("pass") != "1" or not r.get("pdb"):
                continue
            fp = set((r.get("contact_fingerprint") or "").split(";")) - {""}
            try:
                bb = common.read_structure(r["pdb"])
                v = bb.ca(cfg["target"]["binder_chain"]).mean(0) - epicentre
                n = np.linalg.norm(v)
                out[r["design"]] = (fp, v / n if n else np.zeros(3))
            except Exception:
                out[r["design"]] = (fp, np.zeros(3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--accepted", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()

    cfg = common.load_config(a.config)
    root = Path(cfg["run_root"]) / cfg["campaign"]
    dv = cfg["diversity"]
    n_target = a.n or dv["target_list_size"]

    acc_path = Path(a.accepted or root / "accepted.csv")
    with open(acc_path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"no accepted designs in {acc_path}")
    rows.sort(key=lambda r: -float(r.get("composite") or 0))

    feats = features(cfg, root)
    jcut, acut = dv["contact_jaccard_cutoff"], np.cos(np.radians(dv["approach_angle_cutoff"]))
    scut, per_cluster = dv["seq_identity_cutoff"], dv["max_per_cluster"]
    per_bb = dv.get("max_per_backbone", 2)

    clusters, chosen, per_backbone = [], [], {}
    for r in rows:
        if len(chosen) >= n_target:
            break
        if per_backbone.get(r["design"], 0) >= per_bb:
            continue
        fp, vec = feats.get(r["design"], (set(), np.zeros(3)))

        # near-duplicate sequence to anything already picked -> drop outright
        if any(seq_identity(r["seq"], c["seq"]) > scut for c in chosen):
            continue

        hit = None
        for ci, cl in enumerate(clusters):
            same_site = jaccard(fp, cl["fp"]) > jcut
            same_dir = float(np.dot(vec, cl["vec"])) > acut
            if same_site and same_dir:
                hit = ci
                break
        if hit is None:
            clusters.append({"fp": fp, "vec": vec, "n": 1})
            cid = len(clusters) - 1
        elif clusters[hit]["n"] < per_cluster:
            clusters[hit]["n"] += 1
            cid = hit
        else:
            continue

        r["cluster"] = cid
        r["rank"] = len(chosen) + 1
        chosen.append(r)
        per_backbone[r["design"]] = per_backbone.get(r["design"], 0) + 1

    out = Path(a.out or root / "shortlist.csv")
    cols = ["rank", "cluster"] + [c for c in rows[0] if c not in ("rank", "cluster")]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(chosen)

    from collections import Counter
    print(f"accepted pool : {len(rows)}")
    print(f"shortlisted   : {len(chosen)} across {len(clusters)} geometric clusters")
    print(f"  by arm      : {dict(Counter(r['arm'] for r in chosen))}")
    print(f"  by tier     : {dict(Counter(r['tier'] for r in chosen))}")
    print(f"  by length   : {dict(Counter(r['length_bin'] for r in chosen))}")
    print(f"  backbones   : {len(per_backbone)} distinct (cap {per_bb}/backbone)")
    if len(chosen) < n_target:
        print(f"\nNOTE: {len(chosen)} < requested {n_target}. The diversity caps bind "
              f"before the pool runs out -- relax diversity.max_per_cluster or "
              f"contact_jaccard_cutoff, or generate more backbones.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
