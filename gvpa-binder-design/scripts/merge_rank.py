#!/usr/bin/env python
"""Merge every scoring stage, apply the BindCraft gate, and rank.

Acceptance is a HARD GATE, not a score. A design either clears every criterion
for its tier or it does not; the composite below only orders designs that have
already passed, and is never allowed to rescue a failure. That separation is
deliberate -- a weighted sum lets a superb ipTM paper over a positive dG.

Tiers (see configs/filters_strict.json):
  tier 1  >= 55 aa   full BindCraft-default set, including the free-monomer
                     fold criteria.
  tier 2  33-54 aa   every interface criterion identical, but the free-monomer
                     criteria are REPLACED by a tightened in-complex binder
                     pLDDT (0.85) plus a secondary-structure requirement,
                     because a GvpC-style repeat is disordered until it binds
                     and BindCraft's own filters would reject native GvpC.

Where BindCraft gives both a per-model and a mean threshold, both are enforced:
every scored model must pass, AND the mean must pass.

    python scripts/merge_rank.py [--config ...] [--boltz <boltz.csv>]

Writes <run>/merged.csv (everything, with per-criterion verdicts) and
<run>/accepted.csv (the gate survivors).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

NAN = float("nan")
OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
       ">": lambda a, b: a > b, "<": lambda a, b: a < b}

#: Axes used for the composite ordering of ALREADY-ACCEPTED designs.
#: Boltz is deliberately absent -- see scripts/run_boltz.py.
COMPOSITE = [("i_pae", -1), ("i_ptm", +1), ("plddt", +1), ("interface_dG", -1),
             ("interface_sc", +1), ("interface_unsat_hbonds", -1),
             ("scrmsd_complex", -1)]


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fnum(v):
    try:
        f = float(v)
        return f if not math.isnan(f) else NAN
    except (TypeError, ValueError):
        return NAN


def rib_overhang(pdb, binder_len=None):
    """Does the binder stick out past the target rib's own cross-section?

    In an intact gas vesicle the GvpA ribs pack edge to edge (rib spacing
    ~46-57 A against this rib's 51 A footprint), so anything protruding
    sideways occupies space the NEXT rib fills. We model one isolated rib, so
    nothing in the scoring sees that clash -- this is the same blind spot that
    let a third of campaign 1's candidates interpenetrate the omitted 4th
    subunit, rotated 90 degrees.

    Axes are derived from the target itself: the rib axis from its outermost
    subunit centroids, then PCA on what is left -- the wider of the two
    remaining axes is 'lateral' (across the ribbon), the narrower is the
    surface normal. No hotspot or config input needed, so it works on any
    complex.

    lateral_overhang = how far the binder reaches beyond the target's own
    lateral span, in Angstroms. 0 means it stays within the rib's footprint.
    """
    try:
        st = common.read_structure(pdb)
        chains = list(dict.fromkeys(str(c) for c in st.chain))
        lens = {c: int(((st.chain == c) & (st.atom == "CA")).sum()) for c in chains}
        cand = [c for c in chains if lens[c] == int(binder_len)] if binder_len else []
        b = cand[-1] if cand else max(chains, key=lambda c: lens[c])
        tch = [c for c in chains if c != b]
        if len(tch) < 2:
            return {}
        T = st.xyz[np.isin(st.chain, tch)]
        B = st.xyz[st.chain == b]
        cents = np.array([st.xyz[st.chain == c].mean(0) for c in tch])
        core = T.mean(0)
        axis = cents[-1] - cents[0]
        nrm = np.linalg.norm(axis)
        if nrm == 0:
            return {}
        axis /= nrm
        P = T - core
        P = P - np.outer(P @ axis, axis)          # strip the rib axis
        _, _, V = np.linalg.svd(P, full_matrices=False)
        latv, normv = V[0], V[1]                  # wider = lateral, narrower = normal
        TLo, THi = (T - core) @ latv, None
        TL = (T - core) @ latv
        BL = (B - core) @ latv
        BN = (B - core) @ normv
        TN = (T - core) @ normv
        return {
            "lateral_overhang": round(float(max(0.0, BL.max() - TL.max())
                                            + max(0.0, TL.min() - BL.min())), 2),
            "rib_lateral_width": round(float(TL.max() - TL.min()), 1),
            "binder_lateral_span": round(float(BL.max() - BL.min()), 1),
            "binder_above_surface": round(float(BN.max() - TN.max()), 1),
        }
    except Exception:
        return {}


def termini_geometry(pdb, binder_len=None):
    """Binder N/C terminus geometry: how far apart, and which way they face.

    Buildability, not affinity: fusing a tag, grafting into a scaffold or
    displaying multivalently all need the two ends close together and pointing
    away from the target. Measured on the campaign, it costs nothing in binding
    quality (fusable subset median i_pae 6.09 A vs 6.08 A overall), so it is
    REPORTED and never enters the composite -- only ~2% of designs satisfy it
    strictly, and ranking on it would drown the interface metrics.
    """
    try:
        st = common.read_structure(pdb)
        chains = list(dict.fromkeys(str(c) for c in st.chain))
        lens = {c: int(((st.chain == c) & (st.atom == "CA")).sum()) for c in chains}
        # AF2 writes the target first and the binder LAST; identify by length.
        cand = [c for c in chains if lens[c] == int(binder_len)] if binder_len else []
        b = cand[-1] if cand else max(chains, key=lambda c: lens[c])
        bm = (st.chain == b) & (st.atom == "CA")
        ca, rs = st.xyz[bm], st.resseq[bm]
        if len(ca) < 3:
            return {}
        o = np.argsort(rs)
        n_term, c_term = ca[o[0]], ca[o[-1]]
        tgt = st.xyz[~np.isin(st.chain, [b])]
        if len(tgt) == 0:
            return {}
        d = np.linalg.norm(tgt[:, None, :] - ca[None, :, :], axis=2).min(axis=1)
        contact = tgt[d <= 8.0]
        axis = (contact if len(contact) else tgt).mean(0) - ca.mean(0)
        nrm = np.linalg.norm(axis)
        axis = axis / nrm if nrm else axis
        pn = float(np.dot(n_term - ca.mean(0), axis))
        pc = float(np.dot(c_term - ca.mean(0), axis))
        return {
            "termini_dist": round(float(np.linalg.norm(n_term - c_term)), 2),
            "termini_n_to_target": round(float(np.linalg.norm(tgt - n_term, axis=1).min()), 2),
            "termini_c_to_target": round(float(np.linalg.norm(tgt - c_term, axis=1).min()), 2),
            "termini_same_side": int(np.sign(pn) == np.sign(pc)),
            "termini_both_away": int(pn < 0 and pc < 0),
        }
    except Exception:
        return {}


def ss_percentages(pdb, binder_len=None):
    """Binder helix / sheet / loop percentages from a predicted complex."""
    try:
        st = common.read_structure(pdb)
        order = list(dict.fromkeys(st.chain))
        ch = order[-1]
        if binder_len:
            cands = [c for c in order if len(st.ca(c)) == binder_len]
            if cands:
                ch = cands[-1]
        ss = common.secondary_structure(st.ca(ch))
        return (ss["helix"] * 100, ss["sheet"] * 100, ss["loop"] * 100)
    except Exception:
        return NAN, NAN, NAN


#: Kyte-Doolittle, for a one-number hydropathy summary of the binder sequence.
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
      "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
      "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}


def composition(seq):
    """Per-design amino-acid composition and bulk sequence properties.

    Reported, never gated. Composition is how you notice that a run has
    collapsed onto poly-Ala/poly-Glu solutions -- a failure mode that every
    confidence metric happily rates as excellent.
    """
    n = len(seq) or 1
    out = {f"aa_{aa}": round(seq.count(aa) / n, 4) for aa in "ACDEFGHIKLMNPQRSTVWY"}
    out["frac_hydrophobic"] = round(sum(seq.count(a) for a in "AVILMFWC") / n, 4)
    out["frac_charged"] = round(sum(seq.count(a) for a in "DEKR") / n, 4)
    out["frac_aromatic"] = round(sum(seq.count(a) for a in "FWY") / n, 4)
    out["net_charge"] = sum(seq.count(a) for a in "KR") - sum(seq.count(a) for a in "DE")
    out["hydropathy_gravy"] = round(sum(KD.get(a, 0) for a in seq) / n, 3)
    return out


def collect(cfg, root, boltz_path=None):
    """id -> {metric: {"models": {m: v}, "mean": v}} plus identity columns."""
    seqs = {}
    for f in (sorted(glob.glob(str(root / "sequences.shard*.csv")))
              or [str(root / "sequences.csv")]):
        for r in read_csv(f):
            prev = seqs.get(r["id"])
            if prev is not None and prev["seq"] != r["seq"]:
                raise SystemExit(
                    f"sequence id {r['id']} appears twice with different sequences. "
                    f"Every metric downstream is keyed by id, so this would attribute "
                    f"scores to the wrong sequence. Re-run stage 03 after deleting "
                    f"{root/'seqs'} and {root/'sequences.csv'}.")
            seqs[r["id"]] = r

    # Prefer the merged file explicitly. "scores*.csv" also matches the
    # per-shard scores.N.csv; that happens to be safe today only because the
    # merged name sorts last. Relying on sort order is how the Rosetta glob bug
    # happened, so the choice is made explicitly here.
    af2 = {}
    merged_af2 = root / "af2_full" / "scores.csv"
    af2_files = ([str(merged_af2)] if merged_af2.exists()
                 else sorted(glob.glob(str(root / "af2_full" / "scores.[0-9]*.csv"))))
    for f in af2_files:
        for r in read_csv(f):
            af2[r["id"]] = r
    if not af2:
        raise SystemExit(f"no AF2 full scores under {root/'af2_full'}")

    # ONLY the merged per-model files, matched exactly. A loose "rosetta*.csv"
    # glob also picks up the per-shard files that stage 06 leaves behind, and
    # every one of those parsed as model 0 ("rosetta.m1.shard3" -> "1.shard3"
    # -> ValueError -> 0). Sorted order then let stale shard data overwrite the
    # merged file and model-1 results masquerade as model 0, collapsing the two
    # AF2 models into one and voiding every per-model Rosetta check.
    ros = {}
    rx = re.compile(r"^rosetta\.m(\d+)\.csv$")
    for f in sorted(glob.glob(str(root / "rosetta.m*.csv"))):
        mt = rx.match(Path(f).name)
        if not mt:
            continue
        m = int(mt.group(1))
        for r in read_csv(f):
            ros.setdefault(r["id"], {})[m] = r
    if not ros:
        print("WARNING: no rosetta.m<N>.csv found -- every Rosetta criterion "
              "will fail as missing", flush=True)

    boltz = {}
    if boltz_path:
        for f in ([boltz_path] if Path(boltz_path).is_file()
                  else sorted(glob.glob(str(Path(boltz_path) / "*.csv")))):
            for r in read_csv(f):
                boltz[r["id"]] = r

    models = cfg["af2"]["full_models"]
    out = {}
    for sid, srow in seqs.items():
        a = af2.get(sid)
        if not a:
            continue
        blen = int(srow["binder_len"])
        rec = {"id": sid, "design": srow["design"], "cell": srow["cell"],
               "arm": srow["arm"], "length_bin": srow["length_bin"],
               "binder_len": blen, "seq": srow["seq"], "pdb": srow["pdb"],
               "mpnn_seq_recovery": srow.get("mpnn_seq_recovery", "")}
        met = {}

        for key in ["plddt", "ptm", "i_ptm", "pae", "i_pae", "hotspot_rmsd",
                    "scrmsd_complex", "target_rmsd", "binder_alone_plddt",
                    "binder_alone_ptm", "binder_rmsd"]:
            met[key] = {m: fnum(a.get(f"m{m}_{key}")) for m in models}

        for m in models:
            p = a.get(f"m{m}_complex_pdb")
            hel, sht, loop = ss_percentages(p, binder_len=blen) if p else (NAN, NAN, NAN)
            met.setdefault("binder_helix_pct", {})[m] = hel
            met.setdefault("binder_sheet_pct", {})[m] = sht
            met.setdefault("binder_loop_pct", {})[m] = loop
            met.setdefault("binder_helix_sheet_pct", {})[m] = (
                hel + sht if hel == hel and sht == sht else NAN)

        # Termini geometry from the first model's complex (reported, not ranked)
        p0 = a.get(f"m{models[0]}_complex_pdb")
        if p0:
            rec.update(termini_geometry(p0, binder_len=blen))
            rec.update(rib_overhang(p0, binder_len=blen))

        rec.update(composition(srow["seq"]))
        # Carry the predicted structures forward so the report can snapshot them
        # without re-deriving where AF2 put its output.
        m0 = models[0]
        rec["af2_complex_pdb"] = a.get(f"m{m0}_complex_pdb", "")
        rec["af2_alone_pdb"] = a.get(f"m{m0}_alone_pdb", "")

        rkeys = ["binder_score", "surface_hydrophobicity", "interface_sc",
                 "interface_packstat", "interface_dG", "interface_dSASA",
                 "interface_dG_SASA_ratio", "interface_fraction",
                 "interface_hydrophobicity", "interface_nres", "interface_hbonds",
                 "interface_unsat_hbonds", "interface_hbond_pct", "interface_unsat_pct"]
        rkeys += [f"iAA_{aa}" for aa in "ACDEFGHIKLMNPQRSTVWY"]
        rr = ros.get(sid, {})
        for key in rkeys:
            met[key] = {m: fnum(rr[m].get(key)) for m in rr if key in rr[m]} or {0: NAN}
        if sid in ros and 0 in ros[sid]:
            rec["interface_residues"] = ros[sid][0].get("interface_residues", "")

        for k, v in (boltz.get(sid) or {}).items():
            if k.startswith("boltz_"):
                rec[k] = fnum(v)

        rec["_metrics"] = {k: {"models": v,
                               "mean": float(np.nanmean([x for x in v.values()]))
                               if any(x == x for x in v.values()) else NAN}
                           for k, v in met.items()}
        out[sid] = rec
    return out


def evaluate(rec, filters, cfg):
    """Apply the tier's criteria. Returns (tier, passed, [failures])."""
    blen = rec["binder_len"]
    t = filters["tiers"]
    tier = "tier1" if blen >= t["tier1"]["min_len"] else "tier2"
    crit = list(filters["criteria"]["shared"]) + list(filters["criteria"][f"{tier}_only"])
    crit += [c for c in filters["extra"] if c.get("enabled", True)]

    fails = []
    for c in crit:
        met = rec["_metrics"].get(c["metric"])
        if met is None:
            fails.append(f"{c['metric']}:missing")
            continue
        op, thr = OPS[c["op"]], c["threshold"]
        vals = [v for v in met["models"].values() if v == v]
        if not vals:
            fails.append(f"{c['metric']}:nan")
            continue
        if c.get("per_model") and not all(op(v, thr) for v in vals):
            fails.append(f"{c['metric']}{c['op']}{thr}:model")
        elif not c.get("per_model") and not op(met["mean"], thr):
            fails.append(f"{c['metric']}{c['op']}{thr}:mean")
    return tier, not fails, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--filters", default=None)
    ap.add_argument("--out-prefix", default="",
                    help="prefix for merged.csv/accepted.csv, e.g. 'relaxed_'. "
                         "Lets a second filter set be evaluated without "
                         "overwriting the primary (strict) results.")
    ap.add_argument("--boltz", default=None)
    a = ap.parse_args()

    cfg = common.load_config(a.config)
    root = Path(cfg["run_root"]) / cfg["campaign"]

    # A merged scores.csv older than the shards it came from is the failure
    # that silently truncated this campaign: an interrupted stage 05 was
    # resumed in a later session, which backfilled the free-binder pass into
    # af2_full/scores.[0-9].csv, but nothing re-merged them. merge_rank then
    # read a scores.csv with binder_alone_plddt missing for 2341 of 3000 rows,
    # every tier-1 design failed on the MISSING metric, and the accepted set
    # came out looking like "only short binders work".
    for _sub in ("af2_full", "af2_gate"):
        _m = root / _sub / "scores.csv"
        if not _m.exists():
            continue
        _newer = [f for f in (root / _sub).glob("scores.[0-9].csv")
                  if f.stat().st_mtime > _m.stat().st_mtime + 1]
        if _newer:
            raise SystemExit(
                f"{_sub}/scores.csv is OLDER than {len(_newer)} of its shards "
                f"(e.g. {_newer[0].name}). Re-merge before ranking, or metrics "
                f"written after the merge will read as missing and silently "
                f"fail their criteria.")
    fpath = a.filters or Path(cfg["_root"]) / "configs" / "filters_bindcraft.json"
    filters = json.loads(Path(fpath).read_text())

    recs = collect(cfg, root, a.boltz)
    if not recs:
        raise SystemExit("nothing to merge")

    for r in recs.values():
        r["tier"], ok, fails = evaluate(r, filters, cfg)
        r["accepted"] = int(ok)
        r["n_failed_criteria"] = len(fails)
        r["failed_criteria"] = ";".join(fails)

    # Composite: mean percentile rank over orthogonal axes, computed on the
    # ACCEPTED set only so the ordering is not distorted by the failures.
    acc = [r for r in recs.values() if r["accepted"]]
    for axis, sign in COMPOSITE:
        vals = np.array([r["_metrics"].get(axis, {}).get("mean", NAN) for r in acc], float)
        good = vals == vals
        pct = np.full(len(vals), NAN)
        if good.sum() > 1:
            order = np.argsort(np.argsort(sign * vals[good]))
            pct[good] = order / (good.sum() - 1)
        elif good.sum() == 1:
            pct[good] = 1.0
        for r, v in zip(acc, pct):
            r[f"pct_{axis}"] = round(float(v), 4) if v == v else ""
    for r in acc:
        ps = [r[f"pct_{ax}"] for ax, _ in COMPOSITE if r.get(f"pct_{ax}") != ""]
        r["composite"] = round(float(np.mean(ps)), 4) if ps else 0.0
    for r in recs.values():
        r.setdefault("composite", "")

    flat = []
    for r in recs.values():
        row = {k: v for k, v in r.items() if k != "_metrics"}
        for k, m in r["_metrics"].items():
            row[k] = round(m["mean"], 4) if m["mean"] == m["mean"] else ""
            for mi, v in m["models"].items():
                row[f"{k}_m{mi}"] = round(v, 4) if v == v else ""
        flat.append(row)
    flat.sort(key=lambda r: (-(r["accepted"]), -(r["composite"] or 0)))

    head = ["id", "design", "arm", "cell", "length_bin", "binder_len", "tier",
            "accepted", "composite", "n_failed_criteria", "failed_criteria", "seq"]
    cols = head + sorted({k for r in flat for k in r} - set(head) - {"_metrics"})
    for name, rows in [("merged.csv", flat),
                       ("accepted.csv", [r for r in flat if r["accepted"]])]:
        name = a.out_prefix + name
        with open(root / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    n = len(flat); na = sum(r["accepted"] for r in flat)
    print(f"scored   : {n}")
    print(f"accepted : {na} ({na/n:.1%})")
    from collections import Counter
    for tier in ("tier1", "tier2"):
        sub = [r for r in flat if r["tier"] == tier]
        if sub:
            print(f"  {tier}: {sum(r['accepted'] for r in sub)}/{len(sub)}")
    fc = Counter(f.split(":")[0] for r in flat if not r["accepted"]
                 for f in r["failed_criteria"].split(";") if f)
    print("\nmost common failures:")
    for k, v in fc.most_common(12):
        print(f"   {k:<42} {v}")
    print(f"\nwrote {root/'merged.csv'} and {root/'accepted.csv'}")


if __name__ == "__main__":
    main()
