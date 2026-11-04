#!/usr/bin/env python3
"""
Register diagnosis: WHERE on the target does the binder land, and WITH WHICH
helix face?

    python 33_boltz_register.py /path/predictions --binder F --targets A,B,C,D
    python 33_boltz_register.py /path/predictions --only original_GvpC --per-seed
    python 33_boltz_register.py . --single /path/template.cif --binder F --targets A,B,C,D

Complements seed_consistency.py / 32_boltz_rank.py: those measure reproducibility in
Angstrom, this script measures it in *residues*. Three quantities, three questions:

  site_frac    Fraction of the contacts that fall on the epitope residues (--site).
               "Does the binder hit the right surface?"
               REGISTER-FREE: it does not matter WHICH binder residues provide this,
               so that a new binding mode is not penalized.

  face_deg     Direction of the binding face on the helical wheel, relative to the
               native anchor set (--anchors). 100 degrees = one position.
               "Does it bind with the same helix face as the WT?"
               A value around +-100 degrees is a face rotated by one position
               -- exactly the finding from handover 2b (MPNN puts large hydrophobics
               on 4,8,15,22,25,29 instead of 4,11,12,15,19,22).

  tcoord       Centre of mass of the contacts in a continuous target coordinate
               (chain index * chain length + res_id). Across seeds, its spread
               is the axial slip in residues.
               "Does the same binder slip by whole repeat units between seeds?"
               This is the WT signature from handover 4 (shape_rmsd 0.35 A, but
               placement_rmsd 7.2 A, 96 % of it axial) in residues instead of Angstrom.

IMPORTANT FOR INTERPRETATION
  face_R  is the concentration of the contacts on the helical wheel (0..1). Below ~0.4
  the contacts do not lie on one side, and face_deg is meaningless --
  the script then writes n/a instead of a number. Do not ignore this: a binder
  without a defined binding face is the more interesting finding.

  In a run WITH a pocket constraint, site_frac is partly circular -- you
  gave Boltz the target residues. face_deg is NOT: the constraint
  only names the target side (binder: F, contacts: [[A,57],...]); which
  binder residues serve it stays free.

Dependencies: gemmi, numpy.
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

try:
    import gemmi
except ImportError:
    sys.exit("gemmi is missing. Activate the right env or: pip install gemmi")

DEG_PER_RES = 100.0          # alpha helix, 3.6 residues per turn
MIN_FACE_R = 0.40            # below this face_deg is not defined


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------

def load_residues(path):
    """{chain: [(res_id, resname, xyz_heavy_atoms), ...]} in chain order."""
    st = gemmi.read_structure(str(path))
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_waters()
    if len(st) == 0:
        raise SystemExit(f"{path}: gemmi read ZERO models -- broken mmCIF?")
    out = {}
    for chain in st[0]:
        res = []
        for r in chain:
            xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r
                            if a.element != gemmi.Element("H")], dtype=float)
            if xyz.size:
                res.append((r.seqid.num, r.name, xyz))
        if res:
            out[chain.name] = res
    return out


def contacts(res, binder, targets, cutoff):
    """[(b_resid, b_aa, t_chain, t_resid, t_aa, n_atom_pairs)] below cutoff."""
    bres = res[binder]
    bcen = [(x.mean(0), np.linalg.norm(x - x.mean(0), axis=1).max()) for _, _, x in bres]
    out = []
    for tc in targets:
        for tid, tname, txyz in res[tc]:
            tc_cen = txyz.mean(0)
            tc_rad = float(np.linalg.norm(txyz - tc_cen, axis=1).max())
            for (bid, bname, bxyz), (bc, br) in zip(bres, bcen):
                if np.linalg.norm(bc - tc_cen) > cutoff + br + tc_rad:
                    continue
                d = np.linalg.norm(bxyz[:, None, :] - txyz[None, :, :], axis=-1)
                n = int((d < cutoff).sum())
                if n:
                    out.append((bid, bname, tc, tid, tname, n))
    return out


# --------------------------------------------------------------------------
# Register
# --------------------------------------------------------------------------

def face_vector(positions, weights):
    """Circular mean of the contact directions on the helical wheel.

    positions: relative binder positions (1-based), weights: contact counts.
    Returns (angle_deg, R) with R = concentration in [0, 1].
    """
    if not len(positions):
        return np.nan, 0.0
    ang = np.deg2rad(DEG_PER_RES * (np.asarray(positions, float) - 1.0))
    w = np.asarray(weights, float)
    v = np.array([(w * np.cos(ang)).sum(), (w * np.sin(ang)).sum()])
    R = float(np.linalg.norm(v) / w.sum())
    return float(np.rad2deg(np.arctan2(v[1], v[0]))), R


def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0


def metrics(cons, boff, tlen, tindex, site):
    """Metrics for ONE model from its contact list."""
    m = dict(n_contacts=0, n_iface_res=0, site_frac=np.nan, chains_hit="",
             face_deg=np.nan, face_R=0.0, tcoord=np.nan, iface_pos=[],
             per_pos={})
    if not cons:
        return m

    per_pos = defaultdict(int)
    n_site = n_all = 0
    tsum = tw = 0.0
    chains = set()
    for bid, _, tc, tid, _, n in cons:
        per_pos[bid - boff + 1] += n
        n_all += n
        if site[0] <= tid <= site[1]:
            n_site += n
        chains.add(tc)
        tsum += n * (tindex[tc] * tlen + tid)
        tw += n

    pos = sorted(per_pos)
    w = [per_pos[p] for p in pos]
    deg, R = face_vector(pos, w)
    m.update(n_contacts=n_all, n_iface_res=len(pos), site_frac=n_site / n_all,
             chains_hit="".join(sorted(chains)), face_deg=deg, face_R=R,
             tcoord=tsum / tw, iface_pos=pos, per_pos=dict(per_pos))
    return m


def footprint_line(per_pos, nres, anchors=None):
    """One line per model: '.' no contact, 1-9/# log-scaled density.
    Anchor positions are marked as '^' in a legend line below,
    so that the rotation of the binding face can be read off directly."""
    s = []
    for p in range(1, nres + 1):
        n = per_pos.get(p, 0)
        s.append("." if n == 0 else "123456789#"[min(int(np.log2(n)), 9)])
    return "".join(s)


# --------------------------------------------------------------------------

def model_files(vdir):
    fs = [p for p in glob.glob(os.path.join(vdir, "*model_*.pdb"))
          + glob.glob(os.path.join(vdir, "*model_*.cif"))
          if not os.path.basename(p).startswith(("pae_", "pde_", "plddt_",
                                                 "confidence_"))]
    return sorted(fs, key=lambda p: int(re.search(r"model_(\d+)", p).group(1)))


def analyse(path, binder, targets, cutoff, site, anchors):
    res = load_residues(path)
    miss = [c for c in [binder] + targets if c not in res]
    if miss:
        raise KeyError(f"chains {miss} missing (present: {sorted(res)})")
    boff = min(r[0] for r in res[binder])
    nres = len(res[binder])
    tlen = max(len(res[c]) for c in targets) + 1
    tindex = {c: i for i, c in enumerate(targets)}
    m = metrics(contacts(res, binder, targets, cutoff), boff, tlen, tindex, site)
    m["binder_offset"] = boff
    m["binder_len"] = nres
    if anchors is not None and np.isfinite(m["face_deg"]):
        ref, _ = face_vector(anchors, [1] * len(anchors))
        m["face_deg"] = wrap180(m["face_deg"] - ref)
    return m


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split("\n\n")[0])
    ap.add_argument("root", help="predictions folder (ignored with --single)")
    ap.add_argument("--single", metavar="PATH", default=None,
                    help="a single structure instead of a folder")
    ap.add_argument("--binder", default="F")
    ap.add_argument("--targets", default="A,B,C,D")
    ap.add_argument("--site", default="53-64",
                    help="epitope residues of the target for site_frac")
    ap.add_argument("--anchors", default="4,11,12,15,19,22",
                    help="native anchor positions, RELATIVE (1-based). "
                         "'none' = face_deg absolute instead of relative to the WT")
    ap.add_argument("--cutoff", type=float, default=5.0,
                    help="heavy-atom distance (A)")
    ap.add_argument("--only", default="",
                    help="only variants whose name contains this substring")
    ap.add_argument("--per-seed", action="store_true",
                    help="show every seed individually instead of only the mean")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--prefix", default="",
                    help="prefix for all columns except the ID. With "
                         "--prefix bz_ the script writes directly into the "
                         "registry convention.")
    ap.add_argument("--id-col", default="variant",
                    help="name of the key column, e.g. seq_id.")
    ap.add_argument("--per-seed-csv", default=None,
                    help="per-seed table as CSV (default: "
                         "register_per_seed.csv next to --csv). ALWAYS "
                         "written, also without --per-seed: the aggregate only says "
                         "yes/no for chain coverage, but the pipeline needs "
                         "in HOW MANY seeds all chains are hit "
                         "(gate.min_chain_seeds).")
    a = ap.parse_args()

    targets = [t.strip() for t in a.targets.split(",") if t.strip()]
    lo, hi = (int(x) for x in a.site.split("-"))
    anchors = (None if a.anchors.lower() == "none"
               else [int(x) for x in a.anchors.split(",")])

    # ---------------- single structure ----------------
    if a.single:
        try:
            m = analyse(a.single, a.binder, targets, a.cutoff, (lo, hi), anchors)
        except KeyError as e:
            sys.exit(f"{a.single}: {e.args[0]}")
        print(f"\n{os.path.basename(a.single)}")
        print(f"  Binder {a.binder}: {m['binder_len']} residues, "
              f"res_id from {m['binder_offset']}")
        if m["n_contacts"] == 0:
            sys.exit(f"\nZERO contacts between {a.binder} and "
                     f"{','.join(targets)} below {a.cutoff} A.\n"
                     f"That is almost always a chain or numbering error, "
                     f"not a finding.\nCheck: --binder / --targets, and whether the "
                     f"file really contains the complex.")
        print(f"  contacts           {m['n_contacts']} "
              f"on {m['n_iface_res']} binder residues")
        print(f"  chains hit         {m['chains_hit']}  (expected {''.join(targets)})")
        print(f"  site_frac ({lo}-{hi})  {m['site_frac']:.3f}")
        print(f"  face_R             {m['face_R']:.2f}"
              f"{'   << below 0.40: no defined binding face' if m['face_R'] < MIN_FACE_R else ''}")
        if m["face_R"] >= MIN_FACE_R:
            print(f"  face_deg           {m['face_deg']:+.0f} degrees "
                  f"({m['face_deg']/DEG_PER_RES:+.2f} positions)"
                  f"{'  [relative to the anchor set]' if anchors else '  [absolute]'}")
        rel = [p for p in m["iface_pos"]]
        print(f"  interface positions (relative): {','.join(map(str, rel))}")
        print(f"  absolute:                       "
              f"{','.join(str(p + m['binder_offset'] - 1) for p in rel)}")
        if anchors:
            hit = sorted(set(rel) & set(anchors))
            print(f"  Overlap with anchors {anchors}: "
                  f"{hit} ({len(hit)}/{len(anchors)})")
        return

    # ---------------- folder with variants ----------------
    vdirs = sorted(d for d in glob.glob(os.path.join(a.root, "*"))
                   if os.path.isdir(d) and model_files(d))
    if a.only:
        vdirs = [d for d in vdirs if a.only.lower() in os.path.basename(d).lower()]
    if not vdirs:
        sys.exit(f"No variant folders with model files under {a.root}"
                 + (f" (filter --only {a.only})" if a.only else ""))

    rows, per_seed_rows, n_empty = [], [], [0]
    for vd in vdirs:
        v = os.path.basename(vd)
        ms = []
        for f in model_files(vd):
            try:
                m = analyse(f, a.binder, targets, a.cutoff, (lo, hi), anchors)
            except KeyError as e:
                print(f"  ! {v}/{os.path.basename(f)}: {e}", file=sys.stderr)
                continue
            if m["n_contacts"] == 0:
                print(f"  ! {v}/{os.path.basename(f)}: ZERO contacts "
                      f"{a.binder}<->{','.join(targets)} -- skipped",
                      file=sys.stderr)
                n_empty[0] += 1
                continue
            m["model"] = int(re.search(r"model_(\d+)", f).group(1))
            ms.append(m)
            per_seed_rows.append(dict(
                variant=v, footprint=footprint_line(m["per_pos"], m["binder_len"]),
                **{k: m[k] for k in (
                    "model", "n_contacts", "n_iface_res", "site_frac",
                    "chains_hit", "face_deg", "face_R", "tcoord")}))
        if not ms:
            continue

        def mean(k):
            x = np.array([m[k] for m in ms], float)
            return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

        # average face_deg circularly, not arithmetically
        fd = np.array([m["face_deg"] for m in ms if m["face_R"] >= MIN_FACE_R],
                      float)
        if fd.size:
            ang = np.deg2rad(fd)
            face = float(np.rad2deg(np.arctan2(np.sin(ang).mean(),
                                               np.cos(ang).mean())))
            face_sd = float(np.rad2deg(np.sqrt(abs(
                -2 * np.log(min(max(abs(np.exp(1j * ang).mean()), 1e-12), 1.0))))))
        else:
            face, face_sd = np.nan, np.nan

        tc = np.array([m["tcoord"] for m in ms], float)
        allchains = set("".join(m["chains_hit"] for m in ms))
        # In HOW MANY seeds are all target chains hit? `all_chains`
        # below is only yes/no; the number is what the gate threshold is
        # set on (in the 20.09. set: 29 of 50 seeds, but 0 of 10 variants
        # with "in every seed").
        hit_seeds = sum(1 for m in ms if set(targets) <= set(m["chains_hit"]))
        rows.append(dict(
            variant=v, n=len(ms), chains_hit_seeds=hit_seeds,
            n_contacts=mean("n_contacts"), n_iface_res=mean("n_iface_res"),
            site_frac=mean("site_frac"), site_frac_min=float(np.nanmin(
                [m["site_frac"] for m in ms])),
            all_chains=int(all(set(targets) <= set(m["chains_hit"]) for m in ms)),
            chains_union="".join(sorted(allchains)),
            face_deg=face, face_sd=face_sd, face_R=mean("face_R"),
            tcoord=float(np.nanmean(tc)),
            tcoord_sd=float(np.nanstd(tc, ddof=1)) if tc.size > 1 else 0.0,
            tcoord_range=float(np.nanmax(tc) - np.nanmin(tc)) if tc.size > 1 else 0.0,
        ))

    if not rows:
        sys.exit(f"\nNot a single model could be evaluated ({n_empty[0]} of them without "
                 f"any contact).\nThat is a chain or numbering error, "
                 f"not a finding -- check --binder/--targets.")
    if n_empty[0]:
        print(f"\nWarning: {n_empty[0]} models without any contact "
              f"{a.binder}<->{','.join(targets)}.", file=sys.stderr)

    if a.per_seed:
        print(f'\n{"variant":<26}{"m":>2}{"site":>7}{"face":>8}{"R":>6}'
              f'{"tcoord":>9}{"#C":>6}  chains  footprint')
        for r in per_seed_rows:
            fd = "    n/a" if r["face_R"] < MIN_FACE_R else f'{r["face_deg"]:+8.0f}'
            print(f'{r["variant"]:<26}{r["model"]:>2}{r["site_frac"]:>7.2f}{fd}'
                  f'{r["face_R"]:>6.2f}{r["tcoord"]:>9.1f}{r["n_contacts"]:>6d}'
                  f'  {r["chains_hit"]:<7} {r["footprint"]}')
        if anchors:
            pad = 26 + 2 + 7 + 8 + 6 + 9 + 6 + 2 + 7 + 1
            mark = "".join("^" if p in anchors else " "
                           for p in range(1, max(
                               len(r["footprint"]) for r in per_seed_rows) + 1))
            print(" " * pad + mark + "   <- native anchors")

    rows.sort(key=lambda r: -(r["site_frac"] if np.isfinite(r["site_frac"]) else -1))
    hdr = (f'{"variant":<26}{"n":>3}{"site_frac":>11}{"min":>6}{"4ch":>5}'
           f'{"face_deg":>10}{"+-":>6}{"face_R":>8}{"tcoord_sd":>11}{"range":>7}'
           f'{"#C":>7}')
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        fd = ("       n/a     " if not np.isfinite(r["face_deg"])
              else f'{r["face_deg"]:>+10.0f}{r["face_sd"]:>6.0f}')
        print(f'{r["variant"]:<26}{r["n"]:>3}{r["site_frac"]:>11.3f}'
              f'{r["site_frac_min"]:>6.2f}{"yes" if r["all_chains"] else "NO":>5}'
              f'{fd}{r["face_R"]:>8.2f}{r["tcoord_sd"]:>11.2f}'
              f'{r["tcoord_range"]:>7.1f}{r["n_contacts"]:>7.0f}')

    print(f"\nsite_frac  fraction of contacts on target residues {lo}-{hi} "
          f"(min = worst seed)")
    print(f"4ch        all chains {''.join(targets)} contacted in EVERY seed")
    print(f"face_deg   binding face against anchors {anchors if anchors else '(absolute)'}, "
          f"100 degrees = 1 position; +- is the spread across seeds")
    print(f"face_R     concentration on one helix side; < {MIN_FACE_R} -> face_deg n/a")
    print(f"tcoord_sd  axial slip across seeds, in target residues "
          f"(range = max-min)")

    out = a.csv or os.path.join(a.root, "register_check.csv")
    cols = list(rows[0])
    header = [a.id_col] + [a.prefix + c for c in cols[1:]]
    with open(out, "w") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float)
                              else str(r[c]) for c in cols) + "\n")
    print(f"\nCSV: {out}")

    # Always write the per-seed table as well. The aggregate above only has `4ch`
    # yes/no; for the threshold "ABCD in >= N of 5 seeds" the individual seeds are
    # needed. In the 20.09. set, 4ch was met by 0 of 10 variants while
    # 29 of 50 seeds hit ABCD -- this number is only found here.
    if per_seed_rows:
        ps = a.per_seed_csv or os.path.join(os.path.dirname(out) or ".",
                                            "register_per_seed.csv")
        pcols = ["variant", "model", "n_contacts", "n_iface_res", "site_frac",
                 "chains_hit", "face_deg", "face_R", "tcoord", "footprint"]
        with open(ps, "w") as fh:
            fh.write(",".join(pcols) + "\n")
            for r in per_seed_rows:
                fh.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float)
                                  else str(r[c]) for c in pcols) + "\n")
        want = set(targets)
        ok = sum(1 for r in per_seed_rows if want <= set(r["chains_hit"]))
        full = sum(1 for v in rows if v["all_chains"])
        print(f"Per-seed CSV: {ps}")
        print(f"  {ok}/{len(per_seed_rows)} seeds hit "
              f"{''.join(sorted(want))} ({ok / len(per_seed_rows):.0%}), "
              f"{full}/{len(rows)} variants in EVERY seed")


if __name__ == "__main__":
    main()
