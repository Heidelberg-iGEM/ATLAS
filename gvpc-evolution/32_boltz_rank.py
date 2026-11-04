#!/usr/bin/env python3
"""
Stage 7: evaluate the Boltz predictions.

    $BOLTZ_PY 32_boltz_rank.py PREDICTIONS --binder F --targets A,B,C,D \\
        --site 53-64 --prefix bz_ --id-col seq_id --csv scores_boltz.csv

Two separate things, and that is intentional:

  Gate      Does the variant bind the intended surface at all, across all
            target chains, reproducibly? Exclusion criteria, not a score.
  Strength  Only for those that get through. Continuous, so that one can select
            from the gradient instead of taking the top of the list - an
            evolution round explicitly needs weaker binders as well.

Gate quantities
    shape_rmsd    Does the variant form the same structure across the seeds?
                  RMSD of the binder CAs after superposing the binder
                  onto itself, averaged over all seed pairs.
    n_chains_min  Are all target chains contacted in every seed?
    site_frac     Fraction of the contacts that land on the intended surface.
                  Catches variants that stick to the wrong side.
    iface_plddt   Is the interface itself confident?

Strength
    bsa           Buried surface area in A^2 (Shrake-Rupley). Weights how
                  deeply the hydrophobic side sinks in, not just how many
                  residues touch. This is the ranking axis of the selection.
    n_contacts    Only interpretable within the gate passers: disordered
                  chains lie flat and collect contacts.

Similarity to the wild-type footprint is deliberately not scored. That would be
an imitation metric that penalizes every new binding mode and lets the wild
type win by construction. Only the right SURFACE is required.

The gate here is for orientation. The decision is made in stage 10 from the CSV,
with the thresholds from config.yaml.

Needs gemmi and numpy.
"""
import argparse
import glob
import itertools
import json
import os
import re
import sys

import numpy as np

try:
    import gemmi
except ImportError:
    sys.exit("gemmi is missing. Activate the right env or: pip install gemmi")

# van der Waals radii for protein heavy atoms
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "SE": 1.90}
PROBE = 1.4

# Quantities that arise per seed and are averaged
SEED_KEYS = ["iface_plddt", "binder_plddt", "complex_iplddt", "bsa",
             "site_frac", "n_contacts", "n_iface_res", "n_chains",
             "iface_pae", "iface_iptm", "iptm", "iface_iptm_min", "pae_block",
             "complex_plddt", "confidence_score"]
# Quantities that only arise from comparing the seeds
VARIANT_KEYS = ["shape_rmsd", "placement_rmsd", "placement_max",
                "shift_along", "n_chains_min"]


def read_model(path):
    """Residues with coordinates, radii and pLDDT, in chain order.

    The global residue numbering corresponds to the one Boltz uses for the
    PAE matrix.
    """
    st = gemmi.read_structure(path)
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_waters()

    names, counts, coords, radii, plddt, pos, ca = [], {}, [], [], [], [], {}
    for chain in st[0]:
        names.append(chain.name)
        counts[chain.name] = len(chain)
        pts = []
        for i, res in enumerate(chain, start=1):
            xyz, rad, bf = [], [], []
            for atom in res:
                xyz.append([atom.pos.x, atom.pos.y, atom.pos.z])
                rad.append(VDW.get(atom.element.name.upper(), 1.70))
                bf.append(atom.b_iso)
            coords.append(np.array(xyz))
            radii.append(np.array(rad))
            plddt.append(float(np.mean(bf)) if bf else np.nan)
            pos.append(i)
            a = res.find_atom("CA", "*")
            if a is not None:
                pts.append([a.pos.x, a.pos.y, a.pos.z])
        if pts:
            ca[chain.name] = np.array(pts)
    return dict(names=names, counts=counts, coords=coords, radii=radii,
                plddt=np.array(plddt), pos=pos, ca=ca)


def residue_slices(names, counts):
    out, start = {}, 0
    for n in names:
        out[n] = (start, start + counts[n])
        start += counts[n]
    return out, start


def find_contacts(coords, brange, tgt_idx, cutoff):
    """Residue pairs below cutoff. Presort coarsely by centroid and radius,
    otherwise it becomes quadratic."""
    cen, rad = {}, {}
    for k in list(range(*brange)) + tgt_idx:
        c = coords[k]
        if c.size:
            cen[k] = c.mean(axis=0)
            rad[k] = float(np.linalg.norm(c - cen[k], axis=1).max())
    pairs = []
    for i in range(*brange):
        if i not in cen:
            continue
        for j in tgt_idx:
            if j not in cen or np.linalg.norm(cen[i] - cen[j]) > cutoff + rad[i] + rad[j]:
                continue
            if np.linalg.norm(coords[i][:, None, :] - coords[j][None, :, :],
                              axis=-1).min() < cutoff:
                pairs.append((i, j))
    return pairs


def _sphere(n=92):
    """Approximately evenly distributed points on the unit sphere."""
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


_SPH = _sphere(92)


def sasa(xyz, rad):
    """Solvent-accessible surface area in A^2 (Shrake-Rupley).

    Search for neighbours over a grid, otherwise it runs quadratically in the
    number of atoms.
    """
    n = len(xyz)
    if n == 0:
        return 0.0
    r = rad + PROBE
    cell = 2 * r.max()
    keys = np.floor(xyz / cell).astype(int)
    buckets = {}
    for i, k in enumerate(map(tuple, keys)):
        buckets.setdefault(k, []).append(i)
    offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]

    total = 0.0
    for i in range(n):
        near = []
        ki = keys[i]
        for o in offsets:
            near.extend(buckets.get((ki[0] + o[0], ki[1] + o[1], ki[2] + o[2]), ()))
        near = [j for j in near if j != i]
        if not near:
            total += 4 * np.pi * r[i] ** 2
            continue
        nj = np.array(near)
        keep = nj[np.linalg.norm(xyz[nj] - xyz[i], axis=1) < r[i] + r[nj]]
        pts = xyz[i] + _SPH * r[i]
        if len(keep):
            free = np.all(np.linalg.norm(pts[:, None, :] - xyz[keep][None, :, :],
                                         axis=2) >= r[keep][None, :], axis=1)
        else:
            free = np.ones(len(pts), bool)
        total += 4 * np.pi * r[i] ** 2 * free.mean()
    return float(total)


def buried_area(coords, radii, brange, tgt_idx):
    """(SASA_binder + SASA_target - SASA_complex) / 2, in A^2."""
    bi = list(range(*brange))
    bx = np.vstack([coords[k] for k in bi])
    br = np.concatenate([radii[k] for k in bi])
    tx = np.vstack([coords[k] for k in tgt_idx])
    tr = np.concatenate([radii[k] for k in tgt_idx])
    complex_sasa = sasa(np.vstack([bx, tx]), np.concatenate([br, tr]))
    return max(0.0, (sasa(bx, br) + sasa(tx, tr) - complex_sasa) / 2.0)


def kabsch(mobile, ref):
    cm, cr = mobile.mean(0), ref.mean(0)
    U, _, Vt = np.linalg.svd((mobile - cm).T @ (ref - cr))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cr - (R @ cm)


def rmsd(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def metrics_for_model(conf, path, pae_path, binder, targets, cutoff, site, do_bsa):
    """All quantities for a single model."""
    m = dict.fromkeys(SEED_KEYS)
    for k in ("iptm", "complex_iplddt", "complex_plddt", "confidence_score"):
        m[k] = conf.get(k)
    if not path:
        return m, None

    st = read_model(path)
    names, counts, coords = st["names"], st["counts"], st["coords"]
    if binder not in names:
        print(f"  ! Chain '{binder}' missing in {os.path.basename(path)} "
              f"(present: {','.join(names)})", file=sys.stderr)
        return m, None

    slices, n_res = residue_slices(names, counts)
    brange = slices[binder]
    chain_of, tgt_idx = {}, []
    for t in targets:
        if t in slices:
            for k in range(*slices[t]):
                chain_of[k] = t
                tgt_idx.append(k)
    if not tgt_idx:
        print(f"  ! no target chain in {os.path.basename(path)}", file=sys.stderr)
        return m, None

    # Interface ipTM from the pair matrix of the confidence file
    pcm = conf.get("pair_chains_iptm") or {}
    bi = names.index(binder)
    vals = []
    for t in targets:
        if t not in names:
            continue
        ti = names.index(t)
        for x, y in ((bi, ti), (ti, bi)):
            try:
                vals.append(float(pcm[str(x)][str(y)]))
            except (KeyError, TypeError, ValueError):
                pass
    if vals:
        m["iface_iptm"] = sum(vals) / len(vals)
        m["iface_iptm_min"] = min(vals)

    m["binder_plddt"] = float(np.nanmean(st["plddt"][slice(*brange)]))

    pairs = find_contacts(coords, brange, tgt_idx, cutoff)
    if pairs:
        iface_res = sorted({i for i, _ in pairs})
        m["n_contacts"] = float(len(pairs))
        m["n_iface_res"] = float(len(iface_res))
        m["iface_plddt"] = float(np.nanmean(st["plddt"][iface_res]))
        per_chain = {t: 0 for t in targets}
        for _, j in pairs:
            per_chain[chain_of[j]] += 1
        m["n_chains"] = float(sum(1 for t in targets if per_chain[t] > 0))
        if site:
            # Which binder residues provide that is deliberately not included.
            on_site = sum(1 for _, j in pairs if st["pos"][j] in site)
            m["site_frac"] = on_site / len(pairs)
        if do_bsa:
            m["bsa"] = buried_area(coords, st["radii"], brange, tgt_idx)

    if pae_path and os.path.exists(pae_path):
        with np.load(pae_path) as d:
            pae = d[d.files[0]]
        if pae.ndim == 3:
            pae = pae[0]
        if pae.shape[0] != n_res:
            print(f"  ! PAE has {pae.shape[0]} tokens, expected {n_res} - "
                  "skipped", file=sys.stderr)
        else:
            m["pae_block"] = float(np.concatenate([
                pae[slice(*brange), tgt_idx].ravel(),
                pae[tgt_idx, slice(*brange)].ravel()]).mean())
            if pairs:
                ii = np.array([p[0] for p in pairs])
                jj = np.array([p[1] for p in pairs])
                m["iface_pae"] = float(np.concatenate([pae[ii, jj], pae[jj, ii]]).mean())

    geo = {"targets": np.vstack([st["ca"][t] for t in targets if t in st["ca"]]),
           "binder": st["ca"].get(binder)}
    return m, (geo if geo["binder"] is not None and geo["targets"].size else None)


def seed_consistency(geos):
    """How similar are the seeds to each other? First superpose the targets,
    then compare the binder.

    shape_rmsd  shape of the binder, after free superposition
    placement   how far it has slipped in the target frame
    shift_along along the helix axis; for a periodic repeat on a
                lattice, offset is to be expected and not a defect
    """
    geos = [g for g in geos if g is not None]
    if len(geos) < 2:
        return {}
    if len({g["binder"].shape for g in geos}) > 1 or \
       len({g["targets"].shape for g in geos}) > 1:
        print("  ! Chain lengths vary between the seeds", file=sys.stderr)
        return {}

    ref = geos[0]["targets"]
    binders = []
    for g in geos:
        R, t = kabsch(g["targets"], ref)
        binders.append(g["binder"] @ R.T + t)

    place, shape, shift = [], [], []
    for a, b in itertools.combinations(range(len(binders)), 2):
        A, B = binders[a], binders[b]
        place.append(rmsd(A, B))
        R, t = kabsch(B, A)
        shape.append(rmsd(A, B @ R.T + t))
        axis = np.linalg.svd(A - A.mean(0), full_matrices=False)[2][0]
        shift.append(abs(float((B.mean(0) - A.mean(0)) @ axis)))
    return {"shape_rmsd": float(np.mean(shape)),
            "placement_rmsd": float(np.mean(place)),
            "placement_max": float(np.max(place)),
            "shift_along": float(np.mean(shift))}


def parse_site(spec):
    """'53-64' or '53-64,70,80-85' -> set of residue numbers."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Gate and strength axis for Boltz variants.")
    ap.add_argument("root", help="predictions folder from Boltz")
    ap.add_argument("--binder", default="F")
    ap.add_argument("--targets", default="A,B,C,D")
    ap.add_argument("--site", default="53-64",
                    help="intended surface on each target chain; leave empty "
                         "to skip checking site_frac")
    ap.add_argument("--cutoff", type=float, default=5.0,
                    help="heavy-atom distance for a contact")
    ap.add_argument("--max-shape", type=float, default=2.0)
    ap.add_argument("--min-site", type=float, default=0.7)
    ap.add_argument("--min-plddt", type=float, default=60.0)
    ap.add_argument("--all-chains", action="store_true", default=True)
    ap.add_argument("--no-all-chains", dest="all_chains", action="store_false")
    ap.add_argument("--no-bsa", action="store_true", help="faster, without bsa")
    ap.add_argument("--sort", default="bsa")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--prefix", default="",
                    help="prefix for all columns except the ID. Without it "
                         "site_frac and n_contacts collide with stage 8.")
    ap.add_argument("--id-col", default="variant")
    args = ap.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    site = parse_site(args.site) if args.site.strip() else set()
    csv_path = args.csv or os.path.join(args.root, "ranking.csv")

    vdirs = sorted(d for d in glob.glob(os.path.join(args.root, "*"))
                   if os.path.isdir(d)
                   and glob.glob(os.path.join(d, "confidence_*model_*.json")))
    if not vdirs:
        sys.exit(f"No variant folders with confidence_*.json under {args.root}")

    rows, n_ok, n_total = [], 0, 0
    for vdir in vdirs:
        name = os.path.basename(vdir)
        print(f"[{name}]", file=sys.stderr)
        samples, geos = [], []
        for cj in sorted(glob.glob(os.path.join(vdir, "confidence_*model_*.json"))):
            conf = json.load(open(cj))
            mi = re.search(r"model_(\d+)", cj)
            mi = mi.group(1) if mi else "0"
            model_file = None
            for ext in (".cif", ".pdb"):
                hits = [p for p in glob.glob(os.path.join(vdir, f"*model_{mi}{ext}"))
                        if not os.path.basename(p).startswith(
                            ("pae_", "pde_", "plddt_", "confidence_"))]
                if hits:
                    model_file = hits[0]
                    break
            npz = glob.glob(os.path.join(vdir, f"pae_*model_{mi}.npz"))
            m, geo = metrics_for_model(conf, model_file, npz[0] if npz else None,
                                       args.binder, targets, args.cutoff,
                                       site, not args.no_bsa)
            samples.append(m)
            geos.append(geo)
            n_total += 1
            n_ok += geo is not None

        r = {"variant": name, "n": len(samples)}
        for k in SEED_KEYS:
            x = np.array([s[k] for s in samples if s[k] is not None], float)
            x = x[np.isfinite(x)]
            r[k] = float(x.mean()) if len(x) else np.nan
            r[k + "_sd"] = float(x.std(ddof=1)) if len(x) > 1 else 0.0
        r.update(seed_consistency(geos))
        for k in VARIANT_KEYS:
            r.setdefault(k, np.nan)
        chains = [s["n_chains"] for s in samples if s["n_chains"] is not None]
        r["n_chains_min"] = float(min(chains)) if chains else np.nan
        rows.append(r)

    if n_ok == 0:
        sys.exit(f"In none of the {n_total} models was the binder chain "
                 f"'{args.binder}' readable. Check the chain names.")
    if n_ok < n_total:
        print(f"\nWarning: only {n_ok}/{n_total} models evaluable.", file=sys.stderr)

    # Gate for orientation
    for r in rows:
        why = []
        if not np.isfinite(r["shape_rmsd"]):
            why.append("shape n/a")
        elif r["shape_rmsd"] > args.max_shape:
            why.append(f"shape {r['shape_rmsd']:.2f}>{args.max_shape:g}")
        if args.all_chains:
            if not np.isfinite(r["n_chains_min"]):
                why.append("chains n/a")
            elif r["n_chains_min"] < len(targets):
                why.append(f"only {r['n_chains_min']:.0f}/{len(targets)} chains")
        if site:
            if not np.isfinite(r["site_frac"]):
                why.append("site n/a")
            elif r["site_frac"] < args.min_site:
                why.append(f"site {r['site_frac']:.2f}<{args.min_site:g}")
        if not np.isfinite(r["iface_plddt"]):
            why.append("pLDDT n/a")
        elif r["iface_plddt"] < args.min_plddt:
            why.append(f"pLDDT {r['iface_plddt']:.0f}<{args.min_plddt:g}")
        r["gate"] = "FAIL" if why else "PASS"
        r["gate_reason"] = "; ".join(why) or "-"

    key = args.sort
    if key not in SEED_KEYS + VARIANT_KEYS:
        sys.exit(f"--sort {key} unknown.")
    lower_better = key in ("shape_rmsd", "placement_rmsd", "placement_max",
                           "shift_along", "iface_pae", "pae_block")
    sign = 1 if lower_better else -1

    def sort_key(r):
        v = r[key]
        return (sign * v) if np.isfinite(v) else 1e18

    passed = sorted([r for r in rows if r["gate"] == "PASS"], key=sort_key)
    failed = sorted([r for r in rows if r["gate"] == "FAIL"], key=sort_key)

    def f(x, nd=2):
        return "  n/a" if not np.isfinite(x) else f"{x:.{nd}f}"

    print(f"\nGate: shape_rmsd <= {args.max_shape:g}"
          + (f", all {len(targets)} chains in every seed" if args.all_chains else "")
          + (f", site_frac >= {args.min_site:g}" if site else "")
          + f", iface_plddt >= {args.min_plddt:g}")
    print(f"Strength axis: {key}"
          f"{' (lower is stronger)' if lower_better else ' (higher is stronger)'}\n")

    hdr = (f'{"#":>3} {"variant":<24}{"Gate":>6}{"BSA":>8}{"shape":>7}{"site":>7}'
           f'{"Chn":>5}{"ifcPLDDT":>10}{"#Cont":>7}{"place":>7}{"ifc_PAE":>9}   Reason')
    print(hdr)
    print("-" * (len(hdr) + 14))
    for i, r in enumerate(passed + failed, 1):
        if i == len(passed) + 1 and failed and passed:
            print("-" * (len(hdr) + 14))
        print(f'{i:>3} {r["variant"]:<24}{r["gate"]:>6}{f(r["bsa"], 0):>8}'
              f'{f(r["shape_rmsd"]):>7}{f(r["site_frac"]):>7}'
              f'{f(r["n_chains_min"], 0):>5}{f(r["iface_plddt"], 1):>10}'
              f'{f(r["n_contacts"], 0):>7}{f(r["placement_rmsd"]):>7}'
              f'{f(r["iface_pae"]):>9}   {r["gate_reason"]}')

    print(f"\n{len(passed)} of {len(rows)} variants pass.")
    if passed:
        b = np.array([r["bsa"] for r in passed], float)
        b = b[np.isfinite(b)]
        if len(b):
            print(f"BSA range of the passers: {b.min():.0f} to {b.max():.0f} A^2. "
                  "Take the gradient for weaker and stronger "
                  "variants from it.")

    cols = (["variant", "n", "gate", "gate_reason"] + VARIANT_KEYS
            + [c for k in SEED_KEYS for c in (k, k + "_sd")])
    header = [args.id_col] + [args.prefix + c for c in cols[1:]]
    with open(csv_path, "w") as out:
        out.write(",".join(header) + "\n")
        for r in passed + failed:
            out.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])
                               for c in cols) + "\n")
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
