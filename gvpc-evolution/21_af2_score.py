#!/usr/bin/env python
"""
Compute part of stage 9. Started by 20_af2_run.py, not directly.

    $AF2_PY 21_af2_score.py job.yaml -o scores.csv

Every sequence is placed on the same complex backbone: the coordinates of the
binder stay, sequence and side chains are erased. Only the sequence varies.

Job YAML:
    params_dir: /path/to/alphafold_weights
    complex: templates/template.pdb
    binder_chain: F
    target_chains: [A, B, C, D]
    hotspots: [A53, A56, ...]        # numbering of this PDB file
    hotspot_cutoff: 6.0
    num_recycles: 3
    models: [0, 1]
    sequences:
      - {id: L1-0001, seq: VTDM...}
      - {id: L1-0002, seq: VTDF..., complex: poses/L1-0002.pdb}   # optional

Output per sequence:
    plddt           confidence of the binder in the complex, higher is better
    ptm / i_ptm     global and at the interface
    i_pae           uncertainty between binder and contacted
                    target residues, in Angstrom. Lower is better.
    hotspot_rmsd    displacement of only the epitope-near binder residues
    placement_rmsd  displacement of the whole binder in the target frame
    shape_rmsd      shape after free superposition

The three RMSDs answer different questions. A binder can tilt away at the edge
and still sit with its epitope part, or fit globally and miss the hotspots.

Models 0 and 1 are reported separately; the gate acts on model 1.
With templates colabdesign loads only two models, higher indices do not exist.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np
import yaml

# Without this JAX preallocates all memory and cuSolver silently returns NaN.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_UNIFIED_MEMORY", "0")

CONTACT_CUTOFF = 10.0

PER_MODEL_COLS = ["id", "model", "seq", "plddt", "ptm", "i_ptm", "i_pae",
                  "hotspot_rmsd", "placement_rmsd", "shape_rmsd",
                  "n_hotspot_res", "complex"]


def read_pdb(path):
    """Heavy atoms: (chain, resnum, is_CA, coordinates)."""
    ch, rs, ca, xyz = [], [], [], []
    for line in open(path):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        name = line[12:16].strip()
        if name.startswith("H") or line[76:78].strip() == "H":
            continue
        ch.append(line[21].strip() or "_")
        rs.append(int(line[22:26]))
        ca.append(name == "CA")
        xyz.append([float(line[30 + 8 * i:38 + 8 * i]) for i in range(3)])
    return (np.array(ch), np.array(rs), np.array(ca, dtype=bool),
            np.array(xyz, dtype=float))


def superpose(mobile, reference):
    """Rotation and translation that put mobile onto reference (Kabsch)."""
    mc, rc = mobile.mean(0), reference.mean(0)
    U, _, Vt = np.linalg.svd((mobile - mc).T @ (reference - rc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, mc, rc


def kabsch_rmsd(P, Q):
    R, pc, qc = superpose(P, Q)
    return float(np.sqrt((((P - pc) @ R.T - (Q - qc)) ** 2).sum(1).mean()))


def contact_mask(ch, rs, xyz, tchains, bchain):
    """Target residues that lie close to the binder in the input complex.

    Heavy-atom distance, not CA-CA: a residue whose side chain touches the binder
    while its CA sits 12 A away belongs to the interface.
    Without this mask i_pae averages over the whole target and saturates.
    """
    b = xyz[ch == bchain]
    mask = []
    for c in tchains:
        sel = ch == c
        for r in sorted(set(rs[sel].tolist())):
            idx = sel & (rs == r)
            d = np.linalg.norm(xyz[idx][:, None, :] - b[None, :, :], axis=2).min()
            mask.append(float(d) <= CONTACT_CUTOFF)
    return np.array(mask, dtype=bool)


def hotspot_mask(ch, rs, xyz, bchain, hotspots, cutoff):
    """Which binder residues lie close to the hotspots in the input complex."""
    if not hotspots:
        return None, []
    parsed = []
    for tok in hotspots:
        m = re.fullmatch(r"([A-Za-z0-9]+?)(\d+)", str(tok))
        if not m:
            raise SystemExit(f"Hotspot {tok!r} not readable, expected e.g. A53")
        parsed.append((m.group(1), int(m.group(2))))

    missing = [f"{c}{r}" for c, r in parsed if not ((ch == c) & (rs == r)).any()]
    if missing:
        raise SystemExit(
            f"Hotspots {missing} do not occur in the complex PDB. They "
            "refer to the numbering of this file; after a "
            "conversion it may have changed.")

    hxyz = np.concatenate([xyz[(ch == c) & (rs == r)] for c, r in parsed])
    bsel = ch == bchain
    near = []
    for r in sorted(set(rs[bsel].tolist())):
        idx = bsel & (rs == r)
        d = np.linalg.norm(xyz[idx][:, None, :] - hxyz[None, :, :], axis=2).min()
        near.append(bool(d <= cutoff))
    bres = sorted(set(rs[bsel].tolist()))
    return np.array(near, dtype=bool), [r for r, n in zip(bres, near) if n]


def restore_chains(model, tlens, blen):
    """colabdesign treats the target as ONE chain.

    With several subunits it is thereby presented as one long polypeptide
    and the interface does not form (i_ptm drops to
    about 0.1). Only the chain features are corrected here.
    """
    lens = list(tlens) + [blen]
    if sum(lens) != sum(model._lengths):
        raise SystemExit(f"Chain lengths {lens} do not match {model._lengths}")
    asym = np.concatenate([[i] * L for i, L in enumerate(lens)])
    entity = np.concatenate([[0] * L for L in tlens] + [[1] * blen])
    model._lengths = lens
    model._inputs.update(asym_id=asym, sym_id=asym, entity_id=entity)


def write_rows(path, rows, cols):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def aggregate(rows):
    """Per sequence: mean over the models plus each model individually."""
    by_id = defaultdict(list)
    for r in rows:
        by_id[r["id"]].append(r)
    keys = ("plddt", "ptm", "i_ptm", "i_pae", "hotspot_rmsd",
            "placement_rmsd", "shape_rmsd")
    out = []
    for sid, group in by_id.items():
        rec = {"id": sid, "seq": group[0]["seq"], "n_models": len(group),
               "complex": group[0]["complex"]}
        for k in keys:
            vals = [g[k] for g in group if g[k] is not None and np.isfinite(g[k])]
            rec[k] = round(float(np.mean(vals)), 4) if vals else float("nan")
            for g in group:
                rec[f"{k}_m{g['model']}"] = g[k]
        out.append(rec)
    cols = ["id", "seq", "n_models", "complex"]
    for k in keys:
        cols += [k] + sorted({c for r in out for c in r if c.startswith(k + "_m")})
    return out, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml")
    ap.add_argument("-o", "--out", default="af2_scores.csv")
    a = ap.parse_args()
    job = yaml.safe_load(open(a.yaml))

    bchain = job["binder_chain"]
    tchains = list(job["target_chains"])
    models = job.get("models", [0, 1])
    recycles = int(job.get("num_recycles", 3))
    hotspots = job.get("hotspots") or []
    hs_cut = float(job.get("hotspot_cutoff", 10.0))
    if any(m > 1 for m in models):
        raise SystemExit("models may only be 0 and 1: with templates colabdesign loads "
                         "only two models.")
    if not hotspots:
        print("Warning: no hotspots, hotspot_rmsd stays empty.", file=sys.stderr)

    # Group by complex. In round 1 all sequences share the same
    # backbone, so the expensive preparation runs once instead of a hundred times.
    groups = defaultdict(list)
    for entry in job["sequences"]:
        groups[entry.get("complex", job["complex"])].append(entry)
    n_seq = sum(len(v) for v in groups.values())
    print(f"{n_seq} sequences in {len(groups)} complex group(s)")

    from colabdesign import clear_mem, mk_afdesign_model

    rows = []
    per_model_path = a.out.replace(".csv", ".per_model.csv")

    for pdb, entries in groups.items():
        ch, rs, is_ca, xyz = read_pdb(pdb)
        tlens = [int((is_ca & (ch == c)).sum()) for c in tchains]
        blen = int((is_ca & (ch == bchain)).sum())
        if blen == 0:
            raise SystemExit(f"{pdb}: chain {bchain} not found "
                             f"(present: {sorted(set(ch.tolist()))})")
        tmask = contact_mask(ch, rs, xyz, tchains, bchain)
        hmask, hres = hotspot_mask(ch, rs, xyz, bchain, hotspots, hs_cut)
        ref_b = xyz[is_ca & (ch == bchain)]
        ref_t = np.concatenate([xyz[is_ca & (ch == c)] for c in tchains])

        print(f"\n[{os.path.basename(pdb)}] {len(entries)} sequences, "
              f"target {tlens} + binder {blen}")
        if hmask is not None:
            print(f"  epitope-near binder residues (<= {hs_cut} A): {len(hres)} of {blen}")
            if not hres:
                print("  ! No binder residue near a hotspot. Either the pose is "
                      "wrong or the numbering.", file=sys.stderr)
            elif len(hres) >= 0.8 * blen:
                print(f"  ! Almost the whole helix counts as epitope-near. "
                      f"hotspot_rmsd then measures the same as placement_rmsd; "
                      f"set hotspot_cutoff smaller.", file=sys.stderr)

        clear_mem()
        model = mk_afdesign_model(protocol="binder", use_multimer=False,
                                  use_templates=True, data_dir=job["params_dir"])
        model.prep_inputs(pdb_filename=pdb, target_chain=",".join(tchains),
                          binder_chain=bchain, rm_binder=False,
                          rm_binder_seq=True, rm_binder_sc=True)
        restore_chains(model, tlens, blen)

        for entry in entries:
            seq, sid = entry["seq"], entry["id"]
            if blen != len(seq):
                raise SystemExit(f"{sid}: chain {bchain} has {blen} residues, "
                                 f"the sequence {len(seq)}")
            for mi in models:
                model.predict(seq=seq, num_recycles=recycles, models=[mi],
                              verbose=False)
                log = model.aux["log"]
                pae = np.asarray(model.aux["pae"], dtype=float)
                pae = (pae + pae.T) / 2.0
                tL = sum(tlens)
                i_pae = (float(pae[tL:tL + blen, :tL][:, tmask].mean())
                         if tmask.any() else np.nan)

                # AF2 writes the target first, then the binder. Put the prediction into
                # the frame of the input target, then measure how far
                # the binder has moved.
                p = np.asarray(model.aux["atom_positions"])[:, 1]      # CA
                pt, pb = p[:tL], p[tL:tL + blen]
                R, pc, qc = superpose(pt, ref_t)
                moved = (pb - pc) @ R.T + qc

                placement = float(np.sqrt(((moved - ref_b) ** 2).sum(1).mean()))
                hot = (float(np.sqrt(((moved[hmask] - ref_b[hmask]) ** 2).sum(1).mean()))
                       if hmask is not None and hmask.any() else np.nan)

                rows.append(dict(
                    id=sid, model=mi, seq=seq, complex=pdb,
                    plddt=round(float(log.get("plddt", np.nan)), 4),
                    ptm=round(float(log.get("ptm", np.nan)), 4),
                    i_ptm=round(float(log.get("i_ptm", np.nan)), 4),
                    i_pae=round(i_pae, 2),
                    hotspot_rmsd=round(hot, 2) if np.isfinite(hot) else np.nan,
                    placement_rmsd=round(placement, 2),
                    shape_rmsd=round(kabsch_rmsd(pb, ref_b), 2),
                    n_hotspot_res=int(hmask.sum()) if hmask is not None else 0))
                r = rows[-1]
                print(f"  {sid} m{mi}: i_pae {i_pae:6.2f}  i_ptm {r['i_ptm']:.3f}  "
                      f"plddt {r['plddt']:.3f}  hotRMSD {r['hotspot_rmsd']}",
                      flush=True)

        # Intermediate result after each group, so that an abort does not cost everything.
        write_rows(per_model_path, rows, PER_MODEL_COLS)
        out, cols = aggregate(rows)
        write_rows(a.out, out, cols)

    out, cols = aggregate(rows)
    write_rows(a.out, out, cols)
    write_rows(per_model_path, rows, PER_MODEL_COLS)
    print(f"\n{len(out)} sequences -> {a.out} (plus .per_model.csv)")


if __name__ == "__main__":
    main()
