#!/usr/bin/env python
"""Convert RFD3 output to PDB and triage it before a single sequence is designed.

Why triage exists: `infer_ori_strategy=hotspots` PLACES the binder near the
hotspot centroid, it does not GUARANTEE the binder touches the hotspots, and it
certainly does not guarantee the contacts land on the focus subunit rather than
sliding onto a flanking one. Every backbone that survives here costs 8 MPNN
sequences and 8 AF2 predictions downstream, so a cheap CPU pass that discards
off-epitope backbones is the highest-leverage filter in the pipeline.

Nothing here is a quality judgement about the binder -- it is purely "is this
backbone even addressing the epitope we asked for". Design quality starts at
AF2.

    python scripts/collect_backbones.py [--config ...] [--limit N]

Writes <run>/pdb/<cell>/<design>.pdb and <run>/backbones.csv.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def arm_triage(cfg, arm):
    """Global triage settings with this arm's overrides applied.

    The two arms want opposite things from the same check: `span` is supposed
    to spread its contacts across subunits, `focal` is supposed to concentrate
    them on one. A single global threshold cannot express both.
    """
    tri = dict(cfg["triage"])
    tri.update(cfg["arms"][arm].get("triage") or {})
    return tri


def triage_one(st, cfg, hotspots, tri):
    """Geometry summary + pass/fail for one backbone. Returns (dict, [reasons])."""
    t = cfg["target"]
    bch = t["binder_chain"]
    tch = list(t["chain_map"].values())
    focus = t["chain_map"][t["focus_chain"]]

    binder_xyz = st.xyz[st.chain == bch]
    row = {"binder_len": len(st.ca(bch))}

    # per-hotspot closest approach, in RFD3-OUTPUT chain and residue numbering
    n_close, hot_chains = 0, set()
    for hs in hotspots:
        och, rs = common.hotspot_out_chain(cfg, hs)
        d = common.min_dist_to_residue(st, binder_xyz, och, rs)
        row[f"d_{hs}"] = round(d, 2)
        if d <= tri["hotspot_contact_cutoff"]:
            n_close += 1
            hot_chains.add(och)
    row["n_hotspots_contacted"] = n_close
    row["n_hotspot_chains"] = len(hot_chains)

    contacts = common.contact_residues(st, bch, tch, cutoff=4.0)
    row["n_interface_res"] = len(contacts)
    n_focus = sum(1 for (c, _) in contacts if c == focus)
    row["focus_contact_fraction"] = round(n_focus / len(contacts), 3) if contacts else 0.0
    row["n_target_chains_contacted"] = len({c for c, _ in contacts})
    row["contact_fingerprint"] = ";".join(f"{c}{r}" for c, r in sorted(contacts))

    ss = common.secondary_structure(st.ca(bch))
    row.update({f"binder_{k}_frac": round(v, 3) for k, v in ss.items()})

    reasons = []
    if n_close < tri["min_hotspots_contacted"]:
        reasons.append(f"hotspots({n_close}<{tri['min_hotspots_contacted']})")
    if len(hot_chains) < tri.get("min_chains_contacted", 1):
        reasons.append(f"subunits({len(hot_chains)}<{tri['min_chains_contacted']})")
    if row["n_interface_res"] < tri["min_interface_residues"]:
        reasons.append(f"iface({row['n_interface_res']}<{tri['min_interface_residues']})")
    if tri.get("min_focus_chain_contact_fraction", 0) > 0 and \
            row["focus_contact_fraction"] < tri["min_focus_chain_contact_fraction"]:
        reasons.append(f"offfocus({row['focus_contact_fraction']:.2f})")
    if ss["loop"] > tri["max_loop_fraction"]:
        reasons.append(f"loopy({ss['loop']:.2f})")
    return row, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=0, help="stop after N backbones (smoke test)")
    a = ap.parse_args()

    cfg = common.load_config(a.config)
    root = Path(cfg["run_root"]) / cfg["campaign"]
    cells = json.loads((root / "cells.json").read_text())
    tri_default = cfg["triage"]
    out_pdb = root / "pdb"

    rows, n_seen = [], 0
    cap = int(tri_default.get("max_backbones_per_cell", 0) or 0)

    for cell, meta in sorted(cells.items()):
        tri = arm_triage(cfg, meta["arm"])
        cdir = root / "backbones" / cell
        if not cdir.is_dir():
            continue

        cell_rows = []
        for cif in sorted(cdir.glob("*.cif.gz")):
            design = cif.name[: -len(".cif.gz")]
            try:
                st = common.read_structure(cif)
            except Exception as e:              # never let one bad file stop the sweep
                cell_rows.append(({"design": design, "cell": cell, "pass": 0,
                                   "fail_reasons": f"parse:{type(e).__name__}"}, None))
                continue

            # RFD3 systematically misplaces the first residue of the diffused
            # chain (see common.trim_terminal_defects). Trim it before anything
            # measures or designs on this backbone.
            st, n_start, n_end = common.trim_terminal_defects(
                st, cfg["target"]["binder_chain"])
            interior, _ = common.ca_bond_defects(st, cfg["target"]["binder_chain"])

            row, reasons = triage_one(st, cfg, meta["hotspots"], tri)
            row["binder_trimmed"] = n_start + n_end
            row["interior_ca_defects"] = int(len(interior))
            if len(interior) > tri["max_chainbreaks"]:
                reasons.append(f"cabreak({len(interior)})")
            row.update({"design": design, "cell": cell, "arm": meta["arm"],
                        "length_bin": meta["length_bin"],
                        "step_scale": meta.get("step_scale"),
                        "gamma_0": meta.get("gamma_0"),
                        "is_non_loopy": meta.get("is_non_loopy")})

            mj = cif.with_suffix("").with_suffix(".json")
            if mj.exists():
                m = json.loads(mj.read_text()).get("metrics", {})
                cb = int(m.get("n_chainbreaks", 0))
                cl = int(m.get("n_clashing.interresidue_clashes_w_sidechain", 0)) + \
                     int(m.get("n_clashing.interresidue_clashes_w_backbone", 0))
                row.update({"rfd3_chainbreaks": cb, "rfd3_clashes": cl,
                            "rfd3_rg": round(float(m.get("radius_of_gyration", float("nan"))), 2)})
                # NOTE: rfd3_chainbreaks is reported, not gated. It counts the
                # terminal artifact that trimming just removed; gating on it
                # discarded 62% of good backbones. `interior_ca_defects` above
                # is the real check.
                if cl > tri["max_clashes"]:
                    reasons.append(f"clash({cl})")

            row["fail_reasons"] = ",".join(reasons)
            row["pass"] = int(not reasons)
            cell_rows.append((row, cif if not reasons else None))

            n_seen += 1
            if a.limit and n_seen >= a.limit:
                break

        # Per-cell cap. Generation is ~50x cheaper per design than scoring one,
        # so the campaign generates far more than it can afford to score and
        # keeps only the best of each cell. Ranked by how much of the epitope
        # the backbone actually engages -- capping at random would throw away
        # the reason for generating widely in the first place.
        kept = [(r, c) for r, c in cell_rows if r.get("pass")]
        if cap and len(kept) > cap:
            kept.sort(key=lambda rc: (-int(rc[0].get("n_hotspots_contacted", 0)),
                                      -int(rc[0].get("n_interface_res", 0))))
            for r, _ in kept[cap:]:
                r["pass"] = 0
                r["fail_reasons"] = (r["fail_reasons"] + "," if r["fail_reasons"] else "") + "cellcap"
            kept = kept[:cap]

        for r, cif in kept:
            st = common.read_structure(cif)
            st, _, _ = common.trim_terminal_defects(st, cfg["target"]["binder_chain"])
            q = out_pdb / cell / f"{r['design']}.pdb"
            common.write_pdb(st, q, chain_order=[cfg["target"]["binder_chain"]]
                             + list(cfg["target"]["chain_map"].values()))
            r["pdb"] = str(q)

        rows.extend(r for r, _ in cell_rows)
        if a.limit and n_seen >= a.limit:
            break

    if not rows:
        raise SystemExit(f"no backbones found under {root/'backbones'} -- run stage 01 first")

    cols = ["design", "cell", "arm", "length_bin", "step_scale", "gamma_0", "binder_len",
            "binder_trimmed", "interior_ca_defects",
            "n_hotspots_contacted", "n_hotspot_chains", "n_target_chains_contacted",
            "n_interface_res", "focus_contact_fraction",
            "binder_helix_frac", "binder_sheet_frac", "binder_loop_frac",
            "rfd3_chainbreaks", "rfd3_clashes", "rfd3_rg",
            "pass", "fail_reasons", "pdb", "contact_fingerprint"]
    extra = sorted({k for r in rows for k in r} - set(cols))
    cols = cols[:11] + extra + cols[11:]
    out = root / "backbones.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n_pass = sum(r.get("pass", 0) for r in rows)
    n_cap = sum(1 for r in rows if "cellcap" in (r.get("fail_reasons") or ""))
    print(f"backbones seen : {len(rows)}")
    print(f"passed triage  : {n_pass + n_cap}  ({(n_pass+n_cap)/len(rows):.0%})")
    if n_cap:
        print(f"held back      : {n_cap} by the per-cell cap ({cap}/cell)")
    print(f"forwarded      : {n_pass}")
    from collections import Counter
    fails = Counter(x for r in rows if not r.get("pass")
                    for x in (r.get("fail_reasons") or "").split(",") if x)
    for k, v in fails.most_common():
        print(f"   fail {k.split('(')[0]:<12} {v}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
