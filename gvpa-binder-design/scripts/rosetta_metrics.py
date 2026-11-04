#!/usr/bin/env python
"""BindCraft's PyRosetta interface metrics, on the AF2-predicted complexes.

Derived from BindCraft (MIT, Copyright (c) 2024 Martin Pacesa); see NOTICE.

Runs in the BindCraft env, which has PyRosetta plus the DAlphaBall binary and
dssp that BindCraft's own filters depend on.

This is a PARAMETERISED re-implementation of
`Bindcraft/functions/pyrosetta_utils.py::score_interface`, not a rewrite. Every
metric, every mover setting and every XML filter string is taken from that
function unchanged, so the numbers are directly comparable to BindCraft's and
its default thresholds mean the same thing here.

The one necessary change: BindCraft hardcodes `iam.set_interface("A_B")`, a
two-chain complex. This target is a three-subunit GvpA rib, so the interface is
specified as `<all target chains>_<binder>`, and `hotspot_residues` -- which
also hardcodes chain A as the target -- is generalised the same way.

Scoring the AF2 model rather than the RFD3 backbone is deliberate: RFD3 emits a
placeholder side-chain packing for a sequence MPNN has not written yet, so
Rosetta energies on it would describe a protein nobody is going to order.

    python scripts/rosetta_metrics.py --scores <af2_full/scores.csv> \
        --out <run>/rosetta.csv [--model 0] [--shard i --n-shards N] [--relax]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

BINDCRAFT = Path(os.environ.get("BINDCRAFT_DIR", Path.home() / "Bindcraft"))


def init_pyrosetta():
    import pyrosetta as pr
    dab = BINDCRAFT / "functions" / "DAlphaBall.gcc"
    if not dab.exists():
        raise SystemExit(f"DAlphaBall.gcc not found at {dab} -- packstat and the "
                         f"BuriedUnsatHbonds filter cannot run without it. "
                         f"Set BINDCRAFT_DIR.")
    # Flags copied verbatim from Bindcraft/bindcraft.py so the score function,
    # the beta_nov16 corrections and the relax repeat count all match.
    pr.init(f"-ignore_unrecognized_res -ignore_zero_occupancy -mute all "
            f"-holes:dalphaball {dab} -corrections::beta_nov16 true "
            f"-relax:default_repeats 1")
    return pr


def identify_chains(pdb_file, binder_len):
    """Find the binder chain in a PREDICTED complex, by length.

    Chain labels are NOT stable across the pipeline. RFD3 writes the binder as
    chain A followed by the target; colabdesign writes the TARGET FIRST and the
    binder LAST (here: A,B,C = 3x55 GvpA, D = binder). Using the RFD3
    convention on an AF2 output silently scored a GvpA subunit as the binder --
    surface_hydrophobicity came back ~0.65 and nearly constant because it was
    measuring GvpA's own amyloid surface, and interface_dG ~ -100 REU because
    it was measuring a GvpA-GvpA lattice interface.

    Length is the reliable discriminator. Ties resolve to the LAST match,
    matching colabdesign's ordering.
    """
    st = common.read_structure(pdb_file)
    order = list(dict.fromkeys(str(c) for c in st.chain))
    cands = [c for c in order if len(st.ca(c)) == int(binder_len)]
    if not cands:
        raise RuntimeError(
            f"no chain of length {binder_len} in {pdb_file}; "
            f"chains are {[(c, len(st.ca(c))) for c in order]}")
    binder = cands[-1]
    return binder, [c for c in order if c != binder]


def score_interface(pr, pdb_file, binder_chain, target_chains, workdir=None):
    """BindCraft's score_interface, on a two-chain rewrite of the complex.

    The pose is first collapsed to target=A / binder=B (see
    common.write_two_chain_pdb). That is not cosmetic: Rosetta's
    BuriedUnsatHbonds filter with use_ddG_style="true" -- the source of
    BindCraft's n_InterfaceUnsatHbonds -- hard-fails on poses with more than
    three chains, and binder + three GvpA subunits is four. Collapsing the
    target restores exactly the two-chain shape BindCraft scores, so
    set_interface("A_B") below is BindCraft's own call unchanged.
    """
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
    from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

    src = common.read_structure(pdb_file)
    wd = Path(workdir or Path(pdb_file).parent)
    wd.mkdir(parents=True, exist_ok=True)
    flat = wd / (Path(pdb_file).stem + ".2chain.pdb")
    common.write_two_chain_pdb(src, flat, binder_chain, target_chains)
    binder_chain, target_chains = "B", ["A"]
    pdb_file = str(flat)

    pose = pr.pose_from_pdb(pdb_file)
    if pose.num_chains() != 2:
        raise RuntimeError(f"expected a 2-chain pose, got {pose.num_chains()} "
                           f"from {pdb_file}")

    iam = InterfaceAnalyzerMover()
    iam.set_interface("A_B")
    scorefxn = pr.get_fa_scorefxn()
    iam.set_scorefunction(scorefxn)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.set_calc_hbond_sasaE(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    st = common.read_structure(pdb_file)
    contacts = common.contact_residues(st, binder_chain, target_chains, cutoff=4.0)
    # interface_AA counts BINDER residues, so it is recomputed from the binder
    # side rather than from the target-side contact map.
    binder_iface = {}
    from scipy.spatial import cKDTree
    import numpy as np
    bmask = st.chain == binder_chain
    tmask = np.isin(st.chain, list(target_chains))
    tree = cKDTree(st.xyz[tmask])
    bi = np.where(bmask)[0]
    for k, close in zip(bi, tree.query_ball_point(st.xyz[bmask], 4.0)):
        if close:
            binder_iface[int(st.resseq[k])] = common.THREE_TO_ONE.get(str(st.resname[k]), "X")

    interface_AA = {aa: 0 for aa in "ACDEFGHIKLMNPQRSTVWY"}
    for aa in binder_iface.values():
        if aa in interface_AA:
            interface_AA[aa] += 1
    nres = len(binder_iface)
    hydrophobic = set("ACFILMPVWY")
    iface_hydro = (sum(interface_AA[a] for a in hydrophobic) / nres * 100) if nres else 0

    data = iam.get_all_data()
    dG = iam.get_interface_dG()
    dSASA = iam.get_interface_delta_sasa()
    buns = XmlObjects.static_get_filter(
        '<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" '
        'ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="1" '
        'probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />').report_sm(pose)

    sel = ChainSelector(binder_chain)
    tem = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
    tem.set_scorefunction(scorefxn); tem.set_residue_selector(sel)
    binder_score = tem.calculate(pose)
    bs = pr.rosetta.core.simple_metrics.metrics.SasaMetric()
    bs.set_residue_selector(sel)
    binder_sasa = bs.calculate(pose)

    # surface hydrophobicity, on the isolated binder, as BindCraft does it
    chains = {pose.pdb_info().chain(pose.conformation().chain_begin(i)): p
              for i, p in zip(range(1, pose.num_chains() + 1), pose.split_by_chain())}
    bp = chains[binder_chain]
    layer = pr.rosetta.core.select.residue_selector.LayerSelector()
    layer.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
    surf = layer.apply(bp)
    apol = tot = 0
    for i in range(1, len(surf) + 1):
        if surf[i]:
            res = bp.residue(i)
            if res.is_apolar() or res.name() in ("PHE", "TRP", "TYR"):
                apol += 1
            tot += 1

    out = {
        "binder_score": binder_score,
        "surface_hydrophobicity": apol / tot if tot else float("nan"),
        "interface_sc": data.sc_value,
        "interface_packstat": iam.get_interface_packstat(),
        "interface_dG": dG,
        "interface_dSASA": dSASA,
        "interface_dG_SASA_ratio": data.dG_dSASA_ratio * 100,
        "interface_fraction": (dSASA / binder_sasa * 100) if binder_sasa > 0 else 0,
        "interface_hydrophobicity": iface_hydro,
        "interface_nres": nres,
        "interface_hbonds": data.interface_hbonds,
        "interface_unsat_hbonds": buns,
        "interface_hbond_pct": (data.interface_hbonds / nres * 100) if nres else None,
        "interface_unsat_pct": (buns / nres * 100) if nres else None,
        "interface_residues": ",".join(f"{binder_chain}{r}" for r in sorted(binder_iface)),
        "n_target_contacts": len(contacts),
    }
    out.update({f"iAA_{k}": v for k, v in interface_AA.items()})
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--scores", required=True, help="af2_full scores.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--relaxed-dir", default=None,
                    help="where FastRelax poses go (default: <out dir>/relaxed)")
    ap.add_argument("--model", type=int, default=0, help="which AF2 model's complex to score")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    # BindCraft scores the RELAXED complex -- bindcraft.py calls
    # score_interface(mpnn_design_relaxed, ...), not the raw prediction. Its
    # dG / ShapeComplementarity / unsat thresholds are calibrated on relaxed
    # structures, so scoring unrelaxed output against them would reject almost
    # everything for the wrong reason. Relax is therefore ON by default.
    #
    # Relax runs on the ORIGINAL multi-chain pose, before the two-chain
    # rewrite, so FastRelax never sees the artificial junctions that rewrite
    # introduces and cannot distort the target trying to repair them.
    ap.add_argument("--resume", action="store_true",
                    help="skip ids already present in --out")
    ap.add_argument("--no-relax", dest="relax", action="store_false",
                    help="skip FastRelax (faster, but NOT BindCraft-comparable)")
    ap.set_defaults(relax=True)
    a = ap.parse_args()

    cfg = common.load_config(a.config)

    with open(a.scores) as fh:
        rows = [r for r in csv.DictReader(fh) if r.get(f"m{a.model}_complex_pdb")]
    rows.sort(key=lambda r: r["id"])
    mine = rows[a.shard :: a.n_shards]

    # Resume: FastRelax is the slowest stage in the pipeline, so a session that
    # ends mid-stage must not throw the finished work away.
    done_rows = []
    if a.resume and Path(a.out).exists():
        with open(a.out) as fh:
            done_rows = [r for r in csv.DictReader(fh)
                         if r.get("interface_dG") not in ("", None)]
        have = {r["id"] for r in done_rows}
        before = len(mine)
        mine = [r for r in mine if r["id"] not in have]
        if before != len(mine):
            print(f"resume: {before - len(mine)} already scored, {len(mine)} left",
                  flush=True)
    if not mine:
        print("nothing to score")
        if done_rows:
            _write(done_rows, a.out)
        return

    pr = init_pyrosetta()
    from pyrosetta.rosetta.protocols.relax import FastRelax

    out_rows = list(done_rows)
    for i, r in enumerate(mine, 1):
        pdb = r[f"m{a.model}_complex_pdb"]
        rec = {"id": r["id"], "design": r["design"], "scored_pdb": pdb}
        wd = Path(a.out).parent / "poses"
        # Chain labels differ between RFD3 output and AF2 output; never assume.
        bchain, tchains = identify_chains(pdb, r["binder_len"])
        rec["binder_chain_used"] = bchain
        try:
            if a.relax:
                pose = pr.pose_from_pdb(pdb)
                fr = FastRelax(); fr.set_scorefxn(pr.get_fa_scorefxn()); fr.apply(pose)
                # Relaxed poses go in their own directory: at ~0.4 MB each
                # they otherwise bury the run root under thousands of files.
                _rd = Path(a.relaxed_dir) if a.relaxed_dir else Path(a.out).parent / "relaxed"
                _rd.mkdir(parents=True, exist_ok=True)
                pdb = str(_rd / f"{r['id']}_relaxed.pdb")
                pose.dump_pdb(pdb); rec["scored_pdb"] = pdb
            rec.update(score_interface(pr, pdb, bchain, tchains, workdir=wd))
            # sc == 0.0 exactly means Rosetta's shape-complementarity
            # calculation FAILED, not that the interface is flat. Left
            # unflagged it would fail every design against the 0.60 threshold
            # and look like a design problem rather than a scoring one.
            if rec.get("interface_sc") == 0.0:
                print(f"WARN {r['id']}: interface_sc == 0.0 (sc calculation "
                      f"failed, not a flat interface)", flush=True)
        except Exception as e:
            print(f"ROSETTA-ERR {r['id']}: {type(e).__name__}: {e}", flush=True)
            rec["error"] = type(e).__name__
        out_rows.append(rec)
        if i % 20 == 0:
            _write(out_rows, a.out); print(f"  {i}/{len(mine)}", flush=True)

    _write(out_rows, a.out)
    print(f"wrote {len(out_rows)} rows -> {a.out}")


def _write(rows, path):
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); cols.append(k)
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
