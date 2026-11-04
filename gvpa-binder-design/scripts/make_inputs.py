#!/usr/bin/env python
"""Expand the campaign config into RFD3 input JSONs plus a job table.

The campaign is a grid: arm x length-bin x sampler setting. That grid IS the
generation-side diversity strategy -- rather than drawing 2000 backbones from
one distribution and hoping they differ, the budget is spread over cells that
are known a priori to produce different geometry (different epitope face,
different binder size, different sampler temperature). Selection-side
diversity (scripts/diversity.py) then works on top of that.

One cell == one RFD3 invocation == one job-table row, so a cell that fails
or needs more sampling can be re-run alone.

    python scripts/make_inputs.py [--config configs/gvpa_serratia.yaml] [--force]

Writes under <run_root>/<campaign>/:
    specs/<cell>.json   RFD3 input, one task key "d"
    cells.json          cell -> {arm, length_bin, step_scale, gamma_0, ...}
    jobs.tsv            one line per array task, read by run_pipeline.sh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


#: Short tags for sweep axes, so a cell name stays readable.
AXIS_TAG = {"step_scale": "s", "gamma_0": "g", "is_non_loopy": "nl",
            "inference_noise_scaling_factor": "ns"}


def _tag(v):
    if v is None:
        return "any"
    if isinstance(v, bool):
        return "T" if v else "F"
    return f"{v:g}".replace(".", "p") if isinstance(v, (int, float)) else str(v)


def cell_name(arm, lo, hi, combo):
    parts = "_".join(f"{AXIS_TAG.get(k, k)}{_tag(v)}" for k, v in combo.items())
    return f"{arm}_L{lo}-{hi}_{parts}"


def sweep_combos(cfg):
    """Cartesian product over every axis in `sweep`.

    Generic rather than hardcoded to (step_scale, gamma_0): topology turned out
    to need its own axis, and it will not be the last one.
    """
    import itertools
    sw = cfg["sweep"]
    keys = list(sw)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(sw[k] for k in keys))]


def build(cfg, force=False):
    root = Path(cfg["run_root"]) / cfg["campaign"]
    specs = root / "specs"
    specs.mkdir(parents=True, exist_ok=True)

    tgt = cfg["target"]
    cif = common.resolve(cfg, tgt["cif"])
    st = common.read_structure(cif)

    # Contig target segments come from the file, not from the config, so a
    # re-cleaned target with different numbering cannot silently desync.
    segs = []
    for ch in tgt["design_chains"]:
        rs = st.resseq[st.chain == ch]
        if len(rs) == 0:
            raise SystemExit(f"chain {ch} absent from {cif}")
        segs.append(f"{ch}{rs.min()}-{rs.max()}")

    rf = cfg["rfd3"]
    bins = [tuple(int(x) for x in b.split("-")) for b in cfg["length_bins"]]
    sweep = sweep_combos(cfg)

    n_cells_per_arm = len(bins) * len(sweep)
    cells, jobs = {}, []
    total = 0
    for arm, spec in cfg["arms"].items():
        budget = cfg["rfd3"]["n_backbones"] * float(spec["weight"])
        per_cell = budget / (n_cells_per_arm * max(1, len(spec.get("ori_tokens") or [None])))
        n_batches = max(1, round(per_cell / rf["diffusion_batch_size"]))
        # An arm may pin the binder's origin explicitly instead of inferring it
        # from the hotspot centroid. Each entry is a separate binding REGISTER
        # along the rib, so the arm expands into one cell set per origin.
        oris = spec.get("ori_tokens") or [None]
        for lo, hi in bins:
          for oi, ori in enumerate(oris):
            for combo in sweep:
                name = cell_name(arm, lo, hi, combo)
                if ori is not None:
                    name += "_ori" + str(spec.get("ori_names", [])[oi]
                                         if spec.get("ori_names") else oi)
                contig = f"{lo}-{hi},/0," + ",/0,".join(segs)
                d = {
                    "dialect": rf["dialect"],
                    "input": cif,
                    "contig": contig,
                    "select_hotspots": dict(spec["hotspots"]),
                }
                if ori is None:
                    d["infer_ori_strategy"] = rf["infer_ori_strategy"]
                else:
                    # ori_token and infer_ori_strategy are alternatives; setting
                    # both would let the inferred centroid override the register.
                    d["ori_token"] = [float(x) for x in ori]
                # A sweep value of null means UNCONDITIONED -- the key is
                # omitted so RFD3 uses no conditioning at all, which is a
                # genuinely different regime from conditioning it to False.
                if combo.get("is_non_loopy") is not None:
                    d["is_non_loopy"] = bool(combo["is_non_loopy"])
                # Atom-level H-bond conditioning, when the arm asks for it.
                # These map to RFD3's active_donor / active_acceptor features
                # and tell the model which target polar atoms a design must
                # satisfy -- the direct lever on a charged epitope, where
                # BindCraft's dG and buried-unsat criteria do the killing.
                for key in ("select_hbond_donor", "select_hbond_acceptor",
                            "select_buried", "select_partially_buried",
                            "select_exposed"):
                    if spec.get(key):
                        d[key] = dict(spec[key])
                p = specs / f"{name}.json"
                if force or not p.exists():
                    p.write_text(json.dumps({"d": d}, indent=2))
                made = n_batches * rf["diffusion_batch_size"]
                total += made
                cells[name] = {
                    "arm": arm, "length_bin": f"{lo}-{hi}", "len_lo": lo, "len_hi": hi,
                    "n_batches": n_batches, "expected_backbones": made,
                    "hotspots": list(spec["hotspots"].keys()), "spec": str(p),
                    **{k: v for k, v in combo.items()},
                }
                jobs.append((name, str(p),
                             combo.get("step_scale", rf.get("step_scale", 3.0)),
                             combo.get("gamma_0", rf.get("gamma_0", 0.2)),
                             n_batches))

    (root / "cells.json").write_text(json.dumps(cells, indent=2))
    with open(root / "jobs.tsv", "w") as fh:
        fh.write("cell\tspec\tstep_scale\tgamma_0\tn_batches\n")
        for j in jobs:
            fh.write("\t".join(str(x) for x in j) + "\n")

    print(f"campaign      : {cfg['campaign']}")
    print(f"run root      : {root}")
    print(f"cells / tasks : {len(jobs)}  ({len(cfg['arms'])} arms x {len(bins)} bins "
          f"x {len(sweep)} sampler settings)")
    print(f"sweep axes    : " + ", ".join(f"{k}={v}" for k, v in cfg["sweep"].items()))
    for arm in cfg["arms"]:
        n = sum(c["expected_backbones"] for c in cells.values() if c["arm"] == arm)
        print(f"  arm {arm:<8} {n:5d} backbones  hotspots={cfg['arms'][arm]['hotspots']}")
    print(f"expected      : {total} backbones (requested {rf['n_backbones']})")
    return root


def rf_list(cfg, key):
    return cfg["sweep"][key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="rewrite existing spec JSONs")
    a = ap.parse_args()
    build(common.load_config(a.config), a.force)


if __name__ == "__main__":
    main()
