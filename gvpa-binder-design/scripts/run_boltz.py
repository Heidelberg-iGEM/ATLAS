#!/usr/bin/env python
"""Boltz-2 scoring: an orthogonal second opinion, never a gate.

Boltz sees a SEQUENCE, not the designed backbone, so when it agrees with AF2 it
agrees for independent reasons -- that is the entire value. AF2 initial-guess is
seeded with the design and answers "does this sequence hold THIS structure";
Boltz answers "can a structure be found at all".

Boltz deliberately does NOT gate acceptance here, and does not enter the
composite score. In this group's earlier work on a different target, Boltz
ipTM / ligand-ipTM / affinity scored composition-matched scrambles as well as
real designs (z ~ 0), while AF2 initial-guess separated them at AUC 0.99. Until
that is re-established on THIS target -- which scripts/null_control.py measures
directly -- Boltz is reported and used for tie-breaks, nothing more.

Single-sequence by default (`msa: empty`). An MSA would genuinely help Boltz
fold GvpA, but compute nodes have no network for the MSA server and Boltz is
non-gating, so the cost is not worth the complication. Set
boltz.use_msa_server: true in the config to change that.

    python scripts/run_boltz.py --accepted <run>/accepted.csv --out <run>/boltz.csv \
        [--shard i --n-shards N]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

NAN = float("nan")


def boltz_binary():
    for c in [os.environ.get("BOLTZ_BIN"),
              str(Path.home() / ".conda/envs/boltz/bin/boltz")]:
        if c and Path(c).exists():
            return c
    raise SystemExit("boltz executable not found; set BOLTZ_BIN")


def write_job(path, chains, use_msa, affinity_binder=None, template=None,
              pocket=None):
    lines = ["version: 1", "sequences:"]
    for cid, seq in chains.items():
        lines += ["  - protein:", f"      id: {cid}", f"      sequence: {seq}"]
        if not use_msa:
            lines.append("      msa: empty")
    # Without a template Boltz has to fold the three-subunit GvpA rib from
    # single sequence, and it cannot: measured target RMSD 16-24 A (median
    # 19.6) against the crystal, i.e. every interface metric was scoring the
    # binder against a target that does not exist. Pinning the target with the
    # real structure is the fix.
    if template:
        cif, chain_ids, template_ids = template
        lines += ["templates:", f"  - cif: {cif}",
                  f"    chain_id: [{', '.join(chain_ids)}]",
                  f"    template_id: [{', '.join(template_ids)}]"]
    # A template fixes the TARGET but says nothing about where the binder goes:
    # measured, Boltz docked binders 24-43 A from the designed epitope (AF2
    # puts them at ~12 A) while folding the target to ~1.9 A. The pocket
    # constraint is what tells Boltz which surface to dock against, so the
    # interface metrics describe the designed interface rather than some other
    # face of the rib.
    if pocket:
        binder_id, contacts, max_dist, force = pocket
        lines += ["constraints:", "  - pocket:", f"      binder: {binder_id}",
                  "      contacts: [" +
                  ", ".join(f"[{c}, {r}]" for c, r in contacts) + "]",
                  f"      max_distance: {max_dist}"]
        if force:
            lines.append("      force: true")
    if affinity_binder:
        lines += ["properties:", "  - affinity:", f"      binder: {affinity_binder}"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def parse_results(out_dir):
    rows = {}
    for cp in sorted(Path(out_dir).rglob("confidence_*.json")):
        name = cp.stem.replace("confidence_", "")
        # boltz appends _model_N to the job name
        for suf in ("_model_0", "_model_1", "_model_2"):
            if name.endswith(suf):
                name = name[: -len(suf)]
                break
        p = json.loads(cp.read_text())
        rows[name] = {
            "id": name,
            "boltz_confidence": float(p.get("confidence_score", NAN)),
            "boltz_ptm": float(p.get("ptm", NAN)),
            "boltz_iptm": float(p.get("iptm", NAN)),
            "boltz_complex_plddt": float(p.get("complex_plddt", NAN)),
            # complex_ipde is interface Predicted DISTANCE Error, NOT PAE.
            # It is a different quantity on a different scale, so it must not be
            # compared against Bennett's <10 A interface-PAE threshold -- doing
            # so made composition-matched SCRAMBLES look like they passed.
            "boltz_ipde": float(p.get("complex_ipde", NAN)),
            "boltz_complex_iplddt": float(p.get("complex_iplddt", NAN)),
        }
        # Per-chain-pair iPTM: binder (chain 0, written first) against each
        # target subunit. More specific than the global iptm, which averages
        # the target's own inter-subunit pairs into the number.
        pc = p.get("pair_chains_iptm") or {}
        b_row = pc.get("0", {})
        for k, v in b_row.items():
            if k != "0":
                rows[name][f"boltz_iptm_binder_vs_chain{k}"] = float(v)
        ap = next(iter(cp.parent.glob(f"affinity_{name}*.json")), None)
        if ap:
            aff = json.loads(ap.read_text())
            rows[name]["boltz_affinity"] = float(aff.get("affinity_pred_value", NAN))
            rows[name]["boltz_affinity_prob"] = float(
                aff.get("affinity_probability_binary", NAN))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--accepted", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--work-dir", default=None)
    a = ap.parse_args()

    cfg = common.load_config(a.config)
    bz = cfg["boltz"]
    if not bz.get("enabled", True):
        print("boltz disabled in config"); return

    with open(a.accepted) as fh:
        rows = list(csv.DictReader(fh))
    # Boltz is an orthogonal SECOND OPINION on designs that already passed the
    # gate, and a tie-break on the shortlist. It does not need to see every
    # accepted design, and at ~10 s each the full set is hours of GPU for
    # information that only changes the ordering of the top few dozen. Cap it
    # at the best `max_designs` by composite.
    cap = int(bz.get("max_designs") or 0)
    if cap and len(rows) > cap:
        def comp(r):
            try: return -float(r.get("composite") or 0)
            except (TypeError, ValueError): return 0.0
        rows = sorted(rows, key=comp)[:cap]
        print(f"capped to the top {len(rows)} accepted designs by composite "
              f"(boltz.max_designs)", flush=True)
    rows.sort(key=lambda r: r["id"])
    mine = rows[a.shard :: a.n_shards]
    if not mine:
        print("nothing to score"); return

    root = Path(cfg["run_root"]) / cfg["campaign"]
    work = Path(a.work_dir or root / "boltz" / f"shard{a.shard:03d}")
    jobs, outs = work / "jobs", work / "out"
    jobs.mkdir(parents=True, exist_ok=True)

    # Target sequences come from the target file, so Boltz scores the same
    # three-subunit context AF2 and Rosetta saw -- not a lone GvpA monomer.
    st = common.read_structure(common.resolve(cfg, cfg["target"]["cif"]))
    tgt = {cfg["target"]["chain_map"][c]: st.sequence(c)
           for c in cfg["target"]["design_chains"]}

    template = None
    if bz.get("use_template", True):
        cmap = cfg["target"]["chain_map"]          # input chain -> complex chain
        tpl_path = cfg["target"].get("template_cif") or bz.get("template_cif") \
                   or cfg["target"]["cif"]
        template = (common.resolve(cfg, tpl_path),
                    [cmap[c] for c in cfg["target"]["design_chains"]],
                    list(cfg["target"]["design_chains"]))
        if not Path(template[0]).exists():
            raise SystemExit(f"template not found: {template[0]}")
        print(f"  template: {template[0]}  {template[1]} <- {template[2]}", flush=True)

    pocket = None
    if bz.get("pocket_constraint", True):
        # Every hotspot the campaign targets, mapped into the 1-based numbering
        # of the sequences written above (the target chains start at the input
        # structure's first residue, so subtract that offset).
        offs = common.chain_offsets(cfg)
        cmap = cfg["target"]["chain_map"]
        seen, contacts = set(), []
        for arm in cfg["arms"].values():
            for h in arm["hotspots"]:
                ic, rs = h[0], int(h[1:])
                key = (cmap[ic], rs - offs[ic] + 1)
                if key not in seen:
                    seen.add(key); contacts.append(key)
        contacts.sort()
        pocket = (cfg["target"]["binder_chain"], contacts,
                  float(bz.get("pocket_max_distance", 8.0)),
                  bool(bz.get("pocket_force", False)))
        print(f"  pocket: binder {pocket[0]} vs {len(contacts)} hotspot residues "
              f"(max_distance {pocket[2]})", flush=True)

    for r in mine:
        chains = {cfg["target"]["binder_chain"]: r["seq"]}
        chains.update(tgt)
        write_job(jobs / f"{r['id']}.yaml", chains,
                  bz.get("use_msa_server", False), template=template, pocket=pocket)
    print(f"shard {a.shard}: {len(mine)} boltz jobs -> {jobs}", flush=True)

    cmd = [boltz_binary(), "predict", str(jobs), "--out_dir", str(outs),
           "--devices", "1",
           "--recycling_steps", str(bz.get("recycling_steps", 3)),
           "--diffusion_samples", str(bz.get("diffusion_samples", 1)),
           "--output_format", "pdb"]
    if bz.get("use_msa_server", False):
        cmd.append("--use_msa_server")
    print("  " + " ".join(cmd), flush=True)
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        print(f"boltz exited {res.returncode}; parsing whatever completed", flush=True)

    parsed = parse_results(outs)
    out_rows = []
    for r in mine:
        rec = {"id": r["id"], "design": r["design"]}
        rec.update(parsed.get(r["id"], {}))
        out_rows.append(rec)

    cols, seen = [], set()
    for r in out_rows:
        for k in r:
            if k not in seen:
                seen.add(k); cols.append(k)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(out_rows)
    # NaN-check only works when the key EXISTS: None == None is True, so a
    # missing key was being counted as a score and a run that produced nothing
    # reported "20 with scores".
    got = sum(1 for r in out_rows
              if "boltz_iptm" in r and r["boltz_iptm"] == r["boltz_iptm"])
    print(f"wrote {len(out_rows)} rows ({got} with scores) -> {a.out}")
    if got == 0:
        raise SystemExit(
            "Boltz produced no scores at all -- check the log for "
            "'Failed to process' lines. Not writing a table of empty rows.")
    if got < len(out_rows):
        print(f"WARNING: {len(out_rows)-got} of {len(out_rows)} had no Boltz result")


if __name__ == "__main__":
    main()
