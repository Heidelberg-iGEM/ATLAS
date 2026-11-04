#!/usr/bin/env python
"""Score a binder against a multi-chain target with Boltz-2. Standalone.

WHY THIS EXISTS (and why naive Boltz gives garbage on a multi-subunit target)
============================================================================
Running Boltz on "binder + target chains" with nothing else produces confident
nonsense. Measured on a 3-subunit GvpA rib, 10 designs + 10 composition-matched
scrambles:

  setup                       target RMSD   binder->epitope   iptm AUC vs scramble
  sequences only                 19.5 A          28.7 A            0.52  (blind)
  + template                      1.93 A          32.1 A            0.87
  + template + pocket             1.87 A          12.9 A            1.00

Two independent failures, and fixing only one does not help:

1. NO TEMPLATE -> Boltz cannot fold the target.
   Single-sequence, no MSA, it has to assemble three separate subunits from
   scratch and collapses them into a globular clump 19.5 A from the truth.
   Every interface number was then describing a target that does not exist,
   which is why ipTM was blind (AUC 0.52, z 0.04).

2. TEMPLATE ALONE -> right target, wrong docking site.
   A template pins the TARGET; it says nothing about where the binder goes.
   Boltz docked binders 24-43 A from the designed epitope -- often the opposite
   face of the sheet. A `pocket` constraint naming the hotspot residues is what
   puts the binder on the intended surface.

A THIRD trap, which costs an afternoon if you hit it: the template file must be
a REAL mmCIF. A stripped, atom_site-only CIF (the kind many tools emit) is read
by gemmi as ZERO models, so Boltz dies with a bare IndexError that it prints as
"Error: ." with no message. A plain PDB fails too. Build the template with
gemmi: read a full entry, keep/rename the chains you want, `setup_entities()`,
`assign_label_seq_id()`, `make_mmcif_document().write_file(...)`.

CAVEAT: the pocket constraint TELLS Boltz where to dock. It can then no longer
tell you whether that site is right -- only whether a sequence looks like a
real binder at that site. Keep it as an orthogonal second opinion, not a gate.

USAGE
=====
    python scripts/boltz_score_binder.py \
        --binder-seq VAELSLETREFLSVTTAKRQEQAEKQAQELQAF \
        --template inputs/gvpa_template_ABE.cif \
        --pocket-residues 52,55,56,59 \
        --out-dir runs/gvpa_v2/boltz_gvpc \
        --name gvpc_repeat

Target chain sequences are read from the template, so the template is the only
structural input. `--pocket-residues` are given in the TEMPLATE's own numbering
and converted to the 1-based sequence indices Boltz expects.

Runs in the foundry env (needs numpy/scipy/pyyaml); calls the boltz binary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common  # noqa: E402

NAN = float("nan")


def boltz_binary():
    for c in (os.environ.get("BOLTZ_BIN"),
              str(Path.home() / ".conda/envs/boltz/bin/boltz")):
        if c and Path(c).exists():
            return c
    raise SystemExit("boltz not found; set BOLTZ_BIN")


def write_job(path, binder_id, binder_seq, target_seqs, template, pocket, max_dist):
    """One Boltz YAML: sequences + template + pocket constraint."""
    lines = ["version: 1", "sequences:",
             "  - protein:", f"      id: {binder_id}",
             f"      sequence: {binder_seq}", "      msa: empty"]
    for cid, seq in target_seqs.items():
        lines += ["  - protein:", f"      id: {cid}", f"      sequence: {seq}",
                  "      msa: empty"]
    tpl_path, complex_chains, template_chains = template
    lines += ["templates:", f"  - cif: {tpl_path}",
              f"    chain_id: [{', '.join(complex_chains)}]",
              f"    template_id: [{', '.join(template_chains)}]"]
    if pocket:
        lines += ["constraints:", "  - pocket:", f"      binder: {binder_id}",
                  "      contacts: [" + ", ".join(f"[{c}, {r}]" for c, r in pocket) + "]",
                  f"      max_distance: {max_dist}"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def parse_results(out_dir):
    """job name -> flat dict of confidence metrics (keyed by directory name)."""
    rows = {}
    for f in Path(out_dir).rglob("confidence_*.json"):
        job = f.parent.name                      # directory name, no string surgery
        p = json.loads(f.read_text())
        r = {"job": job,
             "confidence": p.get("confidence_score", NAN),
             "ptm": p.get("ptm", NAN),
             "iptm": p.get("iptm", NAN),
             "complex_plddt": p.get("complex_plddt", NAN),
             "complex_iplddt": p.get("complex_iplddt", NAN),
             "ipde": p.get("complex_ipde", NAN)}
        for k, v in (p.get("pair_chains_iptm") or {}).get("0", {}).items():
            if k != "0":
                r[f"iptm_binder_vs_chain{k}"] = v
        rows[job] = r
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binder-seq", required=True)
    ap.add_argument("--name", default="binder")
    ap.add_argument("--template", required=True,
                    help="REAL mmCIF of the target (see module docstring)")
    ap.add_argument("--template-chains", default=None,
                    help="comma list of template chains to use (default: all)")
    ap.add_argument("--pocket-residues", default="",
                    help="comma list in TEMPLATE numbering, e.g. 52,55,56,59")
    ap.add_argument("--pocket-max-distance", type=float, default=8.0)
    ap.add_argument("--binder-chain", default="A")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scramble-control", action="store_true", default=True,
                    help="also score a composition-matched scramble (default on)")
    ap.add_argument("--no-scramble-control", dest="scramble_control",
                    action="store_false")
    ap.add_argument("--recycling-steps", type=int, default=3)
    ap.add_argument("--diffusion-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tpl = Path(a.template).resolve()
    if not tpl.exists():
        raise SystemExit(f"template not found: {tpl}")
    st = common.read_structure(tpl)
    tchains = ([c.strip() for c in a.template_chains.split(",")]
               if a.template_chains else list(dict.fromkeys(str(c) for c in st.chain)))

    # Complex chain ids: binder first, then one per target chain.
    alphabet = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c != a.binder_chain]
    complex_chains = alphabet[:len(tchains)]
    target_seqs = {cc: st.sequence(tc) for cc, tc in zip(complex_chains, tchains)}

    # Pocket residues: template numbering -> 1-based index into the sequence.
    pocket = []
    if a.pocket_residues.strip():
        wanted = [int(x) for x in a.pocket_residues.split(",") if x.strip()]
        for cc, tc in zip(complex_chains, tchains):
            first = int(st.resseq[st.chain == tc].min())
            for r in wanted:
                pocket.append((cc, r - first + 1))

    out = Path(a.out_dir).resolve()
    jobs, outs = out / "jobs", out / "out"
    jobs.mkdir(parents=True, exist_ok=True)

    seq = a.binder_seq.strip().upper()
    entries = [(a.name, seq)]
    if a.scramble_control:
        rng = random.Random(a.seed)
        body = list(seq[1:-1]); rng.shuffle(body)
        entries.append((a.name + "__scramble", seq[0] + "".join(body) + seq[-1]))

    print(f"template      : {tpl}")
    print(f"target chains : {dict(zip(complex_chains, tchains))}")
    print(f"pocket        : {len(pocket)} contacts, max_distance {a.pocket_max_distance}")
    for n, s in entries:
        write_job(jobs / f"{n}.yaml", a.binder_chain, s, target_seqs,
                  (str(tpl), complex_chains, tchains), pocket, a.pocket_max_distance)
        print(f"  job {n}  ({len(s)} aa)")

    cmd = [boltz_binary(), "predict", str(jobs), "--out_dir", str(outs),
           "--devices", "1", "--recycling_steps", str(a.recycling_steps),
           "--diffusion_samples", str(a.diffusion_samples), "--output_format", "pdb"]
    print("\n" + " ".join(cmd) + "\n", flush=True)
    rc = subprocess.run(cmd, text=True).returncode
    if rc != 0:
        print(f"boltz exited {rc}; parsing whatever completed", flush=True)

    res = parse_results(outs)
    if not res:
        raise SystemExit(
            "Boltz produced no results. If the log shows 'Failed to process ... "
            "Error: .' the template is not a real mmCIF -- see the module docstring.")

    # Did the target fold, and did the binder land on the pocket?
    ref_t = np.concatenate([st.ca(c) for c in tchains])
    epi = None
    if pocket:
        import numpy as _np
        tips = [st.xyz[(st.chain == tc) & (st.resseq == r)].mean(0)
                for tc in tchains for r in [int(x) for x in a.pocket_residues.split(",")]]
        epi = _np.mean(tips, axis=0)
    for job, r in res.items():
        pdbs = list((outs).rglob(f"{job}/*_model_0.pdb"))
        if not pdbs:
            continue
        pst = common.read_structure(pdbs[0])
        bt = np.concatenate([pst.ca(c) for c in complex_chains])
        b = pst.ca(a.binder_chain)
        if len(bt) == len(ref_t):
            r["target_rmsd"] = round(common.kabsch_rmsd(bt, ref_t), 2)
            if epi is not None:
                mc, rc2 = bt.mean(0), ref_t.mean(0)
                V, S, Wt = np.linalg.svd((bt - mc).T @ (ref_t - rc2))
                U = V @ np.diag([1, 1, np.sign(np.linalg.det(V @ Wt))]) @ Wt
                r["binder_to_pocket"] = round(
                    float(np.linalg.norm(((b - mc) @ U + rc2).mean(0) - epi)), 1)
        r["structure"] = str(pdbs[0])

    csv_path = out / "scores.csv"
    cols = sorted({k for r in res.values() for k in r})
    cols = ["job"] + [c for c in cols if c != "job"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        w.writerows(res.values())

    print("\n=== RESULTS ===")
    show = ["iptm", "ptm", "confidence", "complex_plddt", "ipde",
            "target_rmsd", "binder_to_pocket"]
    print(f"  {'job':<28}" + "".join(f"{k:>16}" for k in show))
    for job, r in sorted(res.items()):
        print(f"  {job[:26]:<28}" + "".join(
            f"{r.get(k, float('nan')):>16.3f}" if isinstance(r.get(k), (int, float))
            else f"{'-':>16}" for k in show))
    print(f"\n  scores    : {csv_path}")
    print(f"  structures: {outs}")
    print("\n  SANITY: target_rmsd should be ~2 A (template applied) and")
    print("  binder_to_pocket should be ~10-15 A (pocket constraint applied).")
    print("  A large target_rmsd means the template was not used; a large")
    print("  binder_to_pocket means the pocket constraint was not used.")


if __name__ == "__main__":
    main()
