#!/usr/bin/env python3
"""
Stage 9: AlphaFold2 in initial-guess mode.

    $AF2_PY 20_af2_run.py --config config.yaml

Builds the job, starts 21_af2_score.py in the same env and collects the results in
04_loops/<loop>/scores_af2.csv.

Initial guess means: AF2 is given the backbone of the binder in the designed
pose, sequence and side chains are erased. So it answers "does the binder stay
when you put it there", not "does it find the site on its own". That makes it
the most strongly constrained of the three evaluators.

Why this stage runs before Boltz: AF2ig separates better and costs less per
sequence (two models instead of five seeds). Boltz therefore only sees half.
The price is that the set Boltz sees is preselected - a correlation between the
two must not be computed on it.

The controls run in the same job. Without them the AF2 thresholds cannot be
checked, and the gate rests on two of them.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_registry as registry
import lib_checks as checks

HERE = Path(__file__).resolve().parent

# Columns from 21_af2_score.py that move into the registry. The prefix is
# needed, otherwise they collide with the Boltz columns on the join.
KEEP = {"i_pae": "af_i_pae", "i_pae_m0": "af_i_pae_m0", "i_pae_m1": "af_i_pae_m1",
        "i_ptm": "af_i_ptm", "i_ptm_m1": "af_i_ptm_m1",
        "plddt": "af_plddt", "plddt_m1": "af_plddt_m1",
        "hotspot_rmsd": "af_hotspot_rmsd", "hotspot_rmsd_m1": "af_hotspot_rmsd_m1",
        "placement_rmsd": "af_placement_rmsd", "placement_rmsd_m1": "af_placement_rmsd_m1",
        "shape_rmsd": "af_shape_rmsd", "shape_rmsd_m1": "af_shape_rmsd_m1",
        "complex": "af_complex", "n_models": "af_n_models"}

# Is run in an env with gemmi, if the AF2 env has none.
CONVERT = """
import sys, gemmi
st = gemmi.read_structure(sys.argv[1])
st.setup_entities()
if len(st) == 0:
    sys.exit("gemmi read ZERO models")
st.write_pdb(sys.argv[2])
"""


def cif_to_pdb(cif, pdb, binder, targets, brange, trange, gemmi_py=None):
    """One-time conversion, then check it without fail.

    21_af2_score.py reads the residue numbers directly from the file. If a
    converter renames or renumbers the chains, the hotspots silently point
    at the wrong residues.
    """
    try:
        import gemmi
    except ImportError:
        if not gemmi_py:
            sys.exit("gemmi is missing in this env and envs.boltz_py is not "
                     "set. Either set the key or create the PDB by hand once "
                     "and point template.pdb at it.")
        if subprocess.run([str(gemmi_py), "-c", CONVERT, str(cif), str(pdb)]).returncode:
            sys.exit("Conversion failed")
    else:
        st = gemmi.read_structure(str(cif))
        st.setup_entities()
        if len(st) == 0:
            sys.exit(f"{cif}: gemmi read ZERO models")
        st.write_pdb(str(pdb))

    if not Path(pdb).exists():
        sys.exit(f"{pdb} was not written")
    checks.assert_pdb_numbering(pdb, binder=binder, targets=targets,
                                binder_range=brange, target_range=trange)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--loop", default=None)
    ap.add_argument("--collect", action="store_true",
                    help="do not compute, only read an existing CSV")
    ap.add_argument("--models", default="0,1")
    ap.add_argument("--recycles", type=int, default=3)
    a = ap.parse_args()

    cfg = registry.load_config(a.config)
    loop = a.loop or cfg["project"]["loop"]
    root = Path(cfg["project"]["root"])
    loop_dir = root / "04_loops" / loop
    tpl = cfg["template"]
    binder, targets = tpl["binder_chain"], tpl["target_chains"]
    brange, trange = tpl["binder_range"], tpl["target_range"]
    hotspot_cutoff = float((cfg.get("epitope") or {}).get("hotspot_cutoff", 10.0))

    work = root / "03_output" / "af2_output" / loop
    work.mkdir(parents=True, exist_ok=True)
    job_path, out_csv = work / "af2_job.yaml", work / "af2_scores.csv"

    if not a.collect:
        # Provide the complex as PDB
        cif = registry.resolve(cfg, tpl["cif"])
        pdb = registry.resolve(cfg, tpl["pdb"]) if tpl.get("pdb") else cif.with_suffix(".pdb")
        if pdb.exists():
            print(f"[af2] checking existing {pdb.name}")
            checks.assert_pdb_numbering(pdb, binder=binder, targets=targets,
                                        binder_range=brange, target_range=trange)
        else:
            print(f"[af2] converting {cif.name} -> {pdb.name}")
            cif_to_pdb(cif, pdb, binder, targets, brange, trange,
                       gemmi_py=(cfg.get("envs") or {}).get("boltz_py"))

        # Sequences: prefilter passers plus all controls
        rows = registry.join_all(loop_dir)
        sel = registry.to_predict(rows)
        if not sel:
            sys.exit("No prefilter passers. Run stage 4 first.")
        n_ctrl = sum(1 for r in sel if registry.is_control(r))
        if not n_ctrl:
            print("  ! No controls in the run. The AF2 thresholds therefore stay "
                  "uncalibrated.", file=sys.stderr)
        print(f"[af2] {len(sel)} sequences, {n_ctrl} of them controls")

        lines = [
            f"params_dir: {cfg['envs']['af2_params']}",
            f"complex: {pdb}",
            f"binder_chain: {binder}",
            f"target_chains: [{', '.join(targets)}]",
            f"hotspots: [{', '.join(cfg['epitope']['af2_hotspots'])}]",
            f"hotspot_cutoff: {hotspot_cutoff}",
            f"num_recycles: {a.recycles}",
            f"models: [{a.models}]",
            "sequences:",
        ]
        for r in sel:
            line = f"  - {{id: {r['seq_id']}, seq: {r['binder_seq'].strip().upper()}"
            # From round 2 on, the pose of the parent can be specified per entry.
            if r.get("af_complex_override"):
                line += f", complex: {r['af_complex_override']}"
            lines.append(line + "}")
        job_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[af2] Job -> {job_path}")

        registry.run_in_env(cfg["envs"]["af2_py"], HERE / "21_af2_score.py",
                            [job_path, "-o", out_csv])

    if not out_csv.exists():
        sys.exit(f"{out_csv} is missing - the AF2 run did not get as far as the output.")
    with out_csv.open(newline="", encoding="utf-8") as fh:
        scored = list(csv.DictReader(fh))

    out = [{"seq_id": r["id"], **{dst: r[src] for src, dst in KEEP.items() if src in r}}
           for r in scored]

    dest = loop_dir / "scores_af2.csv"
    registry.write_csv(dest, out)
    registry.log_stage(loop_dir, "20_af2_run",
                       {"n": len(out), "job": str(job_path),
                        "models": a.models, "recycles": a.recycles})
    print(f"\n[af2] {len(out)} rows -> {dest}")
    print("      The gate acts on af_i_pae_m1, i.e. model 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
