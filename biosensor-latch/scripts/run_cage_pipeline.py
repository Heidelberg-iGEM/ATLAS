#!/usr/bin/env python3
"""
Run the GvpC cage-latch design pipeline in batches until a target number of
constructs has passed, covering these steps only:

    1. Latch generation     RFdiffusion3 (rfd3 design)                [foundry env]
    2. MPNN                 scripts/batch_partial_mpnn.py             [foundry env]
    3. GFP binder attach    build_gfp_fusion_csv.py                   [FlexCraft env]
    4. Monomer scoring      score_gfp_fusion_monomer.py (AF2)         [FlexCraft env]
    5. Rigid docking        score_gfp_rigid_docking.py                [FlexCraft env]
    6. Partner co-folding   score_gfp_fusion_partners.py (AF2)        [FlexCraft env]
                            OPT-IN via --run_partners (off by default)

NOT included (skipped on purpose): score_latch_monomer.py (Test 1), Rosetta.

Step 6 (added 2026-10-20) runs Test 2/3/4 -- partners (your detection target
and/or the blocked target) are co-folded with the construct by AF2 as extra
placeholder chains. It runs ONLY on the constructs that already passed steps
4+5, because it is the most expensive stage in the pipeline: do not point it
at a whole batch. Two flags:
  --run_partners       run step 6; its metrics land in the final CSV with a
                       partners_ prefix, success criterion UNCHANGED (report-only)
  --partners_required  additionally require the partner filter to pass, i.e.
                       success = monomer gate AND rigid docking AND partners
Without --run_partners nothing about this script's behaviour changes.

Place this file next to the other FlexCraft scripts, e.g.:
    flexcraft/pipelines/dual_binder_pipeline/scripts/run_cage_pipeline.py

The wrapper itself is stdlib-only (plus gemmi, which the FlexCraft env already
has for build_gfp_fusion_csv.py), so start it from the FlexCraft env. It
shells out to the foundry env for steps 1-2 via --rfd3_executable /
--foundry_python (use absolute paths to the foundry env's binaries).

What one batch does
--------------------
    rfd3 design -> batch_partial_mpnn.py -> bridge -> build_gfp_fusion_csv
    -> score_gfp_fusion_monomer -> score_gfp_rigid_docking

A construct = one (backbone x MPNN sequence x register offset). It counts as a
SUCCESS when it passes the monomer gate (status=success under the monomer
filter) AND the rigid-docking filter (status=success in score_gfp_rigid_docking).
The loop repeats with fresh RFD3 backbones until --target_count successes are
collected, --max_batches is reached, or --stall_batches consecutive batches
yield zero successes (likely a filter/architecture problem, not bad luck).

The bridge (replaces score_latch_monomer + prepare_scoring_csv)
----------------------------------------------------------------
Because Test 1 is skipped, nothing produces the "Test-1 CSV" that
build_gfp_fusion_csv.py expects. The wrapper builds it itself from
redesign_summary.csv:
  * converts each RFD3 backbone (.cif.gz) to a PDB, THREADS the MPNN sequence
    onto it (residue names) and keeps only N/CA/C/O/CB atoms -- otherwise the
    RFD3 residue names would end up as full_sequence in the fusion constructs;
  * derives binder_start_residue (1-based position of the fixed/latch
    boundary) from boundary_residue_id, or -- if that column is missing/stale
    (older batch_partial_mpnn.py without the 2026-09-16 patch) -- from the
    --fixed_start_sequence / --fixed_end_sequence anchor;
  * writes task_id, status, input_pdb, chain, binder_start_residue,
    redesign_terminus, full_sequence.
--redesign_terminus on the CLI is authoritative (a stale CSV column is ignored
with a warning).

State / resuming
-----------------
Progress lives in <run_dir>/pipeline_state.json. Re-running with the same
--run_name continues after the last COMPLETED batch (a half-finished batch
directory is deleted and redone). --restart ignores the state (old batch
directories are kept on disk unless they are re-created).

Example (Repeat3-4, latch C-terminal)
--------------------------------------

python -u -m flexcraft.pipelines.dual_binder_pipeline.scripts.run_cage_pipeline \
    --foundry_dir /home/hd/hd_hd/hd_ps334/workspace/foundry \
    --rfd3_executable /path/to/foundry_env/bin/rfd3 \
    --foundry_python /path/to/foundry_env/bin/python \
    --rfd3_yaml input_jsons/gvpc_cage_latch/gvpc_repeat3-4_latch.yaml \
    --mpnn_checkpoint ~/.foundry/checkpoints/proteinmpnn_v_48_020.pt \
    --redesign_terminus c \
    --fixed_end_sequence "<end of the fixed Repeat3-4 region>" \
    --gfp_binder_pdb flexcraft/pipelines/dual_binder_pipeline/inputs/gfp_binder_design_1101_0_complex.pdb \
    --gfp_binder_chain B \
    --gfp_chain A \
    --hotspots <binder-local hotspots> \
    --monomer_filter_json filters/monomer_ptm038.json \
    --rigid_filter_json filters/rigid_docking_min3.json \
    --target_count 5 \
    --run_name gfp_cage_latch/auto/repeat3-4_run01 \
    --af2_params params/af/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_MODULE_PREFIX = "flexcraft.pipelines.dual_binder_pipeline.scripts"

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
KEEP_ATOMS = {"N", "CA", "C", "O", "CB"}

BRIDGE_FIELDS = [
    "task_id", "design_id", "seq_id", "status", "input_pdb", "predicted_pdb",
    "chain", "binder_start_residue", "redesign_terminus", "full_sequence", "error",
]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Batch loop: RFD3 latch generation -> MPNN -> GFP binder fusion -> "
            "monomer scoring -> rigid docking, until --target_count constructs pass."
        )
    )

    g = p.add_argument_group("Loop / output")
    g.add_argument("--target_count", type=int, required=True, help="Stop once this many constructs passed monomer gate + rigid docking.")
    g.add_argument("--max_batches", type=int, default=0, help="Safety cap on batches. 0 = unlimited. Default: 0")
    g.add_argument("--stall_batches", type=int, default=3, help="Abort after this many CONSECUTIVE batches without a success. Default: 3")
    g.add_argument("--run_name", required=True, help="Output subfolder inside --results_root (also holds pipeline_state.json).")
    g.add_argument("--repo_root", default=".", help="FlexCraft repository root (cwd of the FlexCraft steps). Default: current directory.")
    g.add_argument("--results_root", default="flexcraft/pipelines/dual_binder_pipeline/results", help="Results directory relative to --repo_root.")
    g.add_argument("--restart", action="store_true", help="Ignore an existing pipeline_state.json and start at batch 0.")
    g.add_argument("--dry_run", action="store_true", help="Print the commands of the next batch without executing anything.")

    g = p.add_argument_group("Step 1: latch generation (RFdiffusion3, foundry env)")
    g.add_argument("--foundry_dir", required=True, help="Foundry workspace (cwd for rfd3 + batch_partial_mpnn.py; relative paths below resolve here).")
    g.add_argument("--rfd3_executable", default="rfd3", help="rfd3 binary (absolute path into the foundry env recommended). Default: rfd3")
    g.add_argument("--rfd3_yaml", required=True, help="RFD3 input YAML (contig, fixed atoms, hotspots) for the chosen variant.")
    g.add_argument("--rfd3_n_batches", type=int, default=1, help="rfd3 n_batches PER pipeline batch. Default: 1")
    g.add_argument("--rfd3_diffusion_batch_size", type=int, default=8, help="rfd3 diffusion_batch_size. Default: 8")
    g.add_argument("--rfd3_seed_base", type=int, default=None, help="If set, passes seed=<base + batch_index> to rfd3 so batches differ. Unverified that rfd3 accepts 'seed=' -- see --rfd3_extra_args.")
    g.add_argument("--rfd3_extra_args", nargs="*", default=[], help="Extra hydra-style args appended to `rfd3 design`, e.g. some_key=value.")

    g = p.add_argument_group("Step 2: ProteinMPNN (batch_partial_mpnn.py, foundry env)")
    g.add_argument("--foundry_python", default=sys.executable, help="Python of the foundry env. Default: this interpreter.")
    g.add_argument("--mpnn_script", default="scripts/batch_partial_mpnn.py", help="Path to batch_partial_mpnn.py (relative to --foundry_dir). Default: scripts/batch_partial_mpnn.py")
    g.add_argument("--mpnn_checkpoint", required=True, help="ProteinMPNN checkpoint, e.g. ~/.foundry/checkpoints/proteinmpnn_v_48_020.pt")
    g.add_argument("--mpnn_chain", default="A", help="Chain of the RFD3 backbones. Default: A")
    g.add_argument("--redesign_terminus", choices=["n", "c"], required=True, help="Latch position: n = latch N-terminal (Repeat1-2), c = latch C-terminal (Repeat3-4).")
    g.add_argument("--fixed_start_sequence", default=None, help="Only for n: start of the fixed GvpC sequence (as in batch_partial_mpnn.py).")
    g.add_argument("--fixed_end_sequence", default=None, help="Only for c: END of the fixed region, directly before the latch (as in batch_partial_mpnn.py).")
    g.add_argument("--mpnn_extra_args", nargs="*", default=[], help="Extra arguments appended to batch_partial_mpnn.py.")

    g = p.add_argument_group("Step 3: GFP binder attachment (build_gfp_fusion_csv.py)")
    g.add_argument("--gfp_binder_pdb", required=True, help="PDB with the GFP + GFP-binder complex.")
    g.add_argument("--gfp_binder_chain", required=True, help="Chain ID of the GFP binder in --gfp_binder_pdb.")
    g.add_argument("--register_offsets", default="-2,-1,0,1,2", help="Default: -2,-1,0,1,2")
    g.add_argument("--filler_residue", default="A", help="Default: A")

    g = p.add_argument_group("Step 4: monomer scoring (score_gfp_fusion_monomer.py)")
    g.add_argument("--hotspots", required=True, help="Binder-local GvpC-binding hotspot residues (comma-separated) for the monomer gate.")
    g.add_argument("--monomer_filter_profile", default="default", help="Default: default")
    g.add_argument("--monomer_filter_json", default=None, help="Custom thresholds JSON (overrides --monomer_filter_profile).")
    g.add_argument("--af2_params", default="params/af/", help="AF2 parameter directory. Default: params/af/")
    g.add_argument("--num_recycle", type=int, default=4, help="AF2 recycles. Default: 4")

    g = p.add_argument_group("Step 5: rigid docking (score_gfp_rigid_docking.py)")
    g.add_argument("--gfp_complex_pdb", default=None, help="Real GFP + GFP-binder complex PDB. Default: --gfp_binder_pdb.")
    g.add_argument("--gfp_chain", required=True, help="Chain ID of GFP itself within --gfp_complex_pdb.")
    g.add_argument("--rigid_filter_profile", default="default", help="Default: default (weak placeholder: >=1 CA clash).")
    g.add_argument("--rigid_filter_json", default=None, help="Custom thresholds JSON, e.g. {\"min_GFP_Blocking_CAClashes\": 3}.")
    g.add_argument("--dock_all_binders", action="store_true", help="Rigid-dock ALL constructs that were folded by the monomer step (score_all.csv), not only monomer-gate passes. "
                   "Success = rigid-docking filter only; monomer status/metrics are still carried into the final CSV (column monomer_gate_passed). Default: off")
    g.add_argument("--ca_clash_distance", type=float, default=3.5, help="Default: 3.5")

    g = p.add_argument_group("Step 6: partner co-folding (score_gfp_fusion_partners.py, AF2) -- opt-in")
    g.add_argument("--run_partners", action="store_true", help="Run step 6 on the constructs that passed steps 4+5. Default: off")
    g.add_argument("--partners_required", action="store_true", help="Also make the partner filter part of the success criterion "
                   "(success = monomer AND rigid AND partners). Without this, step 6 is report-only. Requires --run_partners.")
    g.add_argument("--partners_filter_profile", default="permissive_generic", help="Default: permissive_generic (pLDDT/pTM only -- no partner-metric thresholds).")
    g.add_argument("--partners_filter_json", default=None, help="Custom thresholds JSON (overrides the profile). REQUIRED in practice: the "
                   "useful direction differs per test (Test 2 wants LOW contact, Test 3/4 want HIGH), so no profile can be right by default.")
    g.add_argument("--partners_seed", type=int, default=37, help="PRNG seed passed to score_gfp_fusion_partners.py. Default: 37")
    g.add_argument("--partners_af_model", type=int, default=1, help="AF2 model number for step 6. Default: 1")
    for n in (1, 2):
        label = "detection target (Test 3)" if n == 1 else "second partner (Test 4)"
        g.add_argument(f"--partner{n}_sequence", default=None, help=f"Amino-acid sequence of the {label}. Partner 1 is required with --run_partners.")
        g.add_argument(f"--partner{n}_chain", default="G" if n == 1 else "H", help=f"Chain ID for this partner. Default: {'G' if n == 1 else 'H'}")
        g.add_argument(f"--partner{n}_label", default=None, help="Readable name recorded in partnerN_identity, e.g. GFP or GvpC.")
        g.add_argument(f"--partner{n}_target", choices=["gfp", "binder"], default=None,
                       help="Which segment this partner binds: 'gfp' = the target-binder segment (Test 3), 'binder' = the fixed motif segment (Test 2).")
        g.add_argument(f"--partner{n}_hotspots", default=None,
                       help="Hotspot residues in TARGET-SEGMENT-LOCAL numbering (position 1 = first residue of that segment). "
                            "Not derivable by the script -- you must supply it.")

    g = p.add_argument_group("Environment")
    g.add_argument("--module_prefix", default=DEFAULT_MODULE_PREFIX, help=f"Module path of the FlexCraft scripts. Default: {DEFAULT_MODULE_PREFIX}")
    g.add_argument("--python_executable", default=sys.executable, help="Python for the FlexCraft steps. Default: this interpreter.")
    return p


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def read_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        return fields, list(reader)


def write_csv_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def run_cmd(cmd: list[str], cwd: Path, log_path: Path, dry_run: bool) -> None:
    print(f"    $ {' '.join(cmd)}")
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Foundry tools (rfd3, mpnn) are found via PATH by the child scripts; without an
    # activated conda env that PATH lacks the env's bin/. Prepend the bin/ dir of the
    # executable (absolute path, symlinks NOT resolved) so `mpnn` etc. are found.
    env = os.environ.copy()
    exe = Path(cmd[0]).expanduser()
    if exe.is_absolute():
        env["PATH"] = str(exe.parent) + os.pathsep + env.get("PATH", "")
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed (exit {proc.returncode}). See log: {log_path}")


def flexcraft_cmd(args: argparse.Namespace, module: str, step_args: list[str]) -> list[str]:
    return [args.python_executable, "-u", "-m", f"{args.module_prefix}.{module}", *step_args]


def task_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {row["task_id"] for row in read_csv_rows(path)[1] if row.get("task_id")}


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

def load_state(path: Path, fresh: bool) -> dict:
    if not fresh and path.is_file():
        return json.loads(path.read_text())
    return {"batch_index": 0, "consecutive_zero_batches": 0, "final_pass_task_ids": [], "batches": []}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n")


# --------------------------------------------------------------------------
# Steps 1-2 (foundry env)
# --------------------------------------------------------------------------

def run_rfd3(args, batch_index: int, foundry_dir: Path, rfd3_dir: Path, batch_dir: Path) -> None:
    cmd = [
        args.rfd3_executable, "design",
        f"out_dir={rfd3_dir}",
        f"inputs={resolve(foundry_dir, args.rfd3_yaml)}",
        f"n_batches={args.rfd3_n_batches}",
        f"diffusion_batch_size={args.rfd3_diffusion_batch_size}",
    ]
    if args.rfd3_seed_base is not None:
        cmd.append(f"seed={args.rfd3_seed_base + batch_index}")
    cmd += list(args.rfd3_extra_args)
    run_cmd(cmd, foundry_dir, batch_dir / "logs" / "01_rfd3.log", args.dry_run)
    if not args.dry_run and not list(rfd3_dir.glob("*.cif*")) and not list(rfd3_dir.glob("*.pdb")):
        raise RuntimeError(f"rfd3 produced no structure files in {rfd3_dir}")


def run_mpnn(args, foundry_dir: Path, rfd3_dir: Path, mpnn_dir: Path, batch_dir: Path) -> Path:
    cmd = [
        args.foundry_python, str(resolve(foundry_dir, args.mpnn_script)),
        "--input-dir", str(rfd3_dir),
        "--output-dir", str(mpnn_dir),
        "--checkpoint", str(Path(args.mpnn_checkpoint).expanduser()),
        "--chain", args.mpnn_chain,
    ]
    if args.redesign_terminus == "c":
        cmd += ["--redesign-terminus", "c", "--fixed-end-sequence", args.fixed_end_sequence]
    else:
        cmd += ["--fixed-start-sequence", args.fixed_start_sequence]
    cmd += list(args.mpnn_extra_args)
    run_cmd(cmd, foundry_dir, batch_dir / "logs" / "02_mpnn.log", args.dry_run)
    summary = mpnn_dir / "redesign_summary.csv"
    if not args.dry_run and not summary.is_file():
        raise RuntimeError(f"batch_partial_mpnn.py wrote no redesign_summary.csv in {mpnn_dir}")
    return summary


# --------------------------------------------------------------------------
# Bridge: redesign_summary.csv -> Test-1-style CSV for build_gfp_fusion_csv
# --------------------------------------------------------------------------

def thread_sequence_onto_backbone(source: Path, chain_id: str, sequence: str, out_pdb: Path) -> list[int]:
    """Write a PDB with `sequence` as residue names and only N/CA/C/O/CB atoms.
    Returns the residue numbers (seqid.num) of the chain, in order."""
    import gemmi

    structure = gemmi.read_structure(str(source))
    structure.setup_entities()
    src_chain = next((c for c in structure[0] if c.name == chain_id), None)
    if src_chain is None:
        raise ValueError(f"chain {chain_id!r} not found in {source}")
    residues = [r for r in src_chain if r.find_atom("CA", "*") is not None]
    if len(residues) != len(sequence):
        raise ValueError(f"{len(residues)} residues in {source.name} but MPNN sequence has {len(sequence)}")

    new_chain = gemmi.Chain(chain_id)
    numbers = []
    for position, (res, letter) in enumerate(zip(residues, sequence), start=1):
        new_res = gemmi.Residue()
        new_res.name = ONE_TO_THREE.get(letter.upper(), "ALA")
        new_res.het_flag = "A"
        new_res.seqid = gemmi.SeqId(position, " ")
        for atom in res:
            if atom.name not in KEEP_ATOMS or (atom.name == "CB" and new_res.name == "GLY"):
                continue
            new_atom = gemmi.Atom()
            new_atom.name = atom.name
            new_atom.element = atom.element
            new_atom.pos = atom.pos
            new_atom.occ = 1.0
            new_atom.b_iso = atom.b_iso
            new_res.add_atom(new_atom)
        new_chain.add_residue(new_res)
        numbers.append(res.seqid.num)

    new_structure = gemmi.Structure()
    new_model = gemmi.Model("1")
    new_model.add_chain(new_chain)
    new_structure.add_model(new_model)
    new_structure.setup_entities()
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    new_structure.write_pdb(str(out_pdb))
    return numbers


def boundary_position(args, row: dict, sequence: str, residue_numbers: list[int]) -> int:
    """1-based POSITIONAL binder_start_residue (build_gfp_fusion_csv convention)."""
    from_csv = None
    csv_terminus = (row.get("redesign_terminus") or "").strip().lower()
    boundary_text = row.get("boundary_residue_id") or ""
    match = re.search(r"(\d+)", boundary_text)
    if match and csv_terminus == args.redesign_terminus:
        number = int(match.group(1))
        from_csv = residue_numbers.index(number) + 1 if number in residue_numbers else number

    # Anchor-derived: n -> first fixed residue = start of --fixed_start_sequence;
    # c -> first latch residue = right after --fixed_end_sequence.
    from_anchor = None
    if args.redesign_terminus == "n" and args.fixed_start_sequence:
        idx = sequence.find(args.fixed_start_sequence)
        if idx >= 0:
            from_anchor = idx + 1
    elif args.redesign_terminus == "c" and args.fixed_end_sequence:
        idx = sequence.find(args.fixed_end_sequence)
        if idx >= 0:
            from_anchor = idx + len(args.fixed_end_sequence) + 1

    if from_csv is not None:
        if from_anchor is not None and from_anchor != from_csv:
            print(f"    WARNING {row.get('binder_id')}: boundary from CSV ({from_csv}) != anchor-derived ({from_anchor}); using CSV")
        return from_csv
    if from_anchor is not None:
        return from_anchor
    raise ValueError("no usable boundary (CSV column missing/stale and anchor sequence not found in MPNN sequence)")


def build_bridge_csv(args, batch_index: int, foundry_dir: Path, summary_csv: Path, batch_dir: Path) -> Path:
    _, rows = read_csv_rows(summary_csv)
    threaded_dir = batch_dir / "threaded_pdb"
    out_rows = []
    stale_terminus_warned = False
    skipped = 0

    for row in rows:
        if row.get("status") != "OK":
            continue
        design_id = row.get("binder_id", "")
        seq_id = str(row.get("design_number", ""))
        sequence = (row.get("new_full_sequence") or "").strip()
        try:
            csv_terminus = (row.get("redesign_terminus") or "").strip().lower()
            if csv_terminus != args.redesign_terminus and not stale_terminus_warned:
                print(
                    f"    WARNING: redesign_summary.csv says redesign_terminus={csv_terminus or '(missing)'} but "
                    f"--redesign_terminus={args.redesign_terminus}. Likely an old batch_partial_mpnn.py without the "
                    "2026-09-16 patch on the cluster; using the CLI value + anchor sequence for the boundary."
                )
                stale_terminus_warned = True
            source = Path(row["structure_file"]).expanduser()
            if not source.is_absolute():
                source = foundry_dir / source
            task_id = safe_id(f"b{batch_index:03d}_{design_id}_seq{seq_id}")
            out_pdb = threaded_dir / f"{task_id}.pdb"
            numbers = thread_sequence_onto_backbone(source, args.mpnn_chain, sequence, out_pdb)
            binder_start = boundary_position(args, row, sequence, numbers)
            out_rows.append({
                "task_id": task_id, "design_id": design_id, "seq_id": seq_id, "status": "success",
                "input_pdb": str(out_pdb), "predicted_pdb": str(out_pdb), "chain": args.mpnn_chain,
                "binder_start_residue": binder_start, "redesign_terminus": args.redesign_terminus,
                "full_sequence": sequence, "error": "",
            })
        except Exception as error:  # keep going, report at the end
            skipped += 1
            print(f"    [skip] {design_id}_seq{seq_id}: {type(error).__name__}: {error}")

    if skipped:
        print(f"    {skipped} MPNN row(s) skipped in the bridge")
    if not out_rows:
        raise RuntimeError("bridge produced no usable rows")
    bridge_csv = batch_dir / "bridge_test1_style.csv"
    write_csv_rows(bridge_csv, BRIDGE_FIELDS, out_rows)
    return bridge_csv


# --------------------------------------------------------------------------
# Steps 3-5 (FlexCraft env)
# --------------------------------------------------------------------------

def run_build_fusion(args, batch_name: str, batch_dir: Path, bridge_csv: Path, repo_root: Path, results_root: Path) -> Path:
    run_name = f"{args.run_name}/{batch_name}/gfp_fusion"
    cmd = flexcraft_cmd(args, "build_gfp_fusion_csv", [
        "--test1_csv", str(bridge_csv),
        "--test1_status_filter", "all",
        "--test1_pdb_source", "input",
        "--gfp_binder_pdb", str(resolve(repo_root, args.gfp_binder_pdb)),
        "--gfp_binder_chain", args.gfp_binder_chain,
        "--filler_residue", args.filler_residue,
        f"--register_offsets={args.register_offsets}",  # "=" form: value may start with "-"
        "--chain", args.mpnn_chain,
        "--run_name", run_name,
        "--repo_root", str(repo_root),
        "--results_root", str(args.results_root),
        "--overwrite",
    ])
    run_cmd(cmd, repo_root, batch_dir / "logs" / "03_build_gfp_fusion_csv.log", args.dry_run)
    return results_root / run_name / "gfp_fusion_constructs.csv"


def run_monomer(args, batch_name: str, batch_dir: Path, constructs_csv: Path, repo_root: Path, results_root: Path) -> Path:
    run_name = f"{args.run_name}/{batch_name}/monomer"
    filt = (["--filter_profile", "custom_json", "--filter_json", str(resolve(repo_root, args.monomer_filter_json))]
            if args.monomer_filter_json else ["--filter_profile", args.monomer_filter_profile])
    cmd = flexcraft_cmd(args, "score_gfp_fusion_monomer", [
        "--sequence_csv", str(constructs_csv),
        "--chain", args.mpnn_chain,
        "--run_name", run_name,
        "--repo_root", str(repo_root),
        "--results_root", str(args.results_root),
        "--af2_params", args.af2_params,
        "--num_recycle", str(args.num_recycle),
        "--hotspots", args.hotspots,
        *filt,
        "--overwrite",
    ])
    run_cmd(cmd, repo_root, batch_dir / "logs" / "04_score_gfp_fusion_monomer.log", args.dry_run)
    return results_root / run_name / "score_success.csv"


def run_rigid(args, batch_name: str, batch_dir: Path, gate_csv: Path, repo_root: Path, results_root: Path,
              gate_status_filter: str = "success") -> Path:
    run_name = f"{args.run_name}/{batch_name}/rigid_docking"
    if args.rigid_filter_json:
        filt = ["--filter_profile", "custom_json", "--filter_json", str(resolve(repo_root, args.rigid_filter_json))]
    else:
        print("    WARNING: no --rigid_filter_json -- default profile only requires >=1 CA clash (uncalibrated placeholder).")
        filt = ["--filter_profile", args.rigid_filter_profile]
    cmd = flexcraft_cmd(args, "score_gfp_rigid_docking", [
        "--gate_csv", str(gate_csv),
        "--gate_status_filter", gate_status_filter,
        "--chain", args.mpnn_chain,
        "--gfp_complex_pdb", str(resolve(repo_root, args.gfp_complex_pdb or args.gfp_binder_pdb)),
        "--gfp_binder_chain", args.gfp_binder_chain,
        "--gfp_chain", args.gfp_chain,
        "--ca_clash_distance", str(args.ca_clash_distance),
        "--run_name", run_name,
        "--repo_root", str(repo_root),
        "--results_root", str(args.results_root),
        *filt,
        "--overwrite",
    ])
    run_cmd(cmd, repo_root, batch_dir / "logs" / "05_score_gfp_rigid_docking.log", args.dry_run)
    return results_root / run_name / "score_success.csv"


def write_gate_subset(source_csv: Path, keep_task_ids: set[str], out_csv: Path) -> int:
    """Restrict a monomer score CSV to `keep_task_ids`, keeping every column.

    Step 6 co-folds with AF2 and is the most expensive stage in the pipeline, so
    it only ever sees constructs that already passed steps 4+5. All columns are
    preserved because score_gfp_fusion_partners.py reads the monomer output's own
    fields (redesign_terminus, the two segment boundaries, predicted_pdb, ...)."""
    fields, rows = read_csv_rows(source_csv)
    kept = [row for row in rows if row.get("task_id") in keep_task_ids]
    write_csv_rows(out_csv, fields, kept)
    return len(kept)


def run_partners(args, batch_name: str, batch_dir: Path, gate_csv: Path, repo_root: Path,
                 results_root: Path) -> Path:
    run_name = f"{args.run_name}/{batch_name}/partners"
    if args.partners_filter_json:
        filt = ["--filter_profile", "custom_json", "--filter_json", str(resolve(repo_root, args.partners_filter_json))]
    else:
        print(f"    WARNING: no --partners_filter_json -- the {args.partners_filter_profile!r} profile sets no "
              "partner-metric thresholds, so this run's 'success' column says nothing about binding.")
        filt = ["--filter_profile", args.partners_filter_profile]

    step_args = [
        "--gate_csv", str(gate_csv),
        # The subset CSV IS the selection, so take every row in it.
        "--gate_status_filter", "all",
        "--chain", args.mpnn_chain,
        "--run_name", run_name,
        "--repo_root", str(repo_root),
        "--results_root", str(args.results_root),
        "--af2_params", args.af2_params,
        "--af_model", str(args.partners_af_model),
        "--num_recycle", str(args.num_recycle),
        "--seed", str(args.partners_seed),
        *filt,
        "--overwrite",
    ]
    for n in (1, 2):
        sequence = getattr(args, f"partner{n}_sequence")
        if not sequence:
            continue
        step_args += [
            f"--partner{n}_sequence", sequence,
            f"--partner{n}_chain", getattr(args, f"partner{n}_chain"),
            f"--partner{n}_target", getattr(args, f"partner{n}_target"),
            f"--partner{n}_hotspots", getattr(args, f"partner{n}_hotspots"),
        ]
        label = getattr(args, f"partner{n}_label")
        if label:
            step_args += [f"--partner{n}_label", label]

    cmd = flexcraft_cmd(args, "score_gfp_fusion_partners", step_args)
    run_cmd(cmd, repo_root, batch_dir / "logs" / "06_score_gfp_fusion_partners.log", args.dry_run)
    return results_root / run_name / "score_success.csv"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def validate(args: argparse.Namespace) -> None:
    if args.redesign_terminus == "n" and not args.fixed_start_sequence:
        sys.exit("--redesign_terminus n requires --fixed_start_sequence (and no --fixed_end_sequence).")
    if args.redesign_terminus == "c" and not args.fixed_end_sequence:
        sys.exit("--redesign_terminus c requires --fixed_end_sequence (and no --fixed_start_sequence).")
    if args.redesign_terminus == "n" and args.fixed_end_sequence:
        sys.exit("--fixed_end_sequence is only valid with --redesign_terminus c.")
    if args.redesign_terminus == "c" and args.fixed_start_sequence:
        sys.exit("--fixed_start_sequence is only valid with --redesign_terminus n.")
    if args.target_count <= 0:
        sys.exit("--target_count must be positive.")

    # Step 6 (opt-in). Partner 1 is what score_gfp_fusion_partners.py itself requires.
    if args.partners_required and not args.run_partners:
        sys.exit("--partners_required needs --run_partners (nothing would run otherwise).")
    if args.run_partners:
        if not args.partner1_sequence:
            sys.exit("--run_partners requires --partner1_sequence (the partner is supplied as a SEQUENCE, not a PDB).")
        if not args.partner1_target:
            sys.exit("--run_partners requires --partner1_target {gfp,binder} (which segment the partner binds).")
        if not args.partner1_hotspots:
            sys.exit("--run_partners requires --partner1_hotspots in target-segment-local numbering; "
                     "score_gfp_fusion_partners.py cannot derive them (pass 'none' to run without a hotspot metric).")
        for n in (2,):
            seq = getattr(args, f"partner{n}_sequence")
            if seq and not getattr(args, f"partner{n}_target"):
                sys.exit(f"--partner{n}_sequence requires --partner{n}_target {{gfp,binder}}.")
            if seq and not getattr(args, f"partner{n}_hotspots"):
                sys.exit(f"--partner{n}_sequence requires --partner{n}_hotspots (target-segment-local numbering, or 'none').")
        if args.partner1_chain == args.partner2_chain:
            sys.exit("--partner1_chain and --partner2_chain must differ.")
        if args.partners_required and not args.partners_filter_json:
            sys.exit("--partners_required without --partners_filter_json would gate on the "
                     f"{args.partners_filter_profile!r} profile, which sets no partner-metric thresholds -- "
                     "every construct would pass. Supply real thresholds (see the bundle's filters/ for a Test-3 example).")


def main() -> None:
    args = build_arg_parser().parse_args()
    validate(args)

    repo_root = Path(args.repo_root).expanduser().resolve()
    foundry_dir = Path(args.foundry_dir).expanduser().resolve()
    results_root = resolve(repo_root, args.results_root)
    run_dir = results_root / args.run_name
    state_path = run_dir / "pipeline_state.json"
    final_path = run_dir / "final_passing_constructs.csv"

    state = load_state(state_path, fresh=args.restart)
    final_rows: dict[str, dict] = {}
    if final_path.is_file() and not args.restart:
        final_rows = {r["task_id"]: r for r in read_csv_rows(final_path)[1]}

    n_steps = 6 if args.run_partners else 5

    print(f"Run dir: {run_dir}")
    print(f"Resume:  batch {state['batch_index']}, {len(final_rows)} successes so far, target {args.target_count}")
    if args.run_partners:
        gate_note = "part of the success criterion" if args.partners_required else "report-only"
        print(f"Step 6:  partner co-folding enabled ({gate_note})")
    print()

    while True:
        if len(final_rows) >= args.target_count:
            print(f"Target reached: {len(final_rows)}/{args.target_count} (a batch is always finished completely, so this may overshoot).")
            break
        if args.max_batches and state["batch_index"] >= args.max_batches:
            print(f"Stopping: --max_batches={args.max_batches} reached with {len(final_rows)}/{args.target_count}.")
            break
        if state["consecutive_zero_batches"] >= args.stall_batches:
            print(f"Stopping: {state['consecutive_zero_batches']} consecutive batches without a success -- check filters/architecture "
                  f"before spending more compute. Total: {len(final_rows)}/{args.target_count}.")
            break

        batch_index = state["batch_index"]
        batch_name = f"batch_{batch_index:03d}"
        batch_dir = run_dir / batch_name
        rfd3_dir = batch_dir / "rfd3"
        mpnn_dir = batch_dir / "mpnn"
        print(f"=== {batch_name} ===")

        if batch_dir.exists() and not args.dry_run:
            print(f"  removing unfinished batch dir {batch_dir}")
            shutil.rmtree(batch_dir)

        summary = {"index": batch_index}
        new_rows: list[dict] = []
        try:
            print(f"  [1/{n_steps}] RFD3 latch generation")
            run_rfd3(args, batch_index, foundry_dir, rfd3_dir, batch_dir)
            print(f"  [2/{n_steps}] ProteinMPNN")
            summary_csv = run_mpnn(args, foundry_dir, rfd3_dir, mpnn_dir, batch_dir)

            print(f"  [3/{n_steps}] bridge + GFP binder attachment")
            if args.dry_run:
                bridge_csv = batch_dir / "bridge_test1_style.csv"
                print("    (dry run: bridge skipped, needs real MPNN output)")
            else:
                bridge_csv = build_bridge_csv(args, batch_index, foundry_dir, summary_csv, batch_dir)
                summary["n_sequences"] = len(read_csv_rows(bridge_csv)[1])
            constructs_csv = run_build_fusion(args, batch_name, batch_dir, bridge_csv, repo_root, results_root)
            summary["n_constructs"] = 0 if args.dry_run else len(read_csv_rows(constructs_csv)[1])

            print(f"  [4/{n_steps}] monomer scoring")
            monomer_csv = run_monomer(args, batch_name, batch_dir, constructs_csv, repo_root, results_root)
            monomer_ids = set() if args.dry_run else task_ids(monomer_csv)
            summary["n_monomer_success"] = len(monomer_ids)
            print(f"        -> {len(monomer_ids)}/{summary['n_constructs']} pass the monomer gate")

            monomer_all_csv = monomer_csv.parent / "score_all.csv"
            if args.dock_all_binders:
                dock_csv, dock_filter = monomer_all_csv, "all"
                print("        --dock_all_binders: docking every folded construct, monomer gate is ignored for success")
            else:
                dock_csv, dock_filter = monomer_csv, "success"

            if args.dock_all_binders or monomer_ids or args.dry_run:
                print(f"  [5/{n_steps}] rigid docking")
                rigid_csv = run_rigid(args, batch_name, batch_dir, dock_csv, repo_root, results_root, dock_filter)
                rigid_ids = set() if args.dry_run else task_ids(rigid_csv)
                passed = rigid_ids if args.dock_all_binders else (monomer_ids & rigid_ids)
                summary["n_rigid_success"] = len(passed)
                if args.dock_all_binders:
                    summary["n_rigid_success_and_monomer"] = len(monomer_ids & rigid_ids)
                    print(f"        -> {len(passed)} pass rigid docking ({len(monomer_ids & rigid_ids)} of them also pass the monomer gate)")
                else:
                    print(f"        -> {len(passed)} pass rigid docking as well")

                source_monomer = monomer_all_csv if args.dock_all_binders else monomer_csv
                partners_by_id: dict[str, dict] = {}
                partner_ids: set[str] = set()

                if args.run_partners and (passed or args.dry_run):
                    print(f"  [6/{n_steps}] partner co-folding (AF2)")
                    gate_subset = batch_dir / "partners_gate.csv"
                    if args.dry_run:
                        print("    (dry run: gate subset skipped, needs real monomer output)")
                    else:
                        n_gate = write_gate_subset(source_monomer, passed, gate_subset)
                        print(f"        co-folding {n_gate} construct(s) that passed steps 4+5")
                    partners_csv = run_partners(args, batch_name, batch_dir, gate_subset, repo_root, results_root)
                    partner_ids = set() if args.dry_run else task_ids(partners_csv)
                    summary["n_partners_success"] = len(partner_ids)
                    if args.partners_required:
                        passed = passed & partner_ids
                        print(f"        -> {len(partner_ids)} pass the partner filter; required, so {len(passed)} remain")
                    else:
                        print(f"        -> {len(partner_ids)} pass the partner filter (report-only, success unchanged)")
                    if not args.dry_run:
                        partners_all_csv = partners_csv.parent / "score_all.csv"
                        # Report-only runs want the metrics of failing rows too.
                        metrics_src = partners_all_csv if partners_all_csv.is_file() else partners_csv
                        partners_by_id = {r["task_id"]: r for r in read_csv_rows(metrics_src)[1]}
                elif args.run_partners:
                    summary["n_partners_success"] = 0
                    print("        nothing passed steps 4+5, skipping partner co-folding")

                if passed:
                    monomer_by_id = {r["task_id"]: r for r in read_csv_rows(source_monomer)[1]}
                    rigid_by_id = {r["task_id"]: r for r in read_csv_rows(rigid_csv)[1]}
                    for tid in sorted(passed):
                        row = {"task_id": tid, "batch": batch_name, "monomer_gate_passed": str(tid in monomer_ids)}
                        if args.run_partners:
                            row["partners_gate_passed"] = str(tid in partner_ids)
                        row.update({k: v for k, v in monomer_by_id.get(tid, {}).items() if k != "task_id"})
                        row.update({f"rigid_{k}": v for k, v in rigid_by_id.get(tid, {}).items() if k != "task_id"})
                        row.update({f"partners_{k}": v for k, v in partners_by_id.get(tid, {}).items() if k != "task_id"})
                        new_rows.append(row)
            else:
                summary["n_rigid_success"] = 0
                print("        no monomer-gate successes, skipping rigid docking")
        except RuntimeError as error:
            print(f"  ERROR in {batch_name}: {error}")
            summary["error"] = str(error)

        n_pass = len(new_rows)
        summary["n_final_pass"] = n_pass
        for row in new_rows:
            final_rows[row["task_id"]] = row

        state["batches"].append(summary)
        state["batch_index"] = batch_index + 1
        state["consecutive_zero_batches"] = 0 if n_pass else state["consecutive_zero_batches"] + 1
        state["final_pass_task_ids"] = sorted(final_rows)

        if not args.dry_run:
            save_state(state_path, state)
            if final_rows:
                fields: list[str] = []
                for row in final_rows.values():
                    fields += [k for k in row if k not in fields]
                write_csv_rows(final_path, fields, list(final_rows.values()))

        print(f"=== {batch_name} done: +{n_pass} (total {len(final_rows)}/{args.target_count}) ===\n")
        if args.dry_run:
            print("Dry run: stopping after one planned batch.")
            break

    print("Finished.")
    print(f"Successes: {len(final_rows)}/{args.target_count}")
    print(f"State:     {state_path}")
    print(f"Report:    {final_path}")


if __name__ == "__main__":
    main()
