#!/usr/bin/env python3
"""
Rigid-docking clash test for GFP-binder-fusion constructs: aligns a REAL
GFP + GFP-binder complex structure onto the GFP-binder segment of a fused
[binder | blocking peptide | GFP binder] chain (score_gfp_fusion_monomer.py's
AF2-folded output), by rigid Kabsch superposition over the GFP-binder's own
CA atoms -- NO new AF2 folding, no jax/AF2 dependency at all.

Place this file next to build_gfp_fusion_csv.py / score_gfp_fusion_monomer.py /
score_gfp_fusion_partners.py, e.g.:
    flexcraft/pipelines/dual_binder_pipeline/scripts/score_gfp_rigid_docking.py

Background / why this exists
-----------------------------
score_gfp_fusion_partners.py's Test 3 (--partner1_target gfp) answers "does
GFP bind the GFP binder AND displace the blocking peptide" by CO-FOLDING GFP
(as a sequence-only placeholder chain, idealized backbone) together with the
whole fused construct via AF2 from scratch -- every metric there
(Blocking_RMSD, HotspotContactCount, PositionalDisplacement, ...) comes out
of that fresh AF2 prediction, not from any pre-existing GFP structure. AF2's
ability to co-fold this correctly is exactly what smoke tests so far have
found unreliable/ambiguous (see Blocking-Peptide-Latch-Design.md,
GvpC-Cage-Latch-Design.md).

This script asks a different, purely geometric question instead, using a
REAL, already-known GFP+GFP-binder complex structure (e.g. the same complex
PDB passed to build_gfp_fusion_csv.py's --gfp_binder_pdb) as ground truth for
how GFP sits relative to the GFP binder when actually bound. It rigidly
superimposes that complex's GFP-binder chain onto the GFP-binder segment of
the fused construct (Kabsch fit over CA atoms only -- same sequence/length in
both, no flexibility assumed), then carries GFP along with that same
rigid-body transform, WITHOUT running AF2 at all. It then asks: in that
rigid, "GFP correctly bound" pose, does GFP's real geometry physically clash
with the blocking peptide and/or the fixed binder segment as the fused chain
currently sits in its own (monomer-gate AF2-folded) closed conformation?

INVERTED success criterion -- please read before trusting results
-------------------------------------------------------------------
Unlike every other filter in this pipeline, HIGH clash counts here are the
DESIRED outcome, not a defect. The switch's whole premise is that GFP, in
its real binding pose, occupies space the blocking peptide currently sits
in -- that overlap is exactly why GFP binding must displace the latch. A
design with ZERO clashes between rigidly-docked GFP and the blocking peptide
means GFP could bind without disturbing the latch at all, i.e. no coupling
-- the switch would not work. The filter profiles below therefore use `min_`
thresholds on the clash-count columns, not `max_`. "High" has not been
calibrated against any known-good/known-bad design yet -- the "default"
profile below (min_GFP_Blocking_CAClashes: 1, i.e. "any clash at all") is a
deliberately weak placeholder, same philosophy as
score_gfp_fusion_partners.py's own permissive_generic/no_filter profiles.
Set a real threshold via --filter_json once score_all.csv's value
distribution is visible.

Independent of score_gfp_fusion_partners.py's Test 3: no AF2/jax/GPU
dependency, so both can run at the same time on the same --gate_csv without
contending for resources. This script's task_id is carried through
unchanged from --gate_csv, so its score_all.csv can be joined 1:1 against
Test 3's score_all.csv by task_id for a side-by-side comparison.

ASSUMPTIONS (please verify before trusting results):
------------------------------------------------------------
1. --gfp_binder_chain in --gfp_complex_pdb has the IDENTICAL sequence (same
   length, same residue order) as the GFP segment of the fused chain -- true
   by construction if --gfp_complex_pdb is the same PDB passed to
   build_gfp_fusion_csv.py's --gfp_binder_pdb/--gfp_binder_chain, since that
   script reads the GFP-binder sequence verbatim from that same chain. A
   length mismatch fails the task loudly (scoring_failure) rather than
   silently truncating or misaligning.
2. Which physical segment is the GFP segment (vs. blocking vs. binder)
   depends on redesign_terminus, read per-row from --gate_csv (same
   convention as score_gfp_fusion_monomer.py/score_gfp_fusion_partners.py,
   default "n" for older CSVs without the column). Segment boundaries are
   applied POSITIONALLY (the Nth residue of the chain), never against
   residue.seqid.num -- the gate structure is AF2's own predicted_pdb by
   default (--gate_pdb_source predicted), which is not guaranteed to be
   1-indexed (same root cause as build_gfp_fusion_csv.py's 2026-09-01
   off-by-one bug; found recurring here 2026-09-17 against real Repeat3-4
   data and fixed the same way -- score_gfp_fusion_partners.py had the
   identical latent bug and was fixed in the same pass).
3. The rigid-body assumption itself -- that the GFP binder's local backbone
   conformation is the same whether free, in the reference complex, or
   fused into this construct -- is a simplification, not a fact. A residual
   fit RMSD (gfp_binder_alignment_rmsd) is written out so a poor fit is
   visible rather than silently producing a meaningless clash count; treat
   a high value here as a reason to distrust that row's clash numbers.
4. Clash counting is CA-CA distance only (--ca_clash_distance, default 3.5
   A), same convention as every other *_CAClashes column in this pipeline
   -- coarse by design, not a real steric/atomic clash detector.

Custom filter JSON example (inverted direction, high clash = success):
------------------------------------------------------------------------
{
  "min_GFP_Blocking_CAClashes": 3
}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import gemmi
import numpy as np


FILTER_PROFILES = {
    "no_filter": {},

    # Deliberately weak placeholder -- see module docstring's "INVERTED
    # success criterion" section. Set a real threshold via --filter_json.
    "default": {
        "min_GFP_Blocking_CAClashes": 1,
    },
}

OUTPUT_FIELDS = [
    "rank",
    "task_id",
    "seq_id",
    "register_offset",
    "redesign_terminus",
    "status",
    "failure_reason",
    "error_message",
    "input_pdb",
    "chain",
    "gfp_segment_end_residue",
    "binder_start_residue",
    "gfp_complex_source_pdb",
    "gfp_binder_chain",
    "gfp_chain",
    "num_gfp_segment_residues",
    "gfp_binder_alignment_rmsd",
    "GFP_Blocking_CAClashes",
    "GFP_Blocking_MinCADistance",
    "GFP_Binder_CAClashes",
    "GFP_Binder_MinCADistance",
    "GFP_BlockingOrBinder_CAClashes",
    "GFP_BlockingOrBinder_MinCADistance",
    "docked_pdb",
]


@dataclass(frozen=True)
class ScoringTask:
    task_id: str
    seq_id: str
    register_offset: str
    redesign_terminus: str
    input_pdb: Path
    chain: str
    gfp_segment_end_residue: int
    binder_start_residue: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rigid-docking clash test: superimposes a real GFP+GFP-binder "
            "complex onto the GFP-binder segment of a fused construct (no "
            "AF2) and counts CA clashes between GFP and the blocking "
            "peptide/binder segments. High clashes = success (see module "
            "docstring)."
        )
    )
    parser.add_argument(
        "--gate_csv",
        required=True,
        help="score_all.csv/score_success.csv output from score_gfp_fusion_monomer.py.",
    )
    parser.add_argument(
        "--gate_status_filter",
        default="success",
        help="Only score gate rows with this status. Use 'all' for every row. Default: success",
    )
    parser.add_argument(
        "--gate_pdb_source",
        choices=["predicted", "input"],
        default="predicted",
        help="Which gate structure to dock GFP onto. Default: predicted",
    )
    parser.add_argument("--chain", default="A", help="Chain ID of the fused chain in the gate output. Default: A")

    parser.add_argument(
        "--gfp_complex_pdb",
        required=True,
        help=(
            "PDB file containing a REAL GFP + GFP-binder complex (e.g. the "
            "same PDB passed to build_gfp_fusion_csv.py's --gfp_binder_pdb)."
        ),
    )
    parser.add_argument(
        "--gfp_binder_chain",
        required=True,
        help="Chain ID of the GFP binder within --gfp_complex_pdb (the part fused into the construct).",
    )
    parser.add_argument(
        "--gfp_chain",
        required=True,
        help="Chain ID of GFP itself within --gfp_complex_pdb (carried along by the rigid transform).",
    )
    parser.add_argument(
        "--gfp_chain_out",
        default="Z",
        help="Chain ID assigned to the docked GFP copy in --docked_pdb output files. Default: Z",
    )

    parser.add_argument("--ca_clash_distance", type=float, default=3.5, help="CA distance below which clashes are counted. Default: 3.5 A")

    parser.add_argument(
        "--write_docked_pdb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a PDB per task with the fused chain + rigidly-docked GFP chain, for visual (PyMOL) inspection. Default: enabled.",
    )

    parser.add_argument("--filter_profile", choices=list(FILTER_PROFILES) + ["custom_json"], default="default")
    parser.add_argument("--filter_json", default=None)

    parser.add_argument("--run_name", required=True, help="Output subfolder name inside --results_root.")
    parser.add_argument("--repo_root", default=".", help="FlexCraft repository root.")
    parser.add_argument(
        "--results_root",
        default="flexcraft/pipelines/dual_binder_pipeline/results",
        help="Pipeline results directory relative to --repo_root.",
    )

    parser.add_argument("--max_tasks", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")

    return parser


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def ensure_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}\nUse a new --run_name or pass --overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def is_protein_residue(residue: gemmi.Residue) -> bool:
    return bool(gemmi.find_tabulated_residue(residue.name).is_amino_acid())


def chain_protein_residues(structure: gemmi.Structure, chain_id: str) -> list[gemmi.Residue]:
    model = structure[0]
    for chain in model:
        if chain.name != chain_id:
            continue
        residues = [
            residue
            for residue in chain
            if is_protein_residue(residue) and residue.find_atom("CA", "*") is not None
        ]
        if not residues:
            raise ValueError(f"Chain {chain_id!r} has no usable protein residues.")
        return residues
    raise ValueError(f"Chain {chain_id!r} not found.")


def residue_ca(residue: gemmi.Residue) -> np.ndarray:
    atom = residue.find_atom("CA", "*")
    return np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)


def ca_array(residues: Sequence[gemmi.Residue]) -> np.ndarray:
    if not residues:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack([residue_ca(r) for r in residues])


def segment_masks(
    residues: Sequence[gemmi.Residue],
    gfp_segment_end_residue: int,
    binder_start_residue: int,
    redesign_terminus: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same head/tail split + redesign_terminus flip as
    score_gfp_fusion_monomer.py/score_gfp_fusion_partners.py: "n" GFP segment
    leads (head), binder segment trails (tail); "c" binder segment leads,
    GFP segment trails. Returns (is_gfp, is_blocking, is_binder).

    Uses 1-based POSITIONAL index (the Nth entry of `residues`, already
    filtered/ordered by chain_protein_residues), NOT residue.seqid.num --
    `residues` typically comes from a gate structure's predicted_pdb (the
    default --gate_pdb_source), which is AF2's own save_pdb() output and is
    NOT guaranteed to be 1-indexed. Comparing seqid.num against
    gfp_segment_end_residue/binder_start_residue (which are always 1-based
    positional indices, per build_gfp_fusion_csv.py's own convention) would
    then be off by exactly one residue -- the same root cause as the
    2026-09-01 off-by-one bug in build_gfp_fusion_csv.py's Test-1-PDB
    handling, recurring here for score_gfp_fusion_monomer.py's predicted
    PDBs instead."""
    positions = np.arange(1, len(residues) + 1, dtype=np.int64)
    head_mask = positions <= gfp_segment_end_residue
    tail_mask = positions >= binder_start_residue
    if redesign_terminus == "n":
        is_gfp, is_binder = head_mask, tail_mask
    else:
        is_binder, is_gfp = head_mask, tail_mask
    is_blocking = (~is_gfp) & (~is_binder)
    return is_gfp, is_blocking, is_binder


def kabsch_rotation_translation(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same Kabsch fit as score_gfp_fusion_partners.py's
    positional_displacement() helper -- rotation+translation that maps
    `mobile` onto `target` in the least-squares sense."""
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, d])
    rotation = vt.T @ correction @ u.T
    translation = target_center - rotation @ mobile_center
    return rotation, translation


def apply_transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points @ rotation.T + translation


def ca_clash_metrics(a_ca: np.ndarray, b_ca: np.ndarray, threshold: float) -> tuple[int, float]:
    if len(a_ca) == 0 or len(b_ca) == 0:
        return 0, math.nan
    distances = np.linalg.norm(a_ca[:, None, :] - b_ca[None, :, :], axis=-1)
    return int(np.sum(distances < threshold)), float(np.min(distances))


def build_transformed_chain(
    residues: Sequence[gemmi.Residue], rotation: np.ndarray, translation: np.ndarray, chain_id: str
) -> gemmi.Chain:
    """Copies every atom of `residues` (not just CA) into a new chain with
    coordinates carried through the same rigid-body transform -- used only
    to write a docked PDB for visual inspection, never for scoring."""
    chain = gemmi.Chain(chain_id)
    for residue in residues:
        new_residue = gemmi.Residue()
        new_residue.name = residue.name
        new_residue.seqid = residue.seqid
        new_residue.het_flag = "A"
        for atom in residue:
            position = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)
            new_position = apply_transform(position[None, :], rotation, translation)[0]
            new_atom = gemmi.Atom()
            new_atom.name = atom.name
            new_atom.element = atom.element
            new_atom.pos = gemmi.Position(*new_position)
            new_atom.occ = atom.occ
            new_atom.b_iso = atom.b_iso
            new_residue.add_atom(new_atom)
        chain.add_residue(new_residue)
    return chain


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def read_scoring_csv_rows(csv_path: Path):
    """Tolerant reader -- see the identically-named helper in
    score_gfp_fusion_partners.py/build_gfp_fusion_csv.py for the rationale."""
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            raw_header = next(reader)
        except StopIteration:
            return
        fieldnames = [name.strip() for name in raw_header]
        for row_number, raw_row in enumerate(reader, start=2):
            if len(raw_row) != len(fieldnames):
                raise ValueError(f"{csv_path}, line {row_number}: field count mismatch with header.")
            yield {key: (value.strip() if isinstance(value, str) else value) for key, value in zip(fieldnames, raw_row)}


def tasks_from_gate_csv(csv_path: Path, repo_root: Path, status_filter: str, pdb_source: str) -> list[ScoringTask]:
    pdb_column = "predicted_pdb" if pdb_source == "predicted" else "input_pdb"
    rows = list(read_scoring_csv_rows(csv_path))
    if rows and pdb_column not in rows[0]:
        raise ValueError(f"Column {pdb_column!r} not found in {csv_path} -- is this a score_gfp_fusion_monomer.py output?")

    tasks: list[ScoringTask] = []
    for row in rows:
        if status_filter != "all" and row.get("status") != status_filter:
            continue
        pdb_text = row.get(pdb_column, "")
        if not pdb_text:
            continue
        input_pdb = Path(pdb_text).expanduser()
        if not input_pdb.is_absolute():
            input_pdb = repo_root / input_pdb

        gfp_end = (row.get("gfp_segment_end_residue") or "").strip()
        binder_start = (row.get("binder_start_residue") or "").strip()
        if not gfp_end or not binder_start:
            raise ValueError(f"Row {row.get('task_id')!r} missing gfp_segment_end_residue/binder_start_residue.")

        # Default "n" for older gate CSVs written before score_gfp_fusion_monomer.py's
        # 2026-09-17 redesign_terminus update -- they are all N-terminal fusions.
        redesign_terminus = (row.get("redesign_terminus") or "n").strip().lower()
        if redesign_terminus not in ("n", "c"):
            raise ValueError(
                f"Invalid redesign_terminus {redesign_terminus!r} for row "
                f"{row.get('task_id')!r} in {csv_path} -- must be 'n' or 'c'."
            )

        tasks.append(
            ScoringTask(
                task_id=row.get("task_id") or input_pdb.stem,
                seq_id=str(row.get("seq_id", "")),
                register_offset=str(row.get("register_offset", "")),
                redesign_terminus=redesign_terminus,
                input_pdb=input_pdb.resolve(),
                chain=(row.get("chain") or "A").strip() or "A",
                gfp_segment_end_residue=int(gfp_end),
                binder_start_residue=int(binder_start),
            )
        )

    if not tasks:
        raise ValueError(f"No usable rows found in {csv_path}")
    return tasks


def load_filter_config(args: argparse.Namespace, repo_root: Path) -> dict[str, float]:
    if args.filter_profile == "custom_json":
        if not args.filter_json:
            raise ValueError("--filter_profile custom_json requires --filter_json.")
        with resolve_path(repo_root, args.filter_json).open() as handle:
            return json.load(handle)
    config = dict(FILTER_PROFILES[args.filter_profile])
    if args.filter_json:
        with resolve_path(repo_root, args.filter_json).open() as handle:
            config.update(json.load(handle))
    return config


def threshold_failure(value: float, threshold: float, mode: str) -> bool:
    if math.isnan(safe_float(value)):
        return True
    if mode == "max":
        return safe_float(value) > threshold
    if mode == "min":
        return safe_float(value) < threshold
    raise ValueError(f"Unknown threshold mode: {mode}")


def evaluate_filters(row: dict, filter_config: dict[str, float]) -> tuple[bool, str]:
    """Generic min_<Column>/max_<Column> evaluation over every numeric
    OUTPUT_FIELDS column -- same pattern as score_gfp_fusion_partners.py's
    evaluate_filters(), just with no test-specific direction baked in (this
    script's whole point is that the "good" direction is inverted vs. every
    other filter in the pipeline, so nothing here assumes a direction)."""
    if not filter_config:
        return True, "none"
    failures = []
    non_metric_fields = {
        "rank", "task_id", "seq_id", "register_offset", "redesign_terminus", "status",
        "failure_reason", "error_message", "input_pdb", "chain", "gfp_segment_end_residue",
        "binder_start_residue", "gfp_complex_source_pdb", "gfp_binder_chain", "gfp_chain",
        "docked_pdb",
    }
    metric_names = [name for name in OUTPUT_FIELDS if name not in non_metric_fields]
    for metric_name in metric_names:
        for prefix, mode in ((f"min_{metric_name}", "min"), (f"max_{metric_name}", "max")):
            if prefix not in filter_config:
                continue
            threshold = float(filter_config[prefix])
            if threshold_failure(safe_float(row.get(metric_name, math.nan)), threshold, mode):
                failures.append(f"{metric_name}_{mode}_{threshold}")
    if failures:
        return False, ";".join(failures)
    return True, "none"


def blank_row(task: ScoringTask, args: argparse.Namespace) -> dict:
    row = {field: "" for field in OUTPUT_FIELDS}
    row.update(
        {
            "rank": 0,
            "task_id": task.task_id,
            "seq_id": task.seq_id,
            "register_offset": task.register_offset,
            "redesign_terminus": task.redesign_terminus,
            "status": "",
            "failure_reason": "",
            "error_message": "",
            "input_pdb": str(task.input_pdb),
            "chain": task.chain,
            "gfp_segment_end_residue": task.gfp_segment_end_residue,
            "binder_start_residue": task.binder_start_residue,
            "gfp_binder_chain": args.gfp_binder_chain,
            "gfp_chain": args.gfp_chain,
        }
    )
    return row


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.ca_clash_distance <= 0:
        raise ValueError("--ca_clash_distance must be greater than zero.")
    if args.max_tasks < 0:
        raise ValueError("--max_tasks must be non-negative.")

    repo_root = Path(args.repo_root).expanduser().resolve()
    results_root = resolve_path(repo_root, args.results_root)
    run_dir = results_root / args.run_name
    docked_dir = run_dir / "docked"
    failure_dir = run_dir / "rejected" / "scoring_failure"

    ensure_output_directory(run_dir, args.overwrite)
    if args.write_docked_pdb:
        docked_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    filter_config = load_filter_config(args, repo_root)

    gate_csv_path = resolve_path(repo_root, args.gate_csv)
    tasks = tasks_from_gate_csv(gate_csv_path, repo_root, args.gate_status_filter, args.gate_pdb_source)
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    # Load the reference GFP+GFP-binder complex ONCE -- it is the same for
    # every task, unlike the per-task gate structure.
    gfp_complex_path = resolve_path(repo_root, args.gfp_complex_pdb)
    gfp_complex_structure = gemmi.read_structure(str(gfp_complex_path))
    mobile_gfp_binder_residues = chain_protein_residues(gfp_complex_structure, args.gfp_binder_chain)
    mobile_gfp_residues = chain_protein_residues(gfp_complex_structure, args.gfp_chain)
    mobile_gfp_binder_ca = ca_array(mobile_gfp_binder_residues)
    mobile_gfp_ca = ca_array(mobile_gfp_residues)

    run_config = {
        **vars(args),
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "filter_config": filter_config,
        "number_of_tasks": len(tasks),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    print(f"Output directory:      {run_dir}")
    print(f"Number of tasks:       {len(tasks)}")
    print(f"GFP-binder residues:   {len(mobile_gfp_binder_residues)}")
    print(f"GFP residues:          {len(mobile_gfp_residues)}")
    print(f"Filter profile:        {args.filter_profile}")

    rows: list[dict] = []

    all_scores = run_dir / "score_all.csv"
    success_scores = run_dir / "score_success.csv"
    incremental = all_scores.open("w", newline="")
    incremental_writer = csv.DictWriter(incremental, fieldnames=OUTPUT_FIELDS)
    incremental_writer.writeheader()
    incremental.flush()

    for task_index, task in enumerate(tasks, start=1):
        row = blank_row(task, args)
        row["gfp_complex_source_pdb"] = str(gfp_complex_path)

        try:
            print(f"[{task_index:>4}/{len(tasks)}] Docking {task.task_id}")

            if not task.input_pdb.is_file():
                raise FileNotFoundError(f"Gate structure not found: {task.input_pdb}")

            fused_structure = gemmi.read_structure(str(task.input_pdb))
            fused_residues = chain_protein_residues(fused_structure, task.chain)
            is_gfp, is_blocking, is_binder = segment_masks(
                fused_residues, task.gfp_segment_end_residue, task.binder_start_residue, task.redesign_terminus
            )
            if not np.any(is_gfp):
                raise ValueError(f"gfp_segment_end_residue/binder_start_residue select no GFP-segment residues.")

            target_gfp_residues = [r for r, m in zip(fused_residues, is_gfp) if m]
            target_gfp_ca = ca_array(target_gfp_residues)
            blocking_ca = ca_array([r for r, m in zip(fused_residues, is_blocking) if m])
            binder_ca = ca_array([r for r, m in zip(fused_residues, is_binder) if m])

            if len(target_gfp_residues) != len(mobile_gfp_binder_residues):
                raise ValueError(
                    f"GFP-segment length mismatch: fused chain has "
                    f"{len(target_gfp_residues)} GFP-segment residues, "
                    f"--gfp_complex_pdb chain {args.gfp_binder_chain!r} has "
                    f"{len(mobile_gfp_binder_residues)} -- is this really the "
                    "same GFP-binder PDB/chain used for build_gfp_fusion_csv.py?"
                )

            row["num_gfp_segment_residues"] = len(target_gfp_residues)

            rotation, translation = kabsch_rotation_translation(mobile_gfp_binder_ca, target_gfp_ca)
            aligned_gfp_binder_ca = apply_transform(mobile_gfp_binder_ca, rotation, translation)
            row["gfp_binder_alignment_rmsd"] = float(
                np.sqrt(np.mean(np.sum((aligned_gfp_binder_ca - target_gfp_ca) ** 2, axis=1)))
            )

            aligned_gfp_ca = apply_transform(mobile_gfp_ca, rotation, translation)

            gb_clashes, gb_min = ca_clash_metrics(aligned_gfp_ca, blocking_ca, args.ca_clash_distance)
            row["GFP_Blocking_CAClashes"] = gb_clashes
            row["GFP_Blocking_MinCADistance"] = gb_min

            gbi_clashes, gbi_min = ca_clash_metrics(aligned_gfp_ca, binder_ca, args.ca_clash_distance)
            row["GFP_Binder_CAClashes"] = gbi_clashes
            row["GFP_Binder_MinCADistance"] = gbi_min

            combined_ca = np.concatenate([blocking_ca, binder_ca], axis=0) if len(blocking_ca) or len(binder_ca) else blocking_ca
            gc_clashes, gc_min = ca_clash_metrics(aligned_gfp_ca, combined_ca, args.ca_clash_distance)
            row["GFP_BlockingOrBinder_CAClashes"] = gc_clashes
            row["GFP_BlockingOrBinder_MinCADistance"] = gc_min

            if args.write_docked_pdb:
                docked_path = docked_dir / f"{task.task_id}_gfp_docked.pdb"
                docked_structure = gemmi.Structure()
                docked_structure.name = f"{task.task_id}_gfp_docked"
                docked_model = gemmi.Model("1")
                fused_chain_copy = gemmi.Chain(task.chain)
                for residue in fused_residues:
                    fused_chain_copy.add_residue(residue)
                docked_model.add_chain(fused_chain_copy)
                docked_model.add_chain(build_transformed_chain(mobile_gfp_residues, rotation, translation, args.gfp_chain_out))
                docked_structure.add_model(docked_model)
                docked_structure.setup_entities()
                docked_structure.write_pdb(str(docked_path))
                row["docked_pdb"] = str(docked_path)

            success, reason = evaluate_filters(row, filter_config)
            row["status"] = "success" if success else "rejected"
            row["failure_reason"] = "none" if success else reason
            row["error_message"] = ""

            print(
                f"[{task_index:>4}/{len(tasks)}] {task.task_id}: {row['status']}; "
                f"GFP_Blocking_CAClashes={gb_clashes}; GFP_Binder_CAClashes={gbi_clashes}; "
                f"alignment_rmsd={row['gfp_binder_alignment_rmsd']:.3f}"
            )

        except Exception as error:
            if args.fail_fast:
                raise
            row["status"] = "rejected"
            row["failure_reason"] = "scoring_failure"
            row["error_message"] = f"{type(error).__name__}: {error}"
            try:
                if task.input_pdb.is_file():
                    shutil.copy2(task.input_pdb, failure_dir / task.input_pdb.name)
            except Exception:
                pass
            print(f"[{task_index:>4}/{len(tasks)}] {task.task_id}: scoring failure; {type(error).__name__}: {error}")

        rows.append(row)
        incremental_writer.writerow(row)
        incremental.flush()

    incremental.close()

    def sort_value(row: dict, key: str, default: float) -> float:
        value = safe_float(row.get(key, math.nan))
        return default if math.isnan(value) else value

    # Higher GFP_Blocking_CAClashes ranks BETTER here -- inverted vs. every
    # other script in this pipeline (see module docstring).
    rows.sort(
        key=lambda item: (
            item["status"] != "success",
            -sort_value(item, "GFP_Blocking_CAClashes", -math.inf),
            sort_value(item, "gfp_binder_alignment_rmsd", math.inf),
            item["task_id"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    write_csv(all_scores, rows)
    write_csv(success_scores, [row for row in rows if row["status"] == "success"])

    print("Done.")
    print(f"Success:    {sum(r['status'] == 'success' for r in rows)}")
    print(f"Rejected:   {sum(r['status'] == 'rejected' for r in rows)}")
    print(f"All scores: {all_scores}")
    print(f"Success:    {success_scores}")


if __name__ == "__main__":
    main()
