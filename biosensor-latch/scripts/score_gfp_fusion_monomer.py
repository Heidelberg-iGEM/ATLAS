#!/usr/bin/env python3
"""
Monomer gate for GFP-binder fusion constructs (3-segment chain), scoring the
output of build_gfp_fusion_csv.py.

Place this file next to score_latch_monomer.py / score_latch_ternary.py /
build_gfp_fusion_csv.py, e.g.:
    flexcraft/pipelines/dual_binder_pipeline/scripts/score_gfp_fusion_monomer.py

This is the Test-1 analog for the 3-segment chain built by
build_gfp_fusion_csv.py (GFP binder + blocking peptide, +/- register offset,
+ binder, no linkers). Unlike score_latch_monomer.py (2 segments, 1
boundary), this chain has 2 boundaries (gfp_segment_end_residue,
binder_start_residue) and 3 logical segments -- which physical segment is
"gfp" vs. "binder" depends on redesign_terminus (read per-row from the
--sequence_csv's own redesign_terminus column, same convention
build_gfp_fusion_csv.py has written since 2026-09-16/17, default "n" for
older CSVs without the column; or --redesign_terminus for --input_dir):
    redesign_terminus "n" (original):
        - "gfp segment":      residue_number <= gfp_segment_end_residue
        - "blocking segment":  gfp_segment_end_residue < residue_number < binder_start_residue
        - "binder segment":    residue_number >= binder_start_residue
    redesign_terminus "c" (added 2026-09-17, for C-terminal latches such as
    Repeat3-4 of GvpC -- build_gfp_fusion_csv.py attaches
    the GFP binder after the blocking peptide instead of before it, so the
    two boundary fields' meaning flips):
        - "binder segment":    residue_number <= gfp_segment_end_residue
        - "blocking segment":  gfp_segment_end_residue < residue_number < binder_start_residue
        - "gfp segment":      residue_number >= binder_start_residue

IMPORTANT -- filler-aware Blocking_RMSD: for positive register offsets, the
blocking segment includes num_filler_residues placeholder residues (no real
Test-1 geometry, see build_gfp_fusion_csv.py), sitting directly next to
wherever the GFP binder attaches -- right after gfp_segment_end_residue for
redesign_terminus "n", right before binder_start_residue for "c".
Blocking_RMSD is computed only over the REAL blocking residues
(is_blocking_real), excluding the filler -- otherwise the metric would be
corrupted by comparing predicted coordinates against meaningless placeholder
input coordinates.

NOTE: score_gfp_fusion_partners.py (Test 2/3/4, downstream of this gate)
does NOT yet read redesign_terminus as of 2026-09-17 and assumes the "n"
segment order -- see that script for the analogous update still needed
before a "c" run's score_success.csv can be used there correctly.

Only designs that pass this gate should be scored further with
score_gfp_fusion_partners.py (Test 2/3/4: GvpC, GFP, or both).

Custom filter JSON example:
---------------------------

{
  "min_pLDDT": 0.75,
  "min_pTM": 0.50,
  "max_Blocking_RMSD": 3.0,
  "max_Binder_RMSD": 1.5,
  "min_Predicted_HotspotContactCount": 1,
  "max_GFPSegment_Binder_CAClashes": 0
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
import jax
import jax.numpy as jnp
import numpy as np

from flexcraft.data.data import DesignData
from flexcraft.structure.af import (
    AFInput,
    AFResult,
    get_model_haiku_params,
    make_af2,
    make_predict,
    model_config,
)
from flexcraft.structure.metrics import RMSD
from flexcraft.utils import load_pdb


FILTER_PROFILES = {
    "no_filter": {},

    "default": {
        "min_pLDDT": 0.70,
        "min_pTM": 0.45,
        "max_Blocking_RMSD": 3.0,
        "max_Binder_RMSD": 1.5,
        "min_InterfaceOverlap_input_to_predicted": 0.25,
        "min_Predicted_HotspotContactCount": 1,
        "min_Input_HotspotContactCount": 1,
        "max_GFPSegment_Binder_CAClashes": 0,
    },

    # Looser first-pass screen -- mainly for ranking rather than hard selection.
    "permissive": {
        "min_pLDDT": 0.55,
        "min_pTM": 0.30,
        "max_Blocking_RMSD": 6.0,
        "max_Binder_RMSD": 3.0,
        "max_GFPSegment_Binder_CAClashes": 5,
    },
}


BASE_SCORE_FIELDS = [
    "rank",
    "task_id",
    "source_test1_task_id",
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
    "num_filler_residues",
    "num_gfp_segment_residues",
    "num_blocking_residues",
    "num_binder_residues",
    "num_blocking_real_residues_compared",
    "hotspots_binder_local",
    "hotspots_chain_local",
    "full_sequence",
    "predicted_pdb",
    "pLDDT",
    "GFPSegment_pLDDT",
    "Blocking_pLDDT",
    "Binder_pLDDT",
    "pTM",
    "Blocking_RMSD",
    "Binder_RMSD",
    "GFPSegment_Blocking_CAClashes",
    "GFPSegment_Blocking_MinCADistance",
    "GFPSegment_Binder_CAClashes",
    "GFPSegment_Binder_MinCADistance",
    "Blocking_Binder_CAClashes",
    "Blocking_Binder_MinCADistance",
    "Input_InterfaceResidues",
    "Predicted_InterfaceResidues",
    "SharedInterfaceResidues",
    "InterfaceOverlap_input_to_predicted",
    "InterfaceOverlap_predicted_to_input",
    "Input_HotspotContactCount",
    "Input_HotspotMinCADistance",
    "Predicted_HotspotContactCount",
    "Predicted_HotspotMinCADistance",
]

SCORE_FIELDS = BASE_SCORE_FIELDS


@dataclass(frozen=True)
class ResidueRecord:
    global_index: int
    selected_index: int | None
    chain: str
    chain_position: int
    residue_number: int
    insertion_code: str
    residue_name: str


@dataclass(frozen=True)
class ScoringTask:
    task_id: str
    source_test1_task_id: str
    seq_id: str
    input_pdb: Path
    input_source: str
    gfp_segment_end_residue: int
    binder_start_residue: int
    num_filler_residues: int
    redesign_terminus: str
    register_offset: str
    full_sequence: str | None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Monomer gate for GFP-binder fusion constructs (3-segment chain: "
            "GFP binder + linker + blocking peptide + linker + binder) using AF2."
        )
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--sequence_csv",
        default=None,
        help="gfp_fusion_constructs.csv output from build_gfp_fusion_csv.py.",
    )
    input_group.add_argument(
        "--input_dir",
        default=None,
        help="Directory containing 3-segment construct PDB files, all sharing the SAME boundaries.",
    )

    parser.add_argument("--run_name", required=True, help="Output subfolder name inside --results_root.")
    parser.add_argument("--repo_root", default=".", help="FlexCraft repository root.")
    parser.add_argument(
        "--results_root",
        default="flexcraft/pipelines/dual_binder_pipeline/results",
        help="Pipeline results directory relative to --repo_root.",
    )
    parser.add_argument("--glob", default="*.pdb", help="Input file glob pattern for --input_dir. Default: *.pdb")
    parser.add_argument("--chain", default="A", help="Chain ID to score. Default: A")

    parser.add_argument(
        "--gfp_segment_end_residue",
        type=int,
        default=None,
        help="Last residue number of the GFP-binder+linker segment. Required with --input_dir.",
    )
    parser.add_argument(
        "--binder_start_residue",
        type=int,
        default=None,
        help="First residue number of the fixed binder segment. Required with --input_dir.",
    )
    parser.add_argument(
        "--num_filler_residues",
        type=int,
        default=0,
        help="Number of blocking-segment residues with no real Test-1 geometry, excluded from Blocking_RMSD. Only used with --input_dir. Default: 0",
    )
    parser.add_argument(
        "--redesign_terminus",
        choices=["n", "c"],
        default="n",
        help=(
            "Which side of the two boundaries the GFP-binder segment is on: "
            "'n' (default) GFP segment is N-terminal (residue_number <= "
            "gfp_segment_end_residue), binder segment is C-terminal "
            "(residue_number >= binder_start_residue) -- original behavior, "
            "matches build_gfp_fusion_csv.py's redesign_terminus 'n'. 'c' "
            "the binder segment is N-terminal instead and the GFP segment is "
            "C-terminal (build_gfp_fusion_csv.py's redesign_terminus 'c', "
            "for latches such as Repeat3-4 of GvpC). Only "
            "used with --input_dir (applied to every file) -- with "
            "--sequence_csv this is read per row from the redesign_terminus "
            "column instead (falls back to 'n' if that column is "
            "missing/empty, e.g. older CSVs)."
        ),
    )

    parser.add_argument(
        "--sequence_csv_status_filter",
        default="success",
        help="When using --sequence_csv, only score rows with this status. Use 'all' for every row. Default: success",
    )
    parser.add_argument(
        "--apply_sequence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Thread the full_sequence column onto the chain before folding (should already match the structure). Default: enabled.",
    )

    parser.add_argument("--af2_params", default="params/af/", help="Path to AF2 parameter directory.")
    parser.add_argument("--af_model", type=int, default=1, help="AF2 model number to run. Default: 1")
    parser.add_argument("--num_recycle", type=int, default=4, help="Number of AF2 recycles. Default: 4")
    parser.add_argument(
        "--fix_binder_template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Template the binder segment (its fold is already known/fixed). Default: enabled.",
    )
    parser.add_argument(
        "--use_guess",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the input structure as an AF2 initial guess. Default: disabled.",
    )

    parser.add_argument("--interface_cutoff", type=float, default=8.0, help="CA distance cutoff for interface overlap. Default: 8.0 A")
    parser.add_argument("--ca_clash_distance", type=float, default=3.5, help="CA distance below which clashes are counted. Default: 3.5 A")
    parser.add_argument(
        "--hotspots",
        default="none",
        help="Binder-local hotspot residues (position 1 = first binder-segment residue), e.g. 40-55,61. Use 'none' to disable.",
    )
    parser.add_argument("--hotspot_contact_cutoff", type=float, default=8.0, help="CA distance cutoff for hotspot contact. Default: 8.0 A")

    parser.add_argument(
        "--filter_profile",
        choices=list(FILTER_PROFILES) + ["custom_json"],
        default="default",
        help="Named filter profile. Use custom_json together with --filter_json.",
    )
    parser.add_argument("--filter_json", default=None, help="Optional JSON file with threshold overrides.")

    parser.add_argument("--max_tasks", type=int, default=0, help="Maximum number of tasks to score. Use 0 for all.")
    parser.add_argument("--overwrite", action="store_true", help="Delete and replace an existing non-empty output directory.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop at the first failed task instead of continuing.")

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


def normalize_icode(value: object) -> str:
    text = str(value or " ")
    if not text or text == "\x00":
        return " "
    return text[0]


def is_protein_residue(residue: gemmi.Residue) -> bool:
    return bool(gemmi.find_tabulated_residue(residue.name).is_amino_acid())


def collect_residue_records(path: Path, chain_id: str) -> list[ResidueRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"Structure file not found: {path}")
    structure = gemmi.read_structure(str(path))
    if len(structure) == 0:
        raise ValueError(f"Structure contains no models: {path}")

    records: list[ResidueRecord] = []
    global_index = 0
    for chain in structure[0]:
        chain_position = 0
        for residue in chain:
            if not is_protein_residue(residue):
                continue
            ca = residue.find_atom("CA", "*")
            if ca is None or ca.occ == 0.0:
                continue
            chain_position += 1
            if chain.name == chain_id:
                records.append(
                    ResidueRecord(
                        global_index=global_index,
                        selected_index=None,
                        chain=chain.name,
                        chain_position=chain_position,
                        residue_number=int(residue.seqid.num),
                        insertion_code=normalize_icode(residue.seqid.icode),
                        residue_name=residue.name.strip().upper(),
                    )
                )
            global_index += 1

    if not records:
        raise ValueError(f"Chain {chain_id!r} not found (or empty) in {path}")
    return records


def select_chain_records(records: Sequence[ResidueRecord]) -> list[ResidueRecord]:
    return [
        ResidueRecord(
            global_index=record.global_index,
            selected_index=index,
            chain=record.chain,
            chain_position=record.chain_position,
            residue_number=record.residue_number,
            insertion_code=record.insertion_code,
            residue_name=record.residue_name,
        )
        for index, record in enumerate(records)
    ]


def aa_code_index(sequence: str) -> np.ndarray:
    from flexcraft.sequence.aa_codes import AF2_CODE

    sequence = sequence.strip().upper()
    try:
        return np.array([AF2_CODE.index(aa) for aa in sequence], dtype=np.int32)
    except ValueError as error:
        raise ValueError(f"Sequence contains residue not supported by AF2_CODE {AF2_CODE!r}: {sequence}") from error


def design_sequence(design: DesignData, records: Sequence[ResidueRecord]) -> str:
    from flexcraft.sequence.aa_codes import AF2_CODE

    aa = np.array(design["aa"], dtype=np.int32)
    return "".join(AF2_CODE[int(aa[int(record.selected_index)])] for record in records)


def apply_sequence_override(design: DesignData, records: Sequence[ResidueRecord], full_sequence: str) -> DesignData:
    if len(full_sequence) != len(records):
        raise ValueError(f"Sequence length mismatch: sequence={len(full_sequence)}, residues={len(records)}")
    aa = np.array(design["aa"], dtype=np.int32)
    indices = [int(record.selected_index) for record in records]
    aa[indices] = aa_code_index(full_sequence)
    return design.update(aa=jnp.asarray(aa, dtype=jnp.int32))


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def parse_hotspot_ranges(text: str) -> list[int]:
    if text is None or text.lower() == "none" or not text.strip():
        return []
    result: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", item)
        if not match:
            raise ValueError(f"Could not parse hotspot specification: {item!r}")
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if end < start:
            raise ValueError(f"Invalid hotspot range: {item!r}")
        result.extend(range(start, end + 1))
    return result


def read_scoring_csv_rows(csv_path: Path):
    """Tolerant reader -- see the identical helper in score_latch_ternary.py /
    build_gfp_fusion_csv.py for the rationale (spreadsheet re-save padding)."""
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            raw_header = next(reader)
        except StopIteration:
            return
        fieldnames = [name.strip() for name in raw_header]
        for row_number, raw_row in enumerate(reader, start=2):
            if len(raw_row) != len(fieldnames):
                raise ValueError(
                    f"{csv_path}, line {row_number}: {len(raw_row)} fields, header has {len(fieldnames)}."
                )
            yield {key: (value.strip() if isinstance(value, str) else value) for key, value in zip(fieldnames, raw_row)}


def task_from_pdb(
    path: Path,
    gfp_segment_end_residue: int,
    binder_start_residue: int,
    num_filler_residues: int,
    redesign_terminus: str,
) -> ScoringTask:
    task_id = safe_id(path.stem)
    return ScoringTask(
        task_id=task_id,
        source_test1_task_id="",
        seq_id="",
        input_pdb=path,
        input_source="pdb",
        gfp_segment_end_residue=gfp_segment_end_residue,
        binder_start_residue=binder_start_residue,
        num_filler_residues=num_filler_residues,
        redesign_terminus=redesign_terminus,
        register_offset="",
        full_sequence=None,
    )


def tasks_from_input_dir(
    input_dir: Path,
    glob: str,
    gfp_segment_end_residue: int | None,
    binder_start_residue: int | None,
    num_filler_residues: int,
    redesign_terminus: str,
) -> list[ScoringTask]:
    if gfp_segment_end_residue is None or binder_start_residue is None:
        raise ValueError("--gfp_segment_end_residue and --binder_start_residue are required with --input_dir.")
    paths = sorted(path for path in input_dir.glob(glob) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No files matched {glob!r} in {input_dir}")
    return [
        task_from_pdb(path, gfp_segment_end_residue, binder_start_residue, num_filler_residues, redesign_terminus)
        for path in paths
    ]


def tasks_from_sequence_csv(csv_path: Path, repo_root: Path, status_filter: str) -> list[ScoringTask]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Sequence CSV not found: {csv_path}")

    tasks: list[ScoringTask] = []
    for row in read_scoring_csv_rows(csv_path):
        if status_filter != "all" and row.get("status") != status_filter:
            continue

        input_pdb_text = row.get("input_pdb", "")
        if not input_pdb_text:
            continue
        input_pdb = Path(input_pdb_text).expanduser()
        if not input_pdb.is_absolute():
            input_pdb = repo_root / input_pdb

        gfp_end_text = (row.get("gfp_segment_end_residue") or "").strip()
        binder_start_text = (row.get("binder_start_residue") or "").strip()
        if not gfp_end_text or not binder_start_text:
            raise ValueError(
                f"Row {row.get('task_id')!r} is missing gfp_segment_end_residue/binder_start_residue -- "
                "is this really a build_gfp_fusion_csv.py output CSV?"
            )

        # Default "n" for older build_gfp_fusion_csv.py CSVs written before its
        # 2026-09-17 redesign_terminus update -- they are all N-terminal fusions.
        redesign_terminus = (row.get("redesign_terminus") or "n").strip().lower()
        if redesign_terminus not in ("n", "c"):
            raise ValueError(
                f"Invalid redesign_terminus {redesign_terminus!r} for row "
                f"{row.get('task_id')!r} in {csv_path} -- must be 'n' or 'c'."
            )

        task_id = safe_id(row.get("task_id") or input_pdb.stem)
        tasks.append(
            ScoringTask(
                task_id=task_id,
                source_test1_task_id=row.get("source_test1_task_id", ""),
                seq_id=str(row.get("seq_id", "")),
                input_pdb=input_pdb.resolve(),
                input_source="sequence_csv",
                gfp_segment_end_residue=int(gfp_end_text),
                binder_start_residue=int(binder_start_text),
                num_filler_residues=int((row.get("num_filler_residues") or "0").strip() or 0),
                redesign_terminus=redesign_terminus,
                register_offset=str(row.get("register_offset", "")),
                full_sequence=row.get("full_sequence") or None,
            )
        )

    if not tasks:
        raise ValueError(f"No scoreable rows found in {csv_path}")
    return tasks


def make_key_generator(seed: int):
    key = jax.random.PRNGKey(seed)

    def next_key():
        nonlocal key
        key, subkey = jax.random.split(key)
        return subkey

    return next_key


def make_af2_predictor(num_recycle: int):
    af2_config = model_config("model_1_ptm")
    af2_config.model.global_config.use_dgram = False
    return jax.jit(make_predict(make_af2(af2_config), num_recycle=num_recycle))


def build_af_input(design: DesignData, is_binder: jnp.ndarray, fix_binder_template: bool, use_guess: bool) -> AFInput:
    af_input = AFInput.from_data(design)
    if fix_binder_template:
        af_input = af_input.add_template(design, where=is_binder)
    if use_guess:
        af_input = af_input.add_guess(design)
    return af_input


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def ca_clash_metrics(data: DesignData, mask_a: np.ndarray, mask_b: np.ndarray, threshold: float) -> tuple[int, float]:
    ca = np.asarray(data["atom_positions"][:, 1], dtype=np.float64)
    valid = np.asarray(data["atom_mask"][:, 1], dtype=bool)
    a = ca[mask_a & valid]
    b = ca[mask_b & valid]
    if len(a) == 0 or len(b) == 0:
        return 0, math.nan
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return int(np.sum(distances < threshold)), float(np.min(distances))


def interface_indices(data: DesignData, probe_mask: np.ndarray, anchor_mask: np.ndarray, cutoff: float) -> set[int]:
    ca = np.asarray(data["atom_positions"][:, 1], dtype=np.float64)
    valid = np.asarray(data["atom_mask"][:, 1], dtype=bool)
    probe_idx = np.where(probe_mask & valid)[0]
    anchor_idx = np.where(anchor_mask & valid)[0]
    if len(probe_idx) == 0 or len(anchor_idx) == 0:
        return set()
    distances = np.linalg.norm(ca[probe_idx][:, None, :] - ca[anchor_idx][None, :, :], axis=-1)
    contacting = np.where((distances < cutoff).any(axis=1))[0]
    return set(int(probe_idx[index]) for index in contacting)


def hotspot_metrics(
    data: DesignData, hotspot_indices: Sequence[int], contactor_mask: np.ndarray, cutoff: float
) -> tuple[int, float]:
    if not hotspot_indices:
        return 0, math.nan
    ca = np.asarray(data["atom_positions"][:, 1], dtype=np.float64)
    valid = np.asarray(data["atom_mask"][:, 1], dtype=bool)
    hotspot_idx = [index for index in hotspot_indices if valid[index]]
    contactor_idx = np.where(contactor_mask & valid)[0]
    if not hotspot_idx or len(contactor_idx) == 0:
        return 0, math.nan
    distances = np.linalg.norm(ca[hotspot_idx][:, None, :] - ca[contactor_idx][None, :, :], axis=-1)
    return int(np.sum((distances < cutoff).any(axis=1))), float(np.min(distances))


def masked_rmsd(predicted: DesignData, reference: DesignData, mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    mask_j = jnp.asarray(mask)
    predicted_sub = predicted[mask_j]
    reference_sub = reference[mask_j]
    return safe_float(RMSD()(predicted_sub, reference_sub, reference_sub["batch_index"], reference_sub["mask"]))


def af2_confidence_metrics(af_result: AFResult, is_gfp: np.ndarray, is_blocking: np.ndarray, is_binder: np.ndarray) -> dict[str, float]:
    def masked_mean(mask: np.ndarray) -> float:
        mask_j = jnp.asarray(mask)
        return safe_float((mask_j * af_result.plddt).sum() / jnp.maximum(mask_j.sum(), 1))

    return {
        "pLDDT": safe_float(af_result.plddt.mean()),
        "GFPSegment_pLDDT": masked_mean(is_gfp),
        "Blocking_pLDDT": masked_mean(is_blocking),
        "Binder_pLDDT": masked_mean(is_binder),
        "pTM": safe_float(af_result.ptm),
    }


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
    if not filter_config:
        return True, "none"

    failures = []
    thresholds: list[tuple[str, str, str]] = [
        ("min_pLDDT", "pLDDT", "min"),
        ("min_pTM", "pTM", "min"),
        ("max_Blocking_RMSD", "Blocking_RMSD", "max"),
        ("max_Binder_RMSD", "Binder_RMSD", "max"),
        ("min_InterfaceOverlap_input_to_predicted", "InterfaceOverlap_input_to_predicted", "min"),
        ("max_GFPSegment_Blocking_CAClashes", "GFPSegment_Blocking_CAClashes", "max"),
        ("max_GFPSegment_Binder_CAClashes", "GFPSegment_Binder_CAClashes", "max"),
        ("min_Input_HotspotContactCount", "Input_HotspotContactCount", "min"),
        ("min_Predicted_HotspotContactCount", "Predicted_HotspotContactCount", "min"),
    ]
    for threshold_key, metric_name, mode in thresholds:
        if threshold_key not in filter_config:
            continue
        threshold = float(filter_config[threshold_key])
        if threshold_failure(safe_float(row.get(metric_name, math.nan)), threshold, mode):
            failures.append(f"{metric_name}_{mode}_{threshold}")

    if failures:
        return False, ";".join(failures)
    return True, "none"


def blank_row(task: ScoringTask, args: argparse.Namespace, hotspots_local: Sequence[int]) -> dict:
    row = {field: "" for field in SCORE_FIELDS}
    # Rough early estimate for the report-only hotspots_chain_local field. Only
    # accurate for redesign_terminus="n", where the binder segment starts
    # exactly at binder_start_residue -- for "c" the binder segment is the
    # chain's N-terminal prefix instead, whose first residue we don't know
    # until the structure is loaded (same caveat as score_latch_monomer.py).
    # The real value is computed and written back into the row once records
    # are available (see main loop); this is just a placeholder for failure
    # rows that never get that far.
    if task.redesign_terminus == "n":
        hotspots_chain_local_estimate = ",".join(
            str(h) for h in sorted(local + task.binder_start_residue - 1 for local in hotspots_local)
        )
    else:
        hotspots_chain_local_estimate = ""
    row.update(
        {
            "rank": 0,
            "task_id": task.task_id,
            "source_test1_task_id": task.source_test1_task_id,
            "seq_id": task.seq_id,
            "register_offset": task.register_offset,
            "redesign_terminus": task.redesign_terminus,
            "status": "",
            "failure_reason": "",
            "error_message": "",
            "input_pdb": str(task.input_pdb),
            "chain": args.chain,
            "gfp_segment_end_residue": task.gfp_segment_end_residue,
            "binder_start_residue": task.binder_start_residue,
            "num_filler_residues": task.num_filler_residues,
            "hotspots_binder_local": args.hotspots,
            "hotspots_chain_local": hotspots_chain_local_estimate,
        }
    )
    return row


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def copy_or_write_success_pdb(row: dict, task: ScoringTask, success_dir: Path) -> None:
    source = row.get("predicted_pdb")
    if not source:
        return
    source_path = Path(str(source))
    if source_path.is_file():
        shutil.copy2(source_path, success_dir / f"{task.task_id}_predicted_success.pdb")


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.interface_cutoff <= 0:
        raise ValueError("--interface_cutoff must be greater than zero.")
    if args.ca_clash_distance <= 0:
        raise ValueError("--ca_clash_distance must be greater than zero.")
    if args.hotspot_contact_cutoff <= 0:
        raise ValueError("--hotspot_contact_cutoff must be greater than zero.")
    if args.num_recycle < 0:
        raise ValueError("--num_recycle must be non-negative.")
    if args.max_tasks < 0:
        raise ValueError("--max_tasks must be zero or greater.")

    repo_root = Path(args.repo_root).expanduser().resolve()
    results_root = resolve_path(repo_root, args.results_root)
    run_dir = results_root / args.run_name

    predicted_dir = run_dir / "predicted"
    success_dir = run_dir / "success"
    rejected_dir = run_dir / "rejected"
    failure_dir = rejected_dir / "scoring_failure"

    ensure_output_directory(run_dir, args.overwrite)
    predicted_dir.mkdir(parents=True, exist_ok=True)
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    af2_params_path = resolve_path(repo_root, args.af2_params)
    hotspots_local = parse_hotspot_ranges(args.hotspots)
    filter_config = load_filter_config(args, repo_root)

    if args.sequence_csv:
        csv_path = resolve_path(repo_root, args.sequence_csv)
        tasks = tasks_from_sequence_csv(csv_path=csv_path, repo_root=repo_root, status_filter=args.sequence_csv_status_filter)
    else:
        input_dir = resolve_path(repo_root, args.input_dir)
        tasks = tasks_from_input_dir(
            input_dir,
            args.glob,
            args.gfp_segment_end_residue,
            args.binder_start_residue,
            args.num_filler_residues,
            args.redesign_terminus,
        )

    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    run_config = {
        **vars(args),
        "repo_root": str(repo_root),
        "results_root": str(results_root),
        "run_dir": str(run_dir),
        "af2_params": str(af2_params_path),
        "filter_config": filter_config,
        "number_of_tasks": len(tasks),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    (run_dir / "filter_config.json").write_text(json.dumps(filter_config, indent=2) + "\n")

    print(f"Output directory: {run_dir}")
    print(f"Number of tasks:  {len(tasks)}")
    print(f"Chain:            {args.chain}")
    print(f"AF2 model:        {args.af_model}")
    print(f"Filter profile:   {args.filter_profile}")

    print("Loading AF2 model parameters...")
    af2_predict = make_af2_predictor(args.num_recycle)
    af_params = get_model_haiku_params(model_name=f"model_{args.af_model}_ptm", data_dir=str(af2_params_path), fuse=True)

    next_key = make_key_generator(37)
    rows: list[dict] = []

    # Write score_all.csv incrementally (arrival order, rank still 0) and flush
    # after every task, so a crashed / cancelled AF2 run leaves a valid partial
    # CSV and progress can be followed live. It is re-written sorted + ranked
    # after the loop, and score_success.csv is written once there.
    all_scores = run_dir / "score_all.csv"
    success_scores = run_dir / "score_success.csv"
    incremental = all_scores.open("w", newline="")
    incremental_writer = csv.DictWriter(incremental, fieldnames=SCORE_FIELDS)
    incremental_writer.writeheader()
    incremental.flush()

    for task_index, task in enumerate(tasks, start=1):
        row = blank_row(task, args, hotspots_local)

        try:
            print(f"[{task_index:>4}/{len(tasks)}] Scoring {task.task_id}")

            if not task.input_pdb.is_file():
                raise FileNotFoundError(f"Input structure not found: {task.input_pdb}")

            all_records = collect_residue_records(task.input_pdb, args.chain)
            records = select_chain_records(all_records)
            selected_indices = np.array([record.global_index for record in records], dtype=np.int32)

            full_design = load_pdb(str(task.input_pdb))
            design = full_design[selected_indices].untie()

            residue_numbers = np.array([record.residue_number for record in records], dtype=np.int64)
            # head/tail split by the two boundaries -- which one is the GFP
            # segment and which is the (existing, unchanged) binder segment
            # depends on redesign_terminus (mirrors build_gfp_fusion_csv.py's
            # own gfp_segment_end_residue/binder_start_residue field-meaning
            # flip): "n" GFP segment leads (head), binder segment trails
            # (tail) -- original behavior. "c" binder segment leads (head),
            # GFP segment trails (tail) -- e.g. Repeat3-4 of
            # GvpC.
            head_mask = residue_numbers <= task.gfp_segment_end_residue
            tail_mask = residue_numbers >= task.binder_start_residue
            if task.redesign_terminus == "n":
                is_gfp, is_binder = head_mask, tail_mask
            else:
                is_binder, is_gfp = head_mask, tail_mask
            is_blocking = (~is_gfp) & (~is_binder)

            if not np.any(is_binder):
                raise ValueError(
                    f"binder_start_residue={task.binder_start_residue} "
                    f"(redesign_terminus={task.redesign_terminus!r}) selects no residues."
                )
            if not np.any(is_blocking):
                raise ValueError("No blocking-segment residues left between the two boundaries.")
            if not np.any(is_gfp):
                raise ValueError(
                    f"gfp_segment_end_residue={task.gfp_segment_end_residue} "
                    f"(redesign_terminus={task.redesign_terminus!r}) selects no residues."
                )

            # Filler residues (positive register_offset) sit directly next to
            # where the GFP binder attaches -- right after the GFP segment
            # for "n", right before it for "c" (see build_gfp_fusion_csv.py).
            if task.redesign_terminus == "n":
                filler_cutoff = task.gfp_segment_end_residue + task.num_filler_residues
                is_blocking_filler = is_blocking & (residue_numbers <= filler_cutoff)
            else:
                filler_cutoff = task.binder_start_residue - task.num_filler_residues
                is_blocking_filler = is_blocking & (residue_numbers >= filler_cutoff)
            is_blocking_real = is_blocking & (~is_blocking_filler)

            row["num_gfp_segment_residues"] = int(np.sum(is_gfp))
            row["num_blocking_residues"] = int(np.sum(is_blocking))
            row["num_binder_residues"] = int(np.sum(is_binder))
            row["num_blocking_real_residues_compared"] = int(np.sum(is_blocking_real))

            if args.apply_sequence and task.full_sequence:
                design = apply_sequence_override(design, records, task.full_sequence)
            row["full_sequence"] = design_sequence(design, records)

            # Chain-local number of the binder segment's own first residue --
            # this is what "position 1" in --hotspots (binder-local) is
            # counted from. Derived from is_binder (already terminus-correct
            # above) rather than assumed, same rationale as
            # score_latch_monomer.py: for "n" this is binder_start_residue,
            # for "c" it's the chain's own first residue.
            binder_first_residue = int(residue_numbers[is_binder].min())
            hotspots_chain_local = [local + binder_first_residue - 1 for local in hotspots_local]
            row["hotspots_chain_local"] = ",".join(str(h) for h in sorted(hotspots_chain_local))
            hotspot_indices = [
                int(record.selected_index) for record in records if record.residue_number in set(hotspots_chain_local)
            ]

            input_hotspot_contacts, input_hotspot_min = hotspot_metrics(design, hotspot_indices, is_blocking, args.hotspot_contact_cutoff)
            row["Input_HotspotContactCount"] = input_hotspot_contacts
            row["Input_HotspotMinCADistance"] = input_hotspot_min
            input_interface = interface_indices(design, is_blocking, is_binder, args.interface_cutoff)

            is_binder_j = jnp.asarray(is_binder)
            af_input = build_af_input(design=design, is_binder=is_binder_j, fix_binder_template=args.fix_binder_template, use_guess=args.use_guess)

            af_result: AFResult = af2_predict(af_params, next_key(), af_input)
            predicted_data = af_result.to_data()

            predicted_pdb = predicted_dir / f"{task.task_id}_model_{args.af_model}.pdb"
            predicted_data.save_pdb(str(predicted_pdb))
            row["predicted_pdb"] = str(predicted_pdb)

            row.update(af2_confidence_metrics(af_result, is_gfp, is_blocking, is_binder))

            row["Binder_RMSD"] = masked_rmsd(predicted_data, design, is_binder)
            row["Blocking_RMSD"] = masked_rmsd(predicted_data, design, is_blocking_real)

            gb_clashes, gb_min = ca_clash_metrics(predicted_data, is_gfp, is_blocking, args.ca_clash_distance)
            row["GFPSegment_Blocking_CAClashes"] = gb_clashes
            row["GFPSegment_Blocking_MinCADistance"] = gb_min

            gbi_clashes, gbi_min = ca_clash_metrics(predicted_data, is_gfp, is_binder, args.ca_clash_distance)
            row["GFPSegment_Binder_CAClashes"] = gbi_clashes
            row["GFPSegment_Binder_MinCADistance"] = gbi_min

            bb_clashes, bb_min = ca_clash_metrics(predicted_data, is_blocking, is_binder, args.ca_clash_distance)
            row["Blocking_Binder_CAClashes"] = bb_clashes
            row["Blocking_Binder_MinCADistance"] = bb_min

            predicted_interface = interface_indices(predicted_data, is_blocking, is_binder, args.interface_cutoff)
            shared_interface = input_interface & predicted_interface
            row["Input_InterfaceResidues"] = len(input_interface)
            row["Predicted_InterfaceResidues"] = len(predicted_interface)
            row["SharedInterfaceResidues"] = len(shared_interface)
            row["InterfaceOverlap_input_to_predicted"] = len(shared_interface) / max(1, len(input_interface))
            row["InterfaceOverlap_predicted_to_input"] = len(shared_interface) / max(1, len(predicted_interface))

            predicted_hotspot_contacts, predicted_hotspot_min = hotspot_metrics(
                predicted_data, hotspot_indices, is_blocking, args.hotspot_contact_cutoff
            )
            row["Predicted_HotspotContactCount"] = predicted_hotspot_contacts
            row["Predicted_HotspotMinCADistance"] = predicted_hotspot_min

            success, reason = evaluate_filters(row, filter_config)
            row["status"] = "success" if success else "rejected"
            row["failure_reason"] = "none" if success else reason
            row["error_message"] = ""

            if success:
                copy_or_write_success_pdb(row, task, success_dir)

            print(
                f"[{task_index:>4}/{len(tasks)}] {task.task_id}: {row['status']}; "
                f"pLDDT={row.get('pLDDT', '')}; Blocking_RMSD={row.get('Blocking_RMSD', '')}; "
                f"Predicted_HotspotContactCount={row.get('Predicted_HotspotContactCount', '')}; "
                f"reason={row['failure_reason']}"
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

    rows.sort(
        key=lambda item: (
            item["status"] != "success",
            -sort_value(item, "Predicted_HotspotContactCount", -math.inf),
            -sort_value(item, "pLDDT", -math.inf),
            sort_value(item, "Blocking_RMSD", math.inf),
            item["task_id"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    # Re-write score_all.csv sorted + ranked (overwrites the incremental version),
    # then score_success.csv once.
    write_csv(all_scores, rows)
    write_csv(success_scores, [row for row in rows if row["status"] == "success"])

    success_count = sum(row["status"] == "success" for row in rows)
    rejected_count = sum(row["status"] == "rejected" for row in rows)

    print("Done.")
    print(f"Success:       {success_count}")
    print(f"Rejected:      {rejected_count}")
    print(f"All scores:    {all_scores}")
    print(f"Success CSV:   {success_scores}")
    print(f"Predicted PDB: {predicted_dir}")


if __name__ == "__main__":
    main()
