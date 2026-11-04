#!/usr/bin/env python3
"""
Score GFP-binder-fusion constructs with 1 or 2 co-folded partners (GvpC
and/or GFP), generalizing score_latch_ternary.py's binder+blocking+GvpC
pattern to the 3-segment [GFP binder][blocking][binder] chain built by
build_gfp_fusion_csv.py and validated by score_gfp_fusion_monomer.py.

Place this file next to score_latch_monomer.py / score_latch_ternary.py /
build_gfp_fusion_csv.py / score_gfp_fusion_monomer.py, e.g.:
    flexcraft/pipelines/dual_binder_pipeline/scripts/score_gfp_fusion_partners.py

Test 2/3/4 (see Blocking-Peptide-Latch-Design.md, "GFP-Binder-Integration")
all run through this one script, distinguished only by which partner
arguments are supplied:
    Test 2: --partner1_* = GvpC, --partner1_target binder, no --partner2_*
            (GvpC should NOT bind)
    Test 3: --partner1_* = GFP, --partner1_target gfp, no --partner2_*
            (GFP should bind; Blocking_RMSD should be high -- the blocking
            peptide gets displaced)
    Test 4: --partner1_* = GvpC (target binder) AND --partner2_* = GFP
            (target gfp) together (both should bind simultaneously)

Each partner is added as its own chain, built from a sequence-only
idealized placeholder backbone (never used as an AF2 template or guess --
see idealized_backbone_atoms()/append_placeholder_partner_chain() below),
exactly like the GvpC placeholder chain in score_latch_ternary.py.

Reference geometry: unlike score_gfp_fusion_monomer.py (whose blocking
segment may include placeholder filler residues with no real geometry),
the reference here is that script's own AF2-folded output (--gate_pdb_source
predicted, default) -- a real fold for every residue, no filler-exclusion
logic needed for RMSD.

One-time per-partner reference complexes: for each partner, once per run,
predict [that partner's target segment alone] + [that partner] (no other
segments) to get a "normal binding" baseline for PositionalDisplacement /
InterfaceOverlap -- generalizes the single GvpC reference complex in
score_latch_ternary.py to whichever segment ("gfp" or "binder") each partner
targets.

CSV columns are deliberately generic (Partner1_*/Partner2_*) rather than
baking each partner's name into the column name -- partner1_identity /
partner2_identity (e.g. "GvpC"/"GFP") record which partner was actually used,
so column names stay identical across Test 2/3/4 runs.

Which physical segment (leading/"head", numbered <= gfp_segment_end_residue,
vs. trailing/"tail", numbered >= binder_start_residue) is actually the GFP
segment vs. the binder segment depends on redesign_terminus, read per-row
from the --gate_csv's own redesign_terminus column (written by
score_gfp_fusion_monomer.py since 2026-09-17, default "n" for older CSVs
without it): "n" (original) GFP segment leads, binder segment trails. "c"
(added 2026-09-17, for C-terminal latches such as Repeat3-4 in
GvpC-Cage-Latch-Design) binder segment leads, GFP segment trails. This
affects both the live is_gfp/is_binder masks used for every task and the
one-time per-partner reference complex (build_partner_reference()), which
now derives which residues belong to the "binder"/"gfp" target from the
same head/tail split instead of assuming "n".

ASSUMPTIONS (please verify before trusting results):
------------------------------------------------------------
1. --partner1_hotspots / --partner2_hotspots are given in the LOCAL numbering
   of whichever segment that partner targets (position 1 = first residue of
   the gfp segment if --partnerN_target gfp, or of the binder segment if
   --partnerN_target binder) -- same convention as --hotspots in
   score_latch_monomer.py/score_latch_ternary.py, just parameterized by
   target segment instead of being hardcoded to the binder segment. The
   chain-global residue number that "position 1" maps to is derived from the
   already terminus-correct target_mask (min residue number where the mask
   is true), not assumed from gfp_segment_end_residue/binder_start_residue
   directly -- works for both redesign_terminus values without a separate
   formula per case.
2. GFP-binder hotspot residues (which residues in the gfp segment actually
   contact GFP) are NOT determined by this script -- they must be supplied
   externally via --partner1_hotspots/--partner2_hotspots. See the
   accompanying note's "Known open points".
3. Test 4's known AF2/AF3 difficulty with multiple simultaneous competing
   binding events (see Blocking-Peptide-Latch-Design.md) is NOT mitigated
   inside this script -- run it multiple times with different --seed values
   and compare, rather than trusting a single Test-4 prediction.

Custom filter JSON example (Test 2, GvpC should not bind):
-----------------------------------------------------------

{
  "min_pLDDT": 0.70,
  "max_Partner1_HotspotContactCount": 0,
  "max_Partner1_InterfaceOverlap": 0.20,
  "min_Partner1_PositionalDisplacement": 3.0
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

    # Deliberately weak/generic -- direction of "success" for partner
    # metrics depends on the test (Test 2 wants low contact, Test 3/4 want
    # high contact), so no partner-metric thresholds are baked in here.
    # Use --filter_json for real Test 2/3/4 thresholds (see examples in the
    # accompanying note).
    "permissive_generic": {
        "min_pLDDT": 0.55,
        "min_pTM": 0.30,
    },
}

BASE_SCORE_FIELDS = [
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
    "full_sequence",
    "predicted_pdb",
    "relaxed_pdb",
    "rosetta_relax_status",
    "pLDDT",
    "GFPSegment_pLDDT",
    "Blocking_pLDDT",
    "Binder_pLDDT",
    "pTM",
    "i_pTM",
    "pAE",
    "i_pAE",
    "GFPSegment_RMSD",
    "Blocking_RMSD",
    "Binder_RMSD",
    "partner1_identity",
    "partner1_target",
    "Partner1_pLDDT",
    "Partner1_HotspotContactCount",
    "Partner1_HotspotMinCADistance",
    "Partner1_TargetSegment_CAClashes",
    "Partner1_TargetSegment_MinCADistance",
    "Partner1_BlockingSegment_CAClashes",
    "Partner1_BlockingSegment_MinCADistance",
    "Partner1_PositionalDisplacement",
    "Partner1_InterfaceOverlap",
    "partner2_identity",
    "partner2_target",
    "Partner2_pLDDT",
    "Partner2_HotspotContactCount",
    "Partner2_HotspotMinCADistance",
    "Partner2_TargetSegment_CAClashes",
    "Partner2_TargetSegment_MinCADistance",
    "Partner2_BlockingSegment_CAClashes",
    "Partner2_BlockingSegment_MinCADistance",
    "Partner2_PositionalDisplacement",
    "Partner2_InterfaceOverlap",
    "Binder_Energy_Score",
    "Surface_Hydrophobicity",
    "ShapeComplementarity",
    "PackStat",
    "dG",
    "dSASA",
    "dG/dSASA",
    "Interface_SASA_%",
    "Interface_Hydrophobicity",
    "n_InterfaceResidues",
    "n_InterfaceHbonds",
    "InterfaceHbondsPercentage",
    "n_InterfaceUnsatHbonds",
    "InterfaceUnsatHbondsPercentage",
]

SCORE_FIELDS = BASE_SCORE_FIELDS

AMINO_ACID_3LETTER = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


@dataclass(frozen=True)
class ResidueRecord:
    global_index: int
    selected_index: int | None
    chain: str
    chain_position: int
    residue_number: int
    residue_name: str


@dataclass(frozen=True)
class ScoringTask:
    task_id: str
    seq_id: str
    register_offset: str
    input_pdb: Path
    chain: str
    gfp_segment_end_residue: int
    binder_start_residue: int
    redesign_terminus: str
    full_sequence: str = ""


@dataclass(frozen=True)
class PartnerSpec:
    index: int  # 1 or 2
    sequence: str
    chain_id: str
    label: str
    target: str  # "gfp" or "binder"
    hotspots_local: tuple[int, ...]
    compute_reference: bool

    @property
    def prefix(self) -> str:
        return f"Partner{self.index}"


@dataclass(frozen=True)
class PartnerReference:
    target_segment_ca: np.ndarray
    partner_ca: np.ndarray
    interface_target_residue_numbers: set[int]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score GFP-binder-fusion constructs (3-segment chain) with 1 or 2 "
            "co-folded partners (GvpC and/or GFP) using AF2. Test 2/3/4 all "
            "run through this script, distinguished by --partner1_*/--partner2_*."
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
        help="Which gate structure to use as the starting point/RMSD reference. Default: predicted",
    )
    parser.add_argument("--chain", default="A", help="Chain ID of the fused chain in the gate output. Default: A")

    for n in (1, 2):
        required = n == 1
        parser.add_argument(f"--partner{n}_sequence", default=None, required=required if n == 1 else False)
        parser.add_argument(f"--partner{n}_chain", default="G" if n == 1 else "H")
        parser.add_argument(f"--partner{n}_label", default=f"Partner{n}")
        parser.add_argument(f"--partner{n}_target", choices=["gfp", "binder"], default=None)
        parser.add_argument(f"--partner{n}_hotspots", default="none")
        parser.add_argument(
            f"--partner{n}_compute_reference",
            action=argparse.BooleanOptionalAction,
            default=True,
        )

    parser.add_argument("--run_name", required=True, help="Output subfolder name inside --results_root.")
    parser.add_argument("--repo_root", default=".", help="FlexCraft repository root.")
    parser.add_argument(
        "--results_root",
        default="flexcraft/pipelines/dual_binder_pipeline/results",
        help="Pipeline results directory relative to --repo_root.",
    )

    parser.add_argument("--af2_params", default="params/af/", help="Path to AF2 parameter directory.")
    parser.add_argument("--af_model", type=int, default=1, help="AF2 model number to run. Default: 1")
    parser.add_argument("--num_recycle", type=int, default=4, help="Number of AF2 recycles. Default: 4")
    parser.add_argument("--seed", type=int, default=37, help="PRNG seed. Vary this across repeated Test-4 runs.")
    parser.add_argument(
        "--fix_binder_template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Template the binder segment only. Default: enabled.",
    )
    parser.add_argument(
        "--use_guess",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the (fused-chain part of the) gate structure as an AF2 initial guess. Default: disabled.",
    )

    parser.add_argument("--run_rosetta_relax", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--relax_confidence_cutoff", type=float, default=0.35)
    parser.add_argument("--alphaball_path", default="dependencies/DAlphaBall.gcc")

    parser.add_argument("--interface_cutoff", type=float, default=8.0)
    parser.add_argument("--ca_clash_distance", type=float, default=3.5)
    parser.add_argument("--hotspot_contact_cutoff", type=float, default=8.0)

    parser.add_argument("--filter_profile", choices=list(FILTER_PROFILES) + ["custom_json"], default="no_filter")
    parser.add_argument("--filter_json", default=None)

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


def idealized_backbone_atoms(index: int) -> list[tuple[str, tuple[float, float, float]]]:
    """Same idealized, non-physical placeholder as score_latch_ternary.py /
    build_gfp_fusion_csv.py -- never used as an AF2 template or guess."""
    x = index * 3.8
    z = 1.2 if index % 2 == 0 else -1.2
    return [
        ("N", (x - 0.9, 0.4, z)),
        ("CA", (x, 0.0, z)),
        ("C", (x + 0.9, 0.4, z)),
        ("O", (x + 0.9, 1.6, z)),
    ]


def append_placeholder_partner_chain(structure: gemmi.Structure, sequence: str, chain_id: str) -> None:
    model = structure[0]
    if any(chain.name == chain_id for chain in model):
        raise ValueError(f"Chain ID {chain_id!r} already used in the input structure.")
    sequence = sequence.strip().upper()
    if not sequence:
        raise ValueError("Partner sequence must not be empty.")

    chain = gemmi.Chain(chain_id)
    for index, letter in enumerate(sequence):
        resname = AMINO_ACID_3LETTER.get(letter)
        if resname is None:
            raise ValueError(f"Unsupported amino acid code {letter!r} in partner sequence.")
        residue = gemmi.Residue()
        residue.name = resname
        residue.seqid = gemmi.SeqId(index + 1, " ")
        residue.het_flag = "A"
        for atom_name, position in idealized_backbone_atoms(index):
            atom = gemmi.Atom()
            atom.name = atom_name
            atom.element = gemmi.Element(atom_name[0])
            atom.pos = gemmi.Position(*position)
            atom.occ = 1.0
            atom.b_iso = 50.0
            residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)


def build_combined_pdb(input_pdb: Path, partners: Sequence[PartnerSpec], out_path: Path) -> None:
    structure = gemmi.read_structure(str(input_pdb))
    if len(structure) == 0:
        raise ValueError(f"Structure contains no models: {input_pdb}")
    for partner in partners:
        append_placeholder_partner_chain(structure, partner.sequence, partner.chain_id)
    structure.setup_entities()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    structure.write_pdb(str(out_path))


def build_partner_reference_pdb(
    fused_pdb: Path,
    chain_id: str,
    target_mask_positions: set[int],
    partner: PartnerSpec,
    out_path: Path,
) -> None:
    """Isolate [partner's target segment] + fresh [partner] chain -- the
    one-time "normal binding" reference for this partner. Generalizes
    build_reference_input_pdb() in score_latch_ternary.py from a hardcoded
    binder segment to whichever segment this partner targets.

    target_mask_positions is a set of 1-based POSITIONAL indices (the Nth
    protein-with-CA residue of the chain), NOT residue.seqid.num -- AF2's
    own predicted_pdb output (the default --gate_pdb_source) is not
    guaranteed to be 1-indexed (same root cause as the 2026-09-01
    off-by-one bug in build_gfp_fusion_csv.py's predicted-PDB handling), so
    filtering by seqid.num here would silently pick the wrong residues by
    exactly one for --gate_pdb_source predicted. Position is counted here
    identically to collect_residue_records()'s chain_position, so it stays
    consistent with how target_mask_positions was derived in
    build_partner_reference()."""
    source = gemmi.read_structure(str(fused_pdb))
    source_model = source[0]

    reference = gemmi.Structure()
    reference.name = f"reference_{partner.label}"
    reference_model = gemmi.Model("1")
    target_chain = gemmi.Chain(chain_id)

    for chain in source_model:
        if chain.name != chain_id:
            continue
        position = 0
        for residue in chain:
            if not is_protein_residue(residue):
                continue
            ca = residue.find_atom("CA", "*")
            if ca is None or ca.occ == 0.0:
                continue
            position += 1
            if position not in target_mask_positions:
                continue
            target_chain.add_residue(residue)

    if len(target_chain) == 0:
        raise ValueError(f"No target-segment residues found while building the {partner.label} reference complex.")

    reference_model.add_chain(target_chain)
    reference.add_model(reference_model)
    append_placeholder_partner_chain(reference, partner.sequence, partner.chain_id)
    reference.setup_entities()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reference.write_pdb(str(out_path))


def collect_residue_records(path: Path) -> list[ResidueRecord]:
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
            records.append(
                ResidueRecord(
                    global_index=global_index,
                    selected_index=None,
                    chain=chain.name,
                    chain_position=chain_position,
                    residue_number=int(residue.seqid.num),
                    residue_name=residue.name.strip().upper(),
                )
            )
            global_index += 1
    if not records:
        raise ValueError(f"No protein residues found in {path}")
    return records


def select_records(records: Sequence[ResidueRecord]) -> list[ResidueRecord]:
    return [
        ResidueRecord(
            global_index=r.global_index,
            selected_index=i,
            chain=r.chain,
            chain_position=r.chain_position,
            residue_number=r.residue_number,
            residue_name=r.residue_name,
        )
        for i, r in enumerate(records)
    ]


def parse_hotspot_ranges(text: str) -> tuple[int, ...]:
    if text is None or text.lower() == "none" or not text.strip():
        return ()
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
        result.extend(range(start, end + 1))
    return tuple(result)


def read_scoring_csv_rows(csv_path: Path):
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
                input_pdb=input_pdb.resolve(),
                chain=(row.get("chain") or "A").strip() or "A",
                gfp_segment_end_residue=int(gfp_end),
                binder_start_residue=int(binder_start),
                redesign_terminus=redesign_terminus,
                full_sequence=(row.get("full_sequence") or "").strip(),
            )
        )

    if not tasks:
        raise ValueError(f"No usable rows found in {csv_path}")
    return tasks


def make_partner_specs(args: argparse.Namespace) -> list[PartnerSpec]:
    specs: list[PartnerSpec] = []
    for n in (1, 2):
        sequence = getattr(args, f"partner{n}_sequence")
        if not sequence:
            if n == 1:
                raise ValueError("--partner1_sequence is required.")
            continue
        target = getattr(args, f"partner{n}_target")
        if target is None:
            raise ValueError(f"--partner{n}_target is required when --partner{n}_sequence is given.")
        specs.append(
            PartnerSpec(
                index=n,
                sequence=sequence.strip().upper(),
                chain_id=getattr(args, f"partner{n}_chain"),
                label=getattr(args, f"partner{n}_label"),
                target=target,
                hotspots_local=parse_hotspot_ranges(getattr(args, f"partner{n}_hotspots")),
                compute_reference=getattr(args, f"partner{n}_compute_reference"),
            )
        )
    chain_ids = [args.chain] + [spec.chain_id for spec in specs]
    if len(set(chain_ids)) != len(chain_ids):
        raise ValueError(f"Duplicate chain IDs: {chain_ids}")
    return specs


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


def build_af_input(design: DesignData, template_mask: jnp.ndarray, num_chains: int, use_guess: bool) -> AFInput:
    af_input = AFInput.from_data(design)
    af_input = af_input.add_template(design, where=template_mask)
    af_input = af_input.block_diagonal(num_sequences=num_chains)
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
    return set(int(probe_idx[i]) for i in contacting)


def hotspot_contact_metrics(data: DesignData, hotspot_indices: Sequence[int], contactor_mask: np.ndarray, cutoff: float) -> tuple[int, float]:
    if not hotspot_indices:
        return 0, math.nan
    ca = np.asarray(data["atom_positions"][:, 1], dtype=np.float64)
    valid = np.asarray(data["atom_mask"][:, 1], dtype=bool)
    hotspot_idx = [i for i in hotspot_indices if valid[i]]
    contactor_idx = np.where(contactor_mask & valid)[0]
    if not hotspot_idx or len(contactor_idx) == 0:
        return 0, math.nan
    distances = np.linalg.norm(ca[hotspot_idx][:, None, :] - ca[contactor_idx][None, :, :], axis=-1)
    return int(np.sum((distances < cutoff).any(axis=1))), float(np.min(distances))


def extract_ca(data: DesignData, mask: np.ndarray) -> np.ndarray:
    ca = np.asarray(data["atom_positions"][:, 1], dtype=np.float64)
    valid = np.asarray(data["atom_mask"][:, 1], dtype=bool)
    return ca[mask & valid]


def kabsch_rotation_translation(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, d])
    rotation = vt.T @ correction @ u.T
    translation = target_center - rotation @ mobile_center
    return rotation, translation


def positional_displacement(predicted_target_ca: np.ndarray, predicted_partner_ca: np.ndarray, reference: PartnerReference) -> float:
    if len(predicted_target_ca) != len(reference.target_segment_ca):
        raise ValueError("Target-segment length mismatch between prediction and reference complex.")
    if len(predicted_partner_ca) != len(reference.partner_ca):
        raise ValueError("Partner length mismatch between prediction and reference complex.")
    rotation, translation = kabsch_rotation_translation(predicted_target_ca, reference.target_segment_ca)
    aligned_partner = predicted_partner_ca @ rotation.T + translation
    return float(np.sqrt(np.mean(np.sum((aligned_partner - reference.partner_ca) ** 2, axis=1))))


def masked_rmsd(predicted: DesignData, reference: DesignData, mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    mask_j = jnp.asarray(mask)
    predicted_sub = predicted[mask_j]
    reference_sub = reference[mask_j]
    return safe_float(RMSD()(predicted_sub, reference_sub, reference_sub["batch_index"], reference_sub["mask"]))


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
    generic_metric_names = [name for name in SCORE_FIELDS if name not in {
        "rank", "task_id", "seq_id", "register_offset", "redesign_terminus", "status", "failure_reason",
        "error_message", "input_pdb", "chain", "gfp_segment_end_residue", "binder_start_residue",
        "predicted_pdb", "relaxed_pdb", "rosetta_relax_status", "partner1_identity", "partner1_target",
        "partner2_identity", "partner2_target",
    }]
    for metric_name in generic_metric_names:
        for prefix, mode in ((f"min_{metric_name}", "min"), (f"max_{metric_name}", "max")):
            if prefix not in filter_config:
                continue
            threshold = float(filter_config[prefix])
            if threshold_failure(safe_float(row.get(metric_name, math.nan)), threshold, mode):
                failures.append(f"{metric_name}_{mode}_{threshold}")
    if failures:
        return False, ";".join(failures)
    return True, "none"


def blank_row(task: ScoringTask, args: argparse.Namespace, partners: Sequence[PartnerSpec]) -> dict:
    row = {field: "" for field in SCORE_FIELDS}
    row.update(
        {
            "rank": 0,
            "task_id": task.task_id,
            "seq_id": task.seq_id,
            "register_offset": task.register_offset,
            "status": "",
            "failure_reason": "",
            "error_message": "",
            "input_pdb": str(task.input_pdb),
            "chain": task.chain,
            "gfp_segment_end_residue": task.gfp_segment_end_residue,
            "binder_start_residue": task.binder_start_residue,
            "redesign_terminus": task.redesign_terminus,
            "full_sequence": task.full_sequence,
        }
    )
    for partner in partners:
        row[f"partner{partner.index}_identity"] = partner.label
        row[f"partner{partner.index}_target"] = partner.target
    return row


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_partner_reference(
    partner: PartnerSpec,
    first_task: ScoringTask,
    args: argparse.Namespace,
    af_params,
    af2_predict,
    next_key,
    run_dir: Path,
) -> PartnerReference:
    reference_dir = run_dir / "reference" / partner.label
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_pdb = reference_dir / "reference_input.pdb"

    all_records = select_records(collect_residue_records(first_task.input_pdb))
    # Which physical segment ("head", positions <= gfp_segment_end_residue, vs.
    # "tail", positions >= binder_start_residue) is the GFP segment vs. the
    # binder segment depends on redesign_terminus -- same field-meaning flip
    # as build_gfp_fusion_csv.py/score_gfp_fusion_monomer.py: "n" GFP=head,
    # binder=tail (original); "c" binder=head, GFP=tail (e.g. Repeat3-4 in
    # GvpC-Cage-Latch-Design). Split by chain_position (1-based positional
    # index), NOT r.residue_number/seqid.num -- first_task.input_pdb is
    # AF2's own predicted_pdb by default (--gate_pdb_source predicted),
    # which is not guaranteed to be 1-indexed (same root cause as the
    # 2026-09-01 off-by-one bug in build_gfp_fusion_csv.py).
    head_positions = {r.chain_position for r in all_records if r.chain_position <= first_task.gfp_segment_end_residue}
    tail_positions = {r.chain_position for r in all_records if r.chain_position >= first_task.binder_start_residue}
    if first_task.redesign_terminus == "n":
        gfp_positions, binder_positions = head_positions, tail_positions
    else:
        binder_positions, gfp_positions = head_positions, tail_positions
    target_positions = binder_positions if partner.target == "binder" else gfp_positions

    build_partner_reference_pdb(
        fused_pdb=first_task.input_pdb,
        chain_id=first_task.chain,
        target_mask_positions=target_positions,
        partner=partner,
        out_path=reference_pdb,
    )

    records = select_records(collect_residue_records(reference_pdb))
    selected_indices = np.array([r.global_index for r in records], dtype=np.int32)
    design = load_pdb(str(reference_pdb))[selected_indices].untie()

    is_partner = np.array([r.chain == partner.chain_id for r in records], dtype=bool)
    is_target = ~is_partner

    af_input = build_af_input(
        design=design,
        template_mask=jnp.asarray(is_target) if args.fix_binder_template and partner.target == "binder" else jnp.zeros(len(records), dtype=bool),
        num_chains=2,
        use_guess=False,
    )
    af_result: AFResult = af2_predict(af_params, next_key(), af_input)
    predicted_data = af_result.to_data()
    predicted_pdb = reference_dir / "reference_predicted.pdb"
    predicted_data.save_pdb(str(predicted_pdb))

    target_ca = extract_ca(predicted_data, is_target)
    partner_ca = extract_ca(predicted_data, is_partner)
    interface_idx = interface_indices(predicted_data, is_target, is_partner, args.interface_cutoff)
    interface_residue_numbers = {records[i].residue_number for i in interface_idx}

    print(
        f"Reference complex ({partner.label}, target={partner.target}): "
        f"target residues={int(np.sum(is_target))}, partner residues={int(np.sum(is_partner))}, "
        f"pLDDT={safe_float(af_result.plddt.mean()):.3f}, interface residues={len(interface_residue_numbers)}"
    )

    return PartnerReference(
        target_segment_ca=target_ca,
        partner_ca=partner_ca,
        interface_target_residue_numbers=interface_residue_numbers,
    )


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.interface_cutoff <= 0 or args.ca_clash_distance <= 0 or args.hotspot_contact_cutoff <= 0:
        raise ValueError("Cutoff arguments must be greater than zero.")
    if args.num_recycle < 0 or args.max_tasks < 0:
        raise ValueError("--num_recycle/--max_tasks must be non-negative.")

    partners = make_partner_specs(args)

    repo_root = Path(args.repo_root).expanduser().resolve()
    results_root = resolve_path(repo_root, args.results_root)
    run_dir = results_root / args.run_name

    predicted_dir = run_dir / "predicted"
    relaxed_dir = run_dir / "relaxed"
    success_dir = run_dir / "success"
    rejected_dir = run_dir / "rejected"
    failure_dir = rejected_dir / "scoring_failure"

    ensure_output_directory(run_dir, args.overwrite)
    for d in (predicted_dir, relaxed_dir, success_dir, failure_dir):
        d.mkdir(parents=True, exist_ok=True)

    af2_params_path = resolve_path(repo_root, args.af2_params)
    alphaball_path = resolve_path(repo_root, args.alphaball_path)
    filter_config = load_filter_config(args, repo_root)

    gate_csv_path = resolve_path(repo_root, args.gate_csv)
    tasks = tasks_from_gate_csv(gate_csv_path, repo_root, args.gate_status_filter, args.gate_pdb_source)
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    run_config = {
        **vars(args),
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "partners": [p.label for p in partners],
        "filter_config": filter_config,
        "number_of_tasks": len(tasks),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    print(f"Output directory: {run_dir}")
    print(f"Number of tasks:  {len(tasks)}")
    print(f"Partners:         {[(p.label, p.target) for p in partners]}")

    if args.run_rosetta_relax and not alphaball_path.is_file():
        raise FileNotFoundError(f"DAlphaBall executable not found: {alphaball_path}")
    if args.run_rosetta_relax:
        import pyrosetta as pr

        pr.init(
            "-ignore_unrecognized_res -ignore_zero_occupancy -mute all "
            f"-holes:dalphaball {alphaball_path} -corrections::beta_nov16 true -relax:default_repeats 1"
        )

    print("Loading AF2 model parameters...")
    af2_predict = make_af2_predictor(args.num_recycle)
    af_params = get_model_haiku_params(model_name=f"model_{args.af_model}_ptm", data_dir=str(af2_params_path), fuse=True)
    next_key = make_key_generator(args.seed)

    partner_references: dict[int, PartnerReference] = {}
    for partner in partners:
        if not partner.compute_reference:
            continue
        print(f"Predicting one-time reference complex for {partner.label} (target={partner.target})...")
        partner_references[partner.index] = build_partner_reference(
            partner, tasks[0], args, af_params, af2_predict, next_key, run_dir
        )

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
        row = blank_row(task, args, partners)
        try:
            print(f"[{task_index:>4}/{len(tasks)}] Scoring {task.task_id}")

            if not task.input_pdb.is_file():
                raise FileNotFoundError(f"Input structure not found: {task.input_pdb}")

            combined_pdb = predicted_dir / f"{task.task_id}_input_with_partners.pdb"
            build_combined_pdb(task.input_pdb, partners, combined_pdb)

            all_records = collect_residue_records(combined_pdb)
            records = select_records(all_records)
            selected_indices = np.array([r.global_index for r in records], dtype=np.int32)
            design = load_pdb(str(combined_pdb))[selected_indices].untie()

            residue_numbers = np.array([r.residue_number for r in records], dtype=np.int64)
            # 1-based positional index (Nth protein-with-CA residue of its
            # own chain), NOT r.residue_number/seqid.num -- combined_pdb's
            # fused-chain part is task.input_pdb, AF2's own predicted_pdb by
            # default (--gate_pdb_source predicted), which is not guaranteed
            # to be 1-indexed (same root cause as the 2026-09-01 off-by-one
            # bug in build_gfp_fusion_csv.py). residue_numbers itself is
            # still used below for cross-structure interface-residue-identity
            # sets, which are self-consistent regardless of indexing origin.
            chain_positions = np.array([r.chain_position for r in records], dtype=np.int64)
            chains = np.array([r.chain for r in records])

            is_fused = chains == task.chain
            # head/tail split by the two boundaries -- which one is the GFP
            # segment and which is the (existing, unchanged) binder segment
            # depends on redesign_terminus, same convention as
            # score_gfp_fusion_monomer.py: "n" GFP segment leads (head),
            # binder segment trails (tail); "c" binder segment leads (head),
            # GFP segment trails (tail).
            head_mask = is_fused & (chain_positions <= task.gfp_segment_end_residue)
            tail_mask = is_fused & (chain_positions >= task.binder_start_residue)
            if task.redesign_terminus == "n":
                is_gfp, is_binder = head_mask, tail_mask
            else:
                is_binder, is_gfp = head_mask, tail_mask
            is_blocking = is_fused & (~is_gfp) & (~is_binder)

            partner_masks: dict[int, np.ndarray] = {}
            for partner in partners:
                partner_masks[partner.index] = chains == partner.chain_id

            gate_only = load_pdb(str(task.input_pdb))[
                np.array([r.global_index for r in records if r.chain == task.chain], dtype=np.int32)
            ].untie()

            template_mask = jnp.asarray(is_binder) if args.fix_binder_template else jnp.zeros(len(records), dtype=bool)
            af_input = build_af_input(design=design, template_mask=template_mask, num_chains=1 + len(partners), use_guess=args.use_guess)

            af_result: AFResult = af2_predict(af_params, next_key(), af_input)
            predicted_data = af_result.to_data()
            predicted_pdb = predicted_dir / f"{task.task_id}_model_{args.af_model}.pdb"
            predicted_data.save_pdb(str(predicted_pdb))
            row["predicted_pdb"] = str(predicted_pdb)

            def masked_mean(mask: np.ndarray) -> float:
                mask_j = jnp.asarray(mask)
                return safe_float((mask_j * af_result.plddt).sum() / jnp.maximum(mask_j.sum(), 1))

            row["pLDDT"] = safe_float(af_result.plddt.mean())
            row["GFPSegment_pLDDT"] = masked_mean(is_gfp)
            row["Blocking_pLDDT"] = masked_mean(is_blocking)
            row["Binder_pLDDT"] = masked_mean(is_binder)
            row["pTM"] = safe_float(af_result.ptm)
            row["i_pTM"] = safe_float(af_result.iptm)
            row["pAE"] = safe_float(af_result.pae.mean())
            row["i_pAE"] = safe_float(af_result.ipae)

            is_fused_j = jnp.asarray(is_fused)
            predicted_fused_only = predicted_data[is_fused_j]
            is_gfp_within_fused = is_gfp[is_fused]
            is_blocking_within_fused = is_blocking[is_fused]
            is_binder_within_fused = is_binder[is_fused]

            row["GFPSegment_RMSD"] = masked_rmsd(predicted_fused_only, gate_only, is_gfp_within_fused)
            row["Blocking_RMSD"] = masked_rmsd(predicted_fused_only, gate_only, is_blocking_within_fused)
            row["Binder_RMSD"] = masked_rmsd(predicted_fused_only, gate_only, is_binder_within_fused)

            for partner in partners:
                partner_mask = partner_masks[partner.index]
                target_mask = is_binder if partner.target == "binder" else is_gfp
                # Chain-local POSITION of the target segment's own first
                # residue -- what "position 1" in --partnerN_hotspots is
                # counted from. Derived from target_mask (already
                # terminus-correct above) rather than assumed, same
                # rationale as score_gfp_fusion_monomer.py: for
                # redesign_terminus "n" this is binder_start_residue
                # (target=binder) or 1 (target=gfp); for "c" it's the other
                # way around. Uses chain_positions, NOT residue_numbers --
                # see the off-by-one note on chain_positions above.
                offset = int(chain_positions[target_mask].min())
                hotspot_chain_positions = {local + offset - 1 for local in partner.hotspots_local}
                hotspot_indices = [
                    i for i, r in enumerate(records) if r.chain == task.chain and r.chain_position in hotspot_chain_positions
                ]

                row[f"Partner{partner.index}_pLDDT"] = masked_mean(partner_mask)

                contacts, contact_min = hotspot_contact_metrics(predicted_data, hotspot_indices, partner_mask, args.hotspot_contact_cutoff)
                row[f"Partner{partner.index}_HotspotContactCount"] = contacts
                row[f"Partner{partner.index}_HotspotMinCADistance"] = contact_min

                ts_clashes, ts_min = ca_clash_metrics(predicted_data, partner_mask, target_mask, args.ca_clash_distance)
                row[f"Partner{partner.index}_TargetSegment_CAClashes"] = ts_clashes
                row[f"Partner{partner.index}_TargetSegment_MinCADistance"] = ts_min

                bs_clashes, bs_min = ca_clash_metrics(predicted_data, partner_mask, is_blocking, args.ca_clash_distance)
                row[f"Partner{partner.index}_BlockingSegment_CAClashes"] = bs_clashes
                row[f"Partner{partner.index}_BlockingSegment_MinCADistance"] = bs_min

                reference = partner_references.get(partner.index)
                if reference is not None:
                    predicted_target_ca = extract_ca(predicted_data, target_mask)
                    predicted_partner_ca = extract_ca(predicted_data, partner_mask)
                    row[f"Partner{partner.index}_PositionalDisplacement"] = safe_float(
                        positional_displacement(predicted_target_ca, predicted_partner_ca, reference)
                    )
                    predicted_interface_idx = interface_indices(predicted_data, target_mask, partner_mask, args.interface_cutoff)
                    predicted_interface_residue_numbers = {records[i].residue_number for i in predicted_interface_idx}
                    shared = reference.interface_target_residue_numbers & predicted_interface_residue_numbers
                    row[f"Partner{partner.index}_InterfaceOverlap"] = len(shared) / max(1, len(reference.interface_target_residue_numbers))

            relaxed_pdb = ""
            row["rosetta_relax_status"] = "skipped"
            if args.run_rosetta_relax and safe_float(row.get("i_pAE", math.nan)) < args.relax_confidence_cutoff:
                try:
                    from flexcraft.rosetta.relax import fastrelax
                    from flexcraft.rosetta.interface_analyzer import score_interface

                    relaxed_path = relaxed_dir / f"{task.task_id}_model_{args.af_model}_relaxed.pdb"
                    relaxed_file = fastrelax(predicted_data, str(relaxed_path), tmpdir=str(relaxed_dir / "tmp"))
                    relaxed_pdb = str(relaxed_path)

                    is_any_partner = np.zeros(len(records), dtype=bool)
                    for mask in partner_masks.values():
                        is_any_partner = is_any_partner | mask
                    interface_scores, _ = score_interface(relaxed_file, jnp.asarray(is_any_partner))
                    row.update(
                        {
                            "Binder_Energy_Score": interface_scores.get("binder_score", math.nan),
                            "Surface_Hydrophobicity": interface_scores.get("surface_hydrophobicity", math.nan),
                            "ShapeComplementarity": interface_scores.get("interface_sc", math.nan),
                            "PackStat": interface_scores.get("interface_packstat", math.nan),
                            "dG": interface_scores.get("interface_dG", math.nan),
                            "dSASA": interface_scores.get("interface_dSASA", math.nan),
                            "dG/dSASA": interface_scores.get("interface_dG_SASA_ratio", math.nan),
                            "Interface_SASA_%": interface_scores.get("interface_fraction", math.nan),
                            "Interface_Hydrophobicity": interface_scores.get("interface_hydrophobicity", math.nan),
                            "n_InterfaceResidues": interface_scores.get("interface_nres", math.nan),
                            "n_InterfaceHbonds": interface_scores.get("interface_interface_hbonds", math.nan),
                            "InterfaceHbondsPercentage": interface_scores.get("interface_hbond_percentage", math.nan),
                            "n_InterfaceUnsatHbonds": interface_scores.get("interface_delta_unsat_hbonds", math.nan),
                            "InterfaceUnsatHbondsPercentage": interface_scores.get("interface_delta_unsat_hbonds_percentage", math.nan),
                        }
                    )
                    row["rosetta_relax_status"] = "ok"
                except Exception as rosetta_error:
                    row["rosetta_relax_status"] = f"failed: {type(rosetta_error).__name__}: {rosetta_error}"
            row["relaxed_pdb"] = relaxed_pdb

            success, reason = evaluate_filters(row, filter_config)
            row["status"] = "success" if success else "rejected"
            row["failure_reason"] = "none" if success else reason
            row["error_message"] = ""

            if success:
                shutil.copy2(predicted_pdb, success_dir / f"{task.task_id}_predicted_success.pdb")

            print(f"[{task_index:>4}/{len(tasks)}] {task.task_id}: {row['status']}; reason={row['failure_reason']}")

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

    rows.sort(key=lambda item: (item["status"] != "success", -sort_value(item, "pLDDT", -math.inf), item["task_id"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    # Re-write score_all.csv sorted + ranked (overwrites the incremental version),
    # then score_success.csv once.
    write_csv(all_scores, rows)
    write_csv(success_scores, [row for row in rows if row["status"] == "success"])

    print("Done.")
    print(f"Success:    {sum(r['status'] == 'success' for r in rows)}")
    print(f"Rejected:   {sum(r['status'] == 'rejected' for r in rows)}")
    print(f"All scores: {all_scores}")
    print(f"Success:    {success_scores}")


if __name__ == "__main__":
    main()
