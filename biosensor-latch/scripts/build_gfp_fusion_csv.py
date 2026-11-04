#!/usr/bin/env python3
"""
Build GFP-binder + blocking-peptide + binder fusion constructs, at several
helix register offsets, from score_latch_monomer.py (Test 1) output.

Place this file next to score_latch_monomer.py / score_latch_ternary.py, e.g.:
    flexcraft/pipelines/dual_binder_pipeline/scripts/build_gfp_fusion_csv.py

Background
----------
Blocking-Peptide-Latch-Design.md, "GFP-Binder-Integration" (2026-08-21) and
"GFP binder fusion pipeline -- architecture" (internal design note, 2026-08-26): the GFP binder is
attached directly (no linker) to the blocking peptide's free end. Whether
the GFP binder's binding face ends up pointing toward the binder depends on
the helical register at the attachment point -- not on any 3D docking,
since AF2 co-folds the whole construct from sequence alone (as already
established for the GvpC ternary test in score_latch_ternary.py). So
"trying different rotations" reduces to: vary the blocking peptide's length
by a few residues at its free end before attaching the GFP binder (an alpha
helix turns ~100 degrees/residue), and let AF2 decide what folds.

Which end of the blocking peptide is free depends on --redesign_terminus
(read per-row from the Test-1 CSV's own redesign_terminus column, same
convention score_latch_monomer.py has written since 2026-09-16, default
"n" for older CSVs without the column):
    - "n" (original architecture): the Test-1 chain runs blocking-peptide
      (free N-terminus) -> binder. The GFP binder attaches at the chain's
      N-terminus.
    - "c" (added 2026-09-17, for C-terminal latches such as Repeat3-4 in
      GvpC-Cage-Latch-Design): the Test-1 chain runs binder ->
      blocking-peptide (free C-terminus). The GFP binder attaches at the
      chain's C-terminus, directly after the blocking peptide.

This script does NOT run AF2. It only builds, for each Test-1 design and
each requested register offset, a new single-chain construct:

    redesign_terminus "n":
    [GFP binder][blocking peptide, N-terminally trimmed/extended by the
    register offset][existing linker+binder, unchanged from Test 1]

    redesign_terminus "c":
    [existing binder+linker, unchanged from Test 1][blocking peptide,
    C-terminally trimmed/extended by the register offset][GFP binder]

and writes a CSV that score_gfp_fusion_monomer.py (the Test-1-analog "does
this fold" gate for this 3-segment chain) can consume directly --
score_gfp_fusion_monomer.py reads redesign_terminus correctly as of
2026-09-17 (same update). NOTE: score_gfp_fusion_partners.py (downstream of
that gate, Test 2/3/4) does NOT yet read redesign_terminus and assumes the
"n" segment order (GFP segment first, binder segment last); see
OUTPUT_FIELDS comment below.

Register offsets and the extension filler
------------------------------------------
--register_offsets (default "-2,-1,0,1,2"): residues removed from (negative)
or added to (positive) the blocking peptide's free end (its N-terminus for
redesign_terminus "n", its C-terminus for "c"), immediately before/after
which the GFP binder attaches directly (no linker). Negative offsets just
drop existing residues (and their real Test-1 coordinates). Positive
offsets need residues that do not exist in the Test-1 backbone -- filled
with a generic helix-favoring filler amino acid (--filler_residue, default
'A'), not a redesigned or sequence-continued patch. This is deliberately
cheap: AF2 folds the whole construct freely in the downstream monomer-gate
step anyway (see score_gfp_fusion_monomer.py), so there is no point
spending a ProteinMPNN redesign pass before the first screen.

The GFP binder segment and any positive-offset filler residues are given
idealized, non-physical placeholder backbone coordinates (same trick as
append_placeholder_chain() in score_latch_ternary.py) -- they are never used
as an AF2 template, so their exact input geometry does not matter, only that
gemmi/load_pdb receive a syntactically valid structure.

IMPORTANT for downstream scoring: filler residues (positive offsets) have no
real Test-1 reference geometry. num_filler_residues is written to the output
CSV so score_gfp_fusion_monomer.py can exclude them from Blocking_RMSD (only
the real, Test-1-derived blocking residues are compared).

ASSUMPTIONS (please verify before trusting downstream results):
------------------------------------------------------------
1. binder_start_residue is a 1-BASED POSITIONAL index (the Nth
   protein-with-CA residue), carried through unchanged from
   score_latch_monomer.py / prepare_scoring_csv. It is sliced positionally,
   NOT compared against residue.seqid.num -- score_latch_monomer.py writes
   0-indexed predicted PDBs, and the old seqid comparison was off by one
   (bug 2026-09-01). Which side of binder_start_residue is the blocking
   peptide vs. the binder depends on redesign_terminus (see above and
   score_latch_monomer.py's own 2026-09-16 redesign_terminus update) --
   "n": blocking peptide is residues 1..binder_start_residue-1, binder is
   binder_start_residue..end. "c": binder is residues 1..
   binder_start_residue-1, blocking peptide is binder_start_residue..end.
2. The GFP binder's own sequence is read verbatim from --gfp_binder_chain in
   --gfp_binder_pdb. No hotspot/interface information is extracted here --
   that is a separate, required argument for the downstream Test-3/4 script
   (score_gfp_fusion_partners.py), not derivable from this step.
3. score_gfp_fusion_monomer.py understands redesign_terminus="c" output as
   of 2026-09-17 (see OUTPUT_FIELDS comment). score_gfp_fusion_partners.py
   (downstream, Test 2/3/4) does NOT yet -- it would need its own
   redesign_terminus-aware update, analogous to score_latch_monomer.py's
   and score_gfp_fusion_monomer.py's, before a "c" run can be scored all
   the way through.

Example
-------

python -u -m flexcraft.pipelines.dual_binder_pipeline.scripts.build_gfp_fusion_csv \
    --test1_csv flexcraft/pipelines/dual_binder_pipeline/results/latch_monomer_templated_03/score_success.csv \
    --gfp_binder_pdb flexcraft/pipelines/dual_binder_pipeline/inputs/gfp_binder_design_1101_0_complex.pdb \
    --gfp_binder_chain B \
    --run_name gfp_fusion_templated_03 \
    --register_offsets -2,-1,0,1,2 \
    --overwrite
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import gemmi


AMINO_ACID_3LETTER = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
AMINO_ACID_1LETTER = {v: k for k, v in AMINO_ACID_3LETTER.items()}

# gfp_segment_end_residue / binder_start_residue: for redesign_terminus "n"
# these mean what their names say (end of the leading GFP segment, start of
# the trailing existing-binder segment). For redesign_terminus "c" the
# segment order is reversed (existing binder leads, GFP binder trails), so
# the SAME two field names are reused with flipped meaning -- exactly the
# precedent set by score_latch_monomer.py's binder_start_residue/
# redesign_terminus pair: gfp_segment_end_residue = end of the (unchanged)
# binder segment, binder_start_residue = start of the trailing GFP segment.
# Always check redesign_terminus before interpreting either field.
OUTPUT_FIELDS = [
    "status",
    "task_id",
    "source_test1_task_id",
    "source_test1_design_id",
    "seq_id",
    "register_offset",
    "redesign_terminus",
    "chain",
    "input_pdb",
    "full_sequence",
    "gfp_segment_end_residue",
    "binder_start_residue",
    "num_filler_residues",
    "num_gfp_segment_residues",
    "num_blocking_residues",
    "num_binder_residues",
    "gfp_binder_source_pdb",
    "gfp_binder_chain",
    "filler_residue",
    "error",
]


@dataclass(frozen=True)
class ResidueRecord:
    chain: str
    chain_position: int
    residue_number: int
    residue_name: str


@dataclass(frozen=True)
class Test1Row:
    task_id: str
    design_id: str
    seq_id: str
    chain: str
    input_pdb: Path
    binder_start_residue: int
    redesign_terminus: str
    full_sequence: str | None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build GFP-binder + blocking-peptide + binder fusion constructs "
            "at several helix register offsets, from score_latch_monomer.py "
            "(Test 1) output."
        )
    )
    parser.add_argument(
        "--test1_csv",
        required=True,
        help="score_all.csv/score_success.csv output from score_latch_monomer.py (Test 1).",
    )
    parser.add_argument(
        "--test1_status_filter",
        default="success",
        help="Only use Test-1 rows with this status. Use 'all' to use every row. Default: success",
    )
    parser.add_argument(
        "--test1_pdb_source",
        choices=["predicted", "input"],
        default="predicted",
        help=(
            "Which Test-1 structure to build on: 'predicted' (Test 1's AF2 "
            "fold, already validated -- recommended) or 'input' (the raw "
            "ProteinMPNN design backbone). Default: predicted"
        ),
    )
    parser.add_argument(
        "--gfp_binder_pdb",
        required=True,
        help="PDB file containing the GFP + GFP-binder complex.",
    )
    parser.add_argument(
        "--gfp_binder_chain",
        required=True,
        help="Chain ID of the GFP binder (not GFP itself) within --gfp_binder_pdb.",
    )
    parser.add_argument(
        "--register_offsets",
        default="-2,-1,0,1,2",
        help=(
            "Comma-separated residue offsets applied to the blocking "
            "peptide's free end (its N-terminus for redesign_terminus 'n', "
            "its C-terminus for 'c') before directly attaching the GFP "
            "binder (no linker). Negative = trim existing residues, "
            "positive = add filler residues (an alpha helix turns ~100 "
            "degrees/residue). Default: -2,-1,0,1,2"
        ),
    )
    parser.add_argument(
        "--filler_residue",
        default="A",
        help=(
            "Single-letter amino acid used to fill positive register "
            "offsets (generic helix-favoring placeholder, not a redesign or "
            "sequence continuation -- AF2 folds it freely downstream). "
            "Default: A (alanine)"
        ),
    )
    parser.add_argument(
        "--run_name",
        required=True,
        help="Output subfolder name inside --results_root.",
    )
    parser.add_argument(
        "--repo_root",
        default=".",
        help="FlexCraft repository root. Default: current working directory.",
    )
    parser.add_argument(
        "--results_root",
        default="flexcraft/pipelines/dual_binder_pipeline/results",
        help="Pipeline results directory relative to --repo_root.",
    )
    parser.add_argument(
        "--chain",
        default="A",
        help="Chain ID to assign to the new fused construct. Default: A",
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=0,
        help="Maximum number of Test-1 rows to process. Use 0 for all. Useful for debugging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and replace an existing non-empty output directory.",
    )
    return parser


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def ensure_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}\nUse a new --run_name or pass --overwrite."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def is_protein_residue(residue: gemmi.Residue) -> bool:
    return bool(gemmi.find_tabulated_residue(residue.name).is_amino_acid())


def read_chain_records(path: Path, chain_id: str) -> list[ResidueRecord]:
    structure = gemmi.read_structure(str(path))
    if len(structure) == 0:
        raise ValueError(f"Structure contains no models: {path}")

    records: list[ResidueRecord] = []
    position = 0
    for chain in structure[0]:
        if chain.name != chain_id:
            continue
        for residue in chain:
            if not is_protein_residue(residue):
                continue
            ca = residue.find_atom("CA", "*")
            if ca is None or ca.occ == 0.0:
                continue
            position += 1
            records.append(
                ResidueRecord(
                    chain=chain.name,
                    chain_position=position,
                    residue_number=int(residue.seqid.num),
                    residue_name=residue.name.strip().upper(),
                )
            )

    if not records:
        raise ValueError(f"Chain {chain_id!r} not found (or empty) in {path}")
    return records


def read_chain_sequence(path: Path, chain_id: str) -> str:
    records = read_chain_records(path, chain_id)
    try:
        return "".join(AMINO_ACID_1LETTER[record.residue_name] for record in records)
    except KeyError as error:
        raise ValueError(
            f"Chain {chain_id!r} in {path} contains a residue with no 1-letter mapping: {error}"
        ) from error


def read_scoring_csv_rows(csv_path: Path):
    """Yield {column: value} dicts, tolerant of rows padded with whitespace
    (e.g. from a spreadsheet re-save) -- same rationale/helper as the
    identically-named function in score_latch_ternary.py."""
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
                    f"{csv_path}, line {row_number}: {len(raw_row)} fields, "
                    f"header has {len(fieldnames)}. Use the original, "
                    "untouched score_all.csv/score_success.csv."
                )
            yield {
                key: (value.strip() if isinstance(value, str) else value)
                for key, value in zip(fieldnames, raw_row)
            }


def rows_from_test1_csv(
    csv_path: Path,
    repo_root: Path,
    status_filter: str,
    pdb_source: str,
) -> list[Test1Row]:
    pdb_column = "predicted_pdb" if pdb_source == "predicted" else "input_pdb"

    rows = list(read_scoring_csv_rows(csv_path))
    if rows and pdb_column not in rows[0]:
        raise ValueError(
            f"Column {pdb_column!r} not found in {csv_path} -- is this "
            "really a score_latch_monomer.py score_all.csv/score_success.csv output?"
        )

    result: list[Test1Row] = []
    for row in rows:
        if status_filter != "all" and row.get("status") != status_filter:
            continue

        pdb_text = row.get(pdb_column, "")
        if not pdb_text:
            continue
        input_pdb = Path(pdb_text).expanduser()
        if not input_pdb.is_absolute():
            input_pdb = repo_root / input_pdb

        binder_start_text = (row.get("binder_start_residue") or "").strip()
        if not binder_start_text:
            raise ValueError(
                f"Row {row.get('task_id')!r} has no binder_start_residue -- "
                "is this really a score_latch_monomer.py output CSV?"
            )

        # Default "n" for older Test-1 CSVs written before score_latch_monomer.py's
        # 2026-09-16 redesign_terminus update -- they are all N-terminal latches.
        redesign_terminus = (row.get("redesign_terminus") or "n").strip().lower()
        if redesign_terminus not in ("n", "c"):
            raise ValueError(
                f"Invalid redesign_terminus {redesign_terminus!r} for row "
                f"{row.get('task_id')!r} in {csv_path} -- must be 'n' or 'c'."
            )

        fallback_id = input_pdb.stem
        result.append(
            Test1Row(
                task_id=row.get("task_id") or fallback_id,
                design_id=row.get("design_id") or row.get("task_id") or fallback_id,
                seq_id=str(row.get("seq_id", "")),
                chain=(row.get("chain") or "A").strip() or "A",
                input_pdb=input_pdb.resolve(),
                binder_start_residue=int(binder_start_text),
                redesign_terminus=redesign_terminus,
                full_sequence=row.get("full_sequence") or None,
            )
        )

    if not result:
        raise ValueError(f"No usable rows found in {csv_path}")
    return result


def idealized_backbone_atoms(index: int) -> list[tuple[str, tuple[float, float, float]]]:
    """Same idealized, non-physical extended-chain placeholder as
    score_latch_ternary.py's idealized_backbone_atoms(). Never used as an AF2
    template/guess -- purely a coordinate placeholder for a syntactically
    valid structure."""
    x = index * 3.8
    z = 1.2 if index % 2 == 0 else -1.2
    return [
        ("N", (x - 0.9, 0.4, z)),
        ("CA", (x, 0.0, z)),
        ("C", (x + 0.9, 0.4, z)),
        ("O", (x + 0.9, 1.6, z)),
    ]


def make_placeholder_residue(letter: str, index: int) -> gemmi.Residue:
    resname = AMINO_ACID_3LETTER.get(letter.upper())
    if resname is None:
        raise ValueError(f"Unsupported amino acid code {letter!r}.")
    residue = gemmi.Residue()
    residue.name = resname
    residue.seqid = gemmi.SeqId(index + 1, " ")  # renumbered sequentially at the end anyway
    residue.het_flag = "A"
    for atom_name, position in idealized_backbone_atoms(index):
        atom = gemmi.Atom()
        atom.name = atom_name
        atom.element = gemmi.Element(atom_name[0])
        atom.pos = gemmi.Position(*position)
        atom.occ = 1.0
        atom.b_iso = 50.0
        residue.add_atom(atom)
    return residue


def build_fusion_construct(
    test1_row: Test1Row,
    gfp_binder_sequence: str,
    register_offset: int,
    filler_residue: str,
    out_chain_id: str,
    out_path: Path,
) -> tuple[str, int, int, int]:
    """Build one fused construct, GFP binder attached directly (no linker)
    to the blocking peptide's free end. redesign_terminus "n": [GFP
    binder][blocking peptide, N-terminally trimmed/extended by
    register_offset][existing binder, unchanged]. "c": [existing binder,
    unchanged][blocking peptide, C-terminally trimmed/extended by
    register_offset][GFP binder]. Returns (full_sequence,
    gfp_segment_end_residue, binder_start_residue, num_filler_residues) in
    the NEW chain's sequential 1-based numbering -- see the OUTPUT_FIELDS
    comment for how the meaning of the last three depends on
    redesign_terminus."""
    source_structure = gemmi.read_structure(str(test1_row.input_pdb))
    source_model = source_structure[0]
    source_chain = next((c for c in source_model if c.name == test1_row.chain), None)
    if source_chain is None:
        raise ValueError(f"Chain {test1_row.chain!r} not found in {test1_row.input_pdb}")

    source_residues = [
        residue
        for residue in source_chain
        if is_protein_residue(residue) and residue.find_atom("CA", "*") is not None
    ]

    # binder_start_residue is a 1-based POSITIONAL index -- the Nth protein-with-CA
    # residue of the chain -- carried through unchanged from
    # score_latch_monomer.py / prepare_scoring_csv (first_fixed_residue_id).
    # It is NOT a residue.seqid.num value: score_latch_monomer.py writes its
    # predicted PDBs 0-indexed, so the previous `seqid.num < binder_start_residue`
    # comparison counted one blocking residue too many, which pushed every
    # downstream binder_start_residue +1 and made score_gfp_fusion_monomer.py
    # check the hotspots one register off (bug found 2026-09-01). Slice
    # positionally instead -- no seqid / contiguity assumption. Which side of
    # the boundary is the blocking peptide depends on redesign_terminus (see
    # module docstring / score_latch_monomer.py's 2026-09-16 update).
    if not 1 <= test1_row.binder_start_residue <= len(source_residues):
        raise ValueError(
            f"{test1_row.task_id}: binder_start_residue="
            f"{test1_row.binder_start_residue} is out of range for a "
            f"{len(source_residues)}-residue chain in {test1_row.input_pdb}."
        )

    num_filler_residues = max(register_offset, 0)
    gfp_binder_sequence = gfp_binder_sequence.strip().upper()

    new_structure = gemmi.Structure()
    new_structure.name = f"{test1_row.task_id}_gfp_fusion_offset{register_offset:+d}"
    new_model = gemmi.Model("1")
    new_chain = gemmi.Chain(out_chain_id)
    index = 0

    def add_gfp_binder() -> None:
        nonlocal index
        for letter in gfp_binder_sequence:
            new_chain.add_residue(make_placeholder_residue(letter, index))
            index += 1

    def add_filler() -> None:
        nonlocal index
        for _ in range(num_filler_residues):
            new_chain.add_residue(make_placeholder_residue(filler_residue, index))
            index += 1

    def add_kept_residues(kept_residues: list[gemmi.Residue]) -> None:
        nonlocal index
        for residue in kept_residues:
            new_residue = gemmi.Residue()
            new_residue.name = residue.name
            new_residue.het_flag = "A"
            new_residue.seqid = gemmi.SeqId(index + 1, " ")  # placeholder, renumbered below
            for atom in residue:
                new_atom = gemmi.Atom()
                new_atom.name = atom.name
                new_atom.element = atom.element
                new_atom.pos = atom.pos
                new_atom.occ = atom.occ
                new_atom.b_iso = atom.b_iso
                new_residue.add_atom(new_atom)
            new_chain.add_residue(new_residue)
            index += 1

    if test1_row.redesign_terminus == "n":
        # Blocking peptide (free N-terminus) = source_residues[:boundary-1],
        # existing binder (unchanged) = source_residues[boundary-1:].
        # register_offset trims/extends the blocking peptide's free
        # N-terminal end, right where the GFP binder attaches.
        num_latch_original = test1_row.binder_start_residue - 1
        if register_offset < 0:
            trimmed = abs(register_offset)
            if trimmed >= num_latch_original:
                raise ValueError(
                    f"register_offset={register_offset} would remove all "
                    f"{num_latch_original} blocking-segment residues of "
                    f"{test1_row.task_id} -- reduce the offset magnitude."
                )
        else:
            trimmed = 0
        kept_residues = source_residues[trimmed:]  # remaining latch + unchanged binder
        latch_kept_count = num_latch_original - trimmed

        add_gfp_binder()
        gfp_segment_end_residue = index  # end of GFP segment
        add_filler()
        add_kept_residues(kept_residues)
        binder_start_residue = gfp_segment_end_residue + num_filler_residues + latch_kept_count + 1
    else:
        # Existing binder (unchanged) = source_residues[:boundary-1],
        # blocking peptide (free C-terminus) = source_residues[boundary-1:].
        # register_offset trims/extends the blocking peptide's free
        # C-terminal end, right where the GFP binder attaches.
        num_latch_original = len(source_residues) - (test1_row.binder_start_residue - 1)
        if register_offset < 0:
            trimmed = abs(register_offset)
            if trimmed >= num_latch_original:
                raise ValueError(
                    f"register_offset={register_offset} would remove all "
                    f"{num_latch_original} blocking-segment residues of "
                    f"{test1_row.task_id} -- reduce the offset magnitude."
                )
        else:
            trimmed = 0
        kept_residues = source_residues[: len(source_residues) - trimmed]  # unchanged binder + remaining latch

        add_kept_residues(kept_residues)
        gfp_segment_end_residue = test1_row.binder_start_residue - 1  # end of (unchanged) binder segment
        add_filler()
        binder_start_residue = index + 1  # start of GFP segment
        add_gfp_binder()

    # Renumber sequentially 1..N so downstream boundaries are unambiguous.
    for position, residue in enumerate(new_chain, start=1):
        residue.seqid = gemmi.SeqId(position, " ")

    new_model.add_chain(new_chain)
    new_structure.add_model(new_model)
    new_structure.setup_entities()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_structure.write_pdb(str(out_path))

    full_sequence = "".join(
        AMINO_ACID_1LETTER.get(residue.name.strip().upper(), "X") for residue in new_chain
    )
    return full_sequence, gfp_segment_end_residue, binder_start_residue, num_filler_residues


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> None:
    args = build_arg_parser().parse_args()

    offsets = [int(item.strip()) for item in args.register_offsets.split(",") if item.strip()]
    if not offsets:
        raise ValueError("--register_offsets must contain at least one value.")
    if len(args.filler_residue.strip()) != 1:
        raise ValueError("--filler_residue must be a single amino acid letter.")

    repo_root = Path(args.repo_root).expanduser().resolve()
    results_root = resolve_path(repo_root, args.results_root)
    run_dir = results_root / args.run_name
    constructs_dir = run_dir / "constructs"

    ensure_output_directory(run_dir, args.overwrite)
    constructs_dir.mkdir(parents=True, exist_ok=True)

    test1_csv_path = resolve_path(repo_root, args.test1_csv)
    gfp_binder_pdb_path = resolve_path(repo_root, args.gfp_binder_pdb)

    test1_rows = rows_from_test1_csv(
        csv_path=test1_csv_path,
        repo_root=repo_root,
        status_filter=args.test1_status_filter,
        pdb_source=args.test1_pdb_source,
    )
    if args.max_tasks > 0:
        test1_rows = test1_rows[: args.max_tasks]

    gfp_binder_sequence = read_chain_sequence(gfp_binder_pdb_path, args.gfp_binder_chain)
    print(
        f"GFP binder sequence ({len(gfp_binder_sequence)} aa, "
        f"chain {args.gfp_binder_chain}): {gfp_binder_sequence}"
    )

    # Create the output CSV up front (header only) and append one row per built
    # construct, flushing after each -- so a crash / cancelled job leaves a valid
    # partial CSV and progress can be watched live.
    output_csv = run_dir / "gfp_fusion_constructs.csv"
    total = len(test1_rows) * len(offsets)
    built = 0

    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        handle.flush()

        for test1_row in test1_rows:
            for offset in offsets:
                task_id = safe_id(f"{test1_row.task_id}_reg{offset:+d}")
                out_pdb = constructs_dir / f"{task_id}.pdb"
                row = {field: "" for field in OUTPUT_FIELDS}
                row.update(
                    {
                        "task_id": task_id,
                        "source_test1_task_id": test1_row.task_id,
                        "source_test1_design_id": test1_row.design_id,
                        "seq_id": test1_row.seq_id,
                        "register_offset": offset,
                        "redesign_terminus": test1_row.redesign_terminus,
                        "chain": args.chain,
                        "gfp_binder_source_pdb": str(gfp_binder_pdb_path),
                        "gfp_binder_chain": args.gfp_binder_chain,
                        "filler_residue": args.filler_residue,
                    }
                )
                try:
                    full_sequence, gfp_end, binder_start, num_filler = build_fusion_construct(
                        test1_row=test1_row,
                        gfp_binder_sequence=gfp_binder_sequence,
                        register_offset=offset,
                        filler_residue=args.filler_residue,
                        out_chain_id=args.chain,
                        out_path=out_pdb,
                    )
                    # For redesign_terminus "n" the GFP segment leads (length
                    # gfp_end) and the binder trails; for "c" it's reversed --
                    # see the OUTPUT_FIELDS comment on gfp_segment_end_residue/
                    # binder_start_residue.
                    head_segment_len = gfp_end
                    tail_segment_len = len(full_sequence) - binder_start + 1
                    if test1_row.redesign_terminus == "n":
                        num_gfp_segment_residues, num_binder_residues = head_segment_len, tail_segment_len
                    else:
                        num_gfp_segment_residues, num_binder_residues = tail_segment_len, head_segment_len
                    row.update(
                        {
                            "status": "success",
                            "input_pdb": str(out_pdb),
                            "full_sequence": full_sequence,
                            "gfp_segment_end_residue": gfp_end,
                            "binder_start_residue": binder_start,
                            "num_filler_residues": num_filler,
                            "num_gfp_segment_residues": num_gfp_segment_residues,
                            "num_blocking_residues": binder_start - gfp_end - 1,
                            "num_binder_residues": num_binder_residues,
                        }
                    )
                    built += 1
                    print(
                        f"[ok] {task_id}: {len(full_sequence)} aa "
                        f"(gfp<= {gfp_end}, binder>= {binder_start}, filler={num_filler})"
                    )
                except Exception as error:
                    row["status"] = "failed"
                    row["error"] = f"{type(error).__name__}: {error}"
                    print(f"[failed] {task_id}: {type(error).__name__}: {error}")

                writer.writerow(row)
                handle.flush()

    print("Done.")
    print(f"Built:      {built}/{total}")
    print(f"Output CSV: {output_csv}")
    print(f"Constructs: {constructs_dir}")


if __name__ == "__main__":
    main()
