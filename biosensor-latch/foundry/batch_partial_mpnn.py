#!/usr/bin/env python3
"""Batch-redesign a variable N- or C-terminal region with Foundry ProteinMPNN.

For each compressed mmCIF (.cif.gz), the script finds the fixed/original
binder region via a sequence match, redesigns the residues on one side of it,
asks ProteinMPNN for 8 sequences by default, and writes one FASTA per binder
plus a combined CSV summary.

Boundary detection can use exactly one of:
  1) --fixed-start-sequence: a short unique anchor at the START of the fixed
     region (immediately after the boundary). Pair with --redesign-terminus n
     (default). The anchor does NOT need to cover the whole fixed region —
     only its start, since the boundary IS the anchor's start position.
  2) --fixed-end-sequence: a short unique anchor at the END of the fixed
     region (immediately before the boundary). Pair with --redesign-terminus
     c. Only its end matters, since the boundary is the anchor's end
     position.
  3) --reference-fasta: the FULL fixed/original binder sequence(s), FASTA IDs
     matching structure stems (e.g. binder_001 for binder_001.cif.gz), or a
     single record applied to every structure. Works with either
     --redesign-terminus value, since the full length lets the script derive
     both ends of the match.

--redesign-terminus selects which side of the fixed region is redesigned:
  - "n" (default, original behavior): redesign all residues *before* the
    fixed region — the variable/generated latch is N-terminal. Requires
    --fixed-start-sequence or --reference-fasta (NOT --fixed-end-sequence —
    a short end-anchor cannot locate where the fixed region starts).
  - "c": redesign all residues *after* the fixed region — the variable/
    generated latch is C-terminal. Requires --fixed-end-sequence or
    --reference-fasta (NOT --fixed-start-sequence — a short start-anchor
    cannot locate where the fixed region ends, and using it here would
    silently truncate the fixed region to the anchor's own length).

Requires an activated Foundry environment with the `mpnn` command available.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shlex
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path


SCRIPT_VERSION = "2026-09-16-cterm-v4"
INPUT_GLOBS = ("*.pdb", "*.cif.gz")


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}

CSV_FIELDS = [
    "status", "error", "binder_id", "structure_file", "chain", "design_number",
    "original_full_sequence", "new_full_sequence",
    "original_redesigned_sequence", "new_redesigned_sequence",
    "fixed_sequence_after_boundary", "num_redesigned_residues",
    "num_mutations_in_redesigned_region", "fraction_mutated_in_redesigned_region",
    "fixed_region_unchanged", "redesign_terminus", "first_fixed_residue_id",
    "boundary_residue_id", "designed_residue_ids",
    "boundary_method", "boundary_match_sequence", "foundry_batch_index",
    "foundry_design_index", "foundry_confidence", "foundry_sequence_recovery",
    "temperature", "checkpoint", "foundry_fasta_header", "foundry_fasta_file",
]


@dataclass(frozen=True)
class Residue:
    chain: str
    number: int
    insertion: str
    aa: str

    @property
    def foundry_id(self) -> str:
        return f"{self.chain}{self.number}{self.insertion}"


@dataclass
class FastaRecord:
    header: str
    sequence: str
    source: Path | None = None


def clean_seq(seq: str) -> str:
    return re.sub(r"[^A-Za-z]", "", seq).upper()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Redesign variable N-terminal binder extensions with Foundry ProteinMPNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--chain", default="A")
    p.add_argument(
        "--redesign-terminus", choices=["n", "c"], default="n",
        help="Which side of the matched fixed region to redesign: 'n' redesigns "
             "residues before the match (variable N-terminal latch, original "
             "behavior); 'c' redesigns residues after the match (variable "
             "C-terminal latch).",
    )

    boundary = p.add_mutually_exclusive_group(required=True)
    boundary.add_argument(
        "--fixed-start-sequence",
        help="Unique exact anchor at the START of the fixed region "
             "(use with --redesign-terminus n).",
    )
    boundary.add_argument(
        "--fixed-end-sequence",
        help="Unique exact anchor at the END of the fixed region, immediately "
             "before the redesigned region (use with --redesign-terminus c).",
    )
    boundary.add_argument(
        "--reference-fasta", type=Path,
        help="Full original/fixed binder sequence(s); record IDs should match "
             ".cif.gz stems. Works with either --redesign-terminus value.",
    )

    p.add_argument("--num-sequences", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--omit", default="UNK",
                   help="Comma-separated residue names, e.g. UNK or CYS,UNK.")
    p.add_argument("--mpnn-command", default="mpnn")
    p.add_argument("--legacy-weights", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--recursive", action="store_true")
    return p.parse_args()


def structure_stem(path: Path) -> str:
    """Return a clean structure ID, stripping .cif.gz (or a single suffix)."""
    name = path.name
    if name.lower().endswith(".cif.gz"):
        return name[:-7]
    return path.stem


def _pick_field(headers: list[str], *names: str) -> int | None:
    for name in names:
        if name in headers:
            return headers.index(name)
    return None

def parse_pdb(path: Path) -> OrderedDict[str, list[Residue]]:
    chains: OrderedDict[str, list[Residue]] = OrderedDict()
    seen: set[tuple[str, int, str]] = set()
    first_model_done = False
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            rec = line[:6].strip()
            if rec == "ENDMDL":
                first_model_done = True
            if first_model_done or rec not in {"ATOM", "HETATM"}:
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[16] not in {" ", "A", "1"}:      # altloc
                continue
            resname = line[17:20].strip().upper()
            if resname not in AA3_TO_1:
                continue
            chain = line[21].strip()
            if not chain:
                raise ValueError("Blank chain ID; assign chain IDs before running.")
            number = int(line[22:26])
            insertion = line[26].strip()
            key = (chain, number, insertion)
            if key in seen:
                continue
            seen.add(key)
            chains.setdefault(chain, []).append(Residue(chain, number, insertion, AA3_TO_1[resname]))
    if not chains:
        raise ValueError("No protein CA atoms found in PDB file.")
    return chains


def parse_structure(path: Path):
    return parse_cif_gz(path) if path.name.lower().endswith(".cif.gz") else parse_pdb(path)

def parse_cif_gz(path: Path) -> OrderedDict[str, list[Residue]]:
    """Extract protein residues from CA atoms in the first model of a .cif.gz file.

    The parser reads the mmCIF ``_atom_site`` loop directly with the Python
    standard library, so no Biopython/Gemmi dependency is required. Author
    chain/residue IDs are preferred when present because they are the most
    natural IDs to pass back to Foundry residue selectors.
    """
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        lines = iter(handle)
        headers: list[str] | None = None

        for raw in lines:
            if raw.strip() != "loop_":
                continue

            loop_headers: list[str] = []
            first_data_line: str | None = None
            for raw2 in lines:
                stripped = raw2.strip()
                if stripped.startswith("_"):
                    loop_headers.append(stripped.split()[0])
                    continue
                first_data_line = raw2
                break

            if loop_headers and all(h.startswith("_atom_site.") for h in loop_headers):
                headers = loop_headers
                break

        if not headers:
            raise ValueError("No _atom_site loop found in compressed mmCIF file.")

        atom_idx = _pick_field(headers, "_atom_site.label_atom_id", "_atom_site.auth_atom_id")
        resname_idx = _pick_field(headers, "_atom_site.label_comp_id", "_atom_site.auth_comp_id")
        auth_chain_idx = _pick_field(headers, "_atom_site.auth_asym_id")
        label_chain_idx = _pick_field(headers, "_atom_site.label_asym_id")
        auth_resid_idx = _pick_field(headers, "_atom_site.auth_seq_id")
        label_resid_idx = _pick_field(headers, "_atom_site.label_seq_id")
        alt_idx = _pick_field(headers, "_atom_site.label_alt_id")
        ins_idx = _pick_field(headers, "_atom_site.pdbx_PDB_ins_code")
        model_idx = _pick_field(headers, "_atom_site.pdbx_PDB_model_num")
        group_idx = _pick_field(headers, "_atom_site.group_PDB")

        required = {
            "atom name": atom_idx,
            "residue name": resname_idx,
            "chain ID": auth_chain_idx if auth_chain_idx is not None else label_chain_idx,
            "residue ID": auth_resid_idx if auth_resid_idx is not None else label_resid_idx,
        }
        missing = [name for name, idx in required.items() if idx is None]
        if missing:
            raise ValueError("Missing required _atom_site field(s): " + ", ".join(missing))

        chains: OrderedDict[str, list[Residue]] = OrderedDict()
        seen: set[tuple[str, int, str]] = set()
        first_model: str | None = None

        def row_lines():
            if first_data_line is not None:
                yield first_data_line
            yield from lines

        buffer = ""
        for raw in row_lines():
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped == "#" or stripped == "loop_" or stripped.startswith("_"):
                break
            if stripped.startswith(";"):
                raise ValueError("Unexpected multiline value inside _atom_site loop.")

            buffer = f"{buffer} {stripped}".strip()
            try:
                tokens = shlex.split(buffer, posix=True)
            except ValueError:
                # A quoted token may continue on the next physical line.
                continue

            if len(tokens) < len(headers):
                continue
            if len(tokens) != len(headers):
                raise ValueError(
                    f"Could not parse _atom_site row: expected {len(headers)} values, "
                    f"found {len(tokens)}."
                )
            buffer = ""

            if group_idx is not None and tokens[group_idx].upper() not in {"ATOM", "HETATM"}:
                continue
            if tokens[atom_idx].strip('"\'') != "CA":
                continue
            if alt_idx is not None and tokens[alt_idx] not in {".", "?", "A", "1"}:
                continue

            model = tokens[model_idx] if model_idx is not None else "1"
            if first_model is None:
                first_model = model
            elif model != first_model:
                continue

            resname = tokens[resname_idx].upper()
            if resname not in AA3_TO_1:
                continue

            chain = ""
            if auth_chain_idx is not None:
                chain = tokens[auth_chain_idx]
            if chain in {".", "?", ""} and label_chain_idx is not None:
                chain = tokens[label_chain_idx]
            if chain in {".", "?", ""}:
                raise ValueError("Blank/unknown chain ID found; assign chain IDs before running.")

            resid_token = "?"
            if auth_resid_idx is not None:
                resid_token = tokens[auth_resid_idx]
            if resid_token in {".", "?", ""} and label_resid_idx is not None:
                resid_token = tokens[label_resid_idx]
            if resid_token in {".", "?", ""}:
                continue
            try:
                number = int(resid_token)
            except ValueError as exc:
                raise ValueError(f"Non-integer residue ID {resid_token!r} in chain {chain}.") from exc

            insertion = ""
            if ins_idx is not None and tokens[ins_idx] not in {".", "?"}:
                insertion = tokens[ins_idx]

            key = (chain, number, insertion)
            if key in seen:
                continue
            seen.add(key)
            chains.setdefault(chain, []).append(
                Residue(chain, number, insertion, AA3_TO_1[resname])
            )

    if not chains:
        raise ValueError("No protein CA atoms found in compressed mmCIF file.")
    return chains


def read_fasta(path: Path) -> list[FastaRecord]:
    records, header, parts = [], None, []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append(FastaRecord(header, "".join(parts), path))
            header, parts = line[1:].strip(), []
        else:
            if header is None:
                raise ValueError(f"Malformed FASTA: {path}")
            parts.append(line)
    if header is not None:
        records.append(FastaRecord(header, "".join(parts), path))
    return records


def load_references(path: Path) -> OrderedDict[str, str]:
    refs: OrderedDict[str, str] = OrderedDict()
    for record in read_fasta(path):
        key = record.header.split()[0]
        if key in refs:
            raise ValueError(f"Duplicate reference FASTA ID: {key}")
        refs[key] = clean_seq(record.sequence)
    if not refs:
        raise ValueError("Reference FASTA contains no records.")
    return refs


def find_unique(sequence: str, query: str) -> int:
    positions, start = [], 0
    while True:
        idx = sequence.find(query, start)
        if idx < 0:
            break
        positions.append(idx)
        start = idx + 1
    if not positions:
        raise ValueError(f"Boundary sequence not found: {query}")
    if len(positions) != 1:
        raise ValueError(f"Boundary sequence occurs {len(positions)} times; use a longer sequence.")
    return positions[0]


def boundary_query(
    binder_id: str,
    fixed_start: str | None,
    fixed_end: str | None,
    refs: OrderedDict[str, str] | None,
) -> tuple[str, str]:
    if fixed_start is not None:
        return clean_seq(fixed_start), "fixed_start_sequence"
    if fixed_end is not None:
        return clean_seq(fixed_end), "fixed_end_sequence"
    assert refs is not None
    if binder_id in refs:
        return refs[binder_id], "reference_fasta"
    if len(refs) == 1:
        return next(iter(refs.values())), "reference_fasta_single_record"
    raise ValueError(f"No reference FASTA record named {binder_id!r}.")


def header_metadata(header: str) -> dict[str, str]:
    md: dict[str, str] = {}
    for item in [x.strip() for x in header.split(",")]:
        if "=" in item:
            k, v = item.split("=", 1)
            md[k.strip()] = v.strip()
    m = re.search(r"_b(\d+)_d(\d+)(?:$|,|\s)", header)
    if m:
        md["batch_idx"], md["design_idx"] = m.group(1), m.group(2)
    return md


def collect_fastas(folder: Path) -> list[FastaRecord]:
    paths = sorted([*folder.rglob("*.fa"), *folder.rglob("*.fasta"), *folder.rglob("*.faa")])
    records = [record for path in paths for record in read_fasta(path)]
    return sorted(
        records,
        key=lambda r: (
            int(header_metadata(r.header).get("batch_idx", 10**9)),
            int(header_metadata(r.header).get("design_idx", 10**9)),
            r.header,
        ),
    )


def target_chain_from_output(
    sequence: str,
    chains: OrderedDict[str, list[Residue]],
    target: str,
) -> str:
    order = list(chains)
    lengths = {c: len(chains[c]) for c in order}
    raw = sequence.strip().upper()

    if "/" in raw:
        parts = [clean_seq(x) for x in raw.split("/")]
        if len(parts) == len(order):
            return parts[order.index(target)]

    seq = clean_seq(raw)
    if len(seq) == lengths[target]:
        return seq
    if len(seq) == sum(lengths.values()):
        offset = 0
        for chain in order:
            if chain == target:
                return seq[offset:offset + lengths[chain]]
            offset += lengths[chain]

    raise ValueError(
        f"Cannot map Foundry FASTA to chain {target}. Output length={len(seq)}, "
        f"chain lengths={lengths}."
    )


def empty_row() -> dict[str, object]:
    return {k: "" for k in CSV_FIELDS}


def write_binder_fasta(path: Path, binder_id: str, rows: list[dict[str, object]]) -> None:
    with path.open("w") as f:
        for row in rows:
            extras = [
                f"redesigned_residues={row['num_redesigned_residues']}",
                f"mutations={row['num_mutations_in_redesigned_region']}",
            ]
            if row["foundry_confidence"]:
                extras.append(f"confidence={row['foundry_confidence']}")
            f.write(f">{binder_id}_design_{int(row['design_number']):02d} {' '.join(extras)}\n")
            f.write(str(row["new_full_sequence"]) + "\n")


def run_binder(
    structure: Path,
    index: int,
    args: argparse.Namespace,
    refs: OrderedDict[str, str] | None,
    fasta_dir: Path,
    raw_dir: Path,
    log_dir: Path,
) -> list[dict[str, object]]:
    binder_id = structure_stem(structure)
    chains = parse_structure(structure)
    if args.chain not in chains:
        raise ValueError(f"Chain {args.chain!r} not found; present chains: {list(chains)}")

    residues = chains[args.chain]
    original = "".join(r.aa for r in residues)
    query, method = boundary_query(binder_id, args.fixed_start_sequence, args.fixed_end_sequence, refs)
    match_start = find_unique(original, query)

    if args.redesign_terminus == "n":
        # Fixed region starts at the match; redesign everything before it.
        boundary = match_start
    else:
        # Fixed region ends at the match; redesign everything after it.
        boundary = match_start + len(query)

    design_slice = slice(0, boundary) if args.redesign_terminus == "n" else slice(boundary, None)
    fixed_slice = slice(boundary, None) if args.redesign_terminus == "n" else slice(0, boundary)

    design_residues = residues[design_slice]
    if not design_residues:
        side = "before" if args.redesign_terminus == "n" else "after"
        raise ValueError(f"Detected 0 residues {side} the fixed region; nothing to redesign.")

    design_ids = [r.foundry_id for r in design_residues]
    old_region = original[design_slice]
    fixed_region = original[fixed_slice]
    first_fixed = residues[fixed_slice][0].foundry_id
    # Residue at the split itself: first residue of whichever segment comes
    # second in chain order. For redesign_terminus=n this is the same as
    # first_fixed (fixed region starts here). For redesign_terminus=c this is
    # the first *designed* residue instead (fixed region is the prefix, so
    # first_fixed above is just the chain's first residue and isn't useful as
    # a boundary marker) -- downstream tools that need to split the chain
    # (e.g. score_latch_monomer.py) should use this field together with
    # redesign_terminus, not first_fixed_residue_id, when terminus is c.
    boundary_residue_id = residues[boundary].foundry_id

    binder_raw = raw_dir / binder_id
    if binder_raw.exists():
        shutil.rmtree(binder_raw)
    binder_raw.mkdir(parents=True)

    cmd = [
        args.mpnn_command,
        "--model_type", "protein_mpnn",
        "--checkpoint_path", str(args.checkpoint),
        "--is_legacy_weights", str(args.legacy_weights),
        "--structure_path", str(structure.resolve()),
        "--name", binder_id,
        "--out_directory", str(binder_raw.resolve()),
        "--designed_residues", ",".join(design_ids),
        "--batch_size", str(args.num_sequences),
        "--number_of_batches", "1",
        "--temperature", str(args.temperature),
        "--omit", json.dumps(args.omit_list),
        "--write_fasta", "True",
        "--write_structures", "False",
    ]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed + index)]

    log_path = log_dir / f"{binder_id}.log"
    with log_path.open("w") as log:
        log.write("COMMAND:\n" + " ".join(cmd) + "\n\n")
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"MPNN exited with code {result.returncode}; see {log_path}")

    fasta_records = collect_fastas(binder_raw)
    if not fasta_records:
        raise RuntimeError(f"MPNN produced no FASTA files; see {log_path}")

    rows: list[dict[str, object]] = []
    for design_no, record in enumerate(fasta_records, 1):
        new = target_chain_from_output(record.sequence, chains, args.chain)
        if len(new) != len(original):
            raise RuntimeError(f"Output sequence length changed: {len(original)} -> {len(new)}")

        new_region = new[design_slice]
        fixed_unchanged = new[fixed_slice] == fixed_region
        mutations = sum(a != b for a, b in zip(old_region, new_region))
        md = header_metadata(record.header)

        row = empty_row()
        row.update({
            "status": "OK" if fixed_unchanged else "WARNING_FIXED_REGION_CHANGED",
            "error": "" if fixed_unchanged else "Nominally fixed suffix changed in output FASTA.",
            "binder_id": binder_id,
            "structure_file": str(structure.resolve()),
            "chain": args.chain,
            "design_number": design_no,
            "original_full_sequence": original,
            "new_full_sequence": new,
            "original_redesigned_sequence": old_region,
            "new_redesigned_sequence": new_region,
            "fixed_sequence_after_boundary": fixed_region,
            "num_redesigned_residues": len(design_residues),
            "num_mutations_in_redesigned_region": mutations,
            "fraction_mutated_in_redesigned_region": f"{mutations / len(design_residues):.6f}",
            "fixed_region_unchanged": fixed_unchanged,
            "redesign_terminus": args.redesign_terminus,
            "first_fixed_residue_id": first_fixed,
            "boundary_residue_id": boundary_residue_id,
            "designed_residue_ids": ",".join(design_ids),
            "boundary_method": method,
            "boundary_match_sequence": query,
            "foundry_batch_index": md.get("batch_idx", ""),
            "foundry_design_index": md.get("design_idx", ""),
            "foundry_confidence": md.get("confidence", ""),
            "foundry_sequence_recovery": md.get("sequence_recovery", ""),
            "temperature": args.temperature,
            "checkpoint": str(args.checkpoint),
            "foundry_fasta_header": record.header,
            "foundry_fasta_file": str(record.source.resolve()) if record.source else "",
        })
        rows.append(row)

    if len(rows) != args.num_sequences:
        for row in rows:
            row["status"] = "WARNING_SEQUENCE_COUNT"
            row["error"] = f"Requested {args.num_sequences}, found {len(rows)} FASTA records."

    write_binder_fasta(fasta_dir / f"{binder_id}.fasta", binder_id, rows)
    return rows


def main() -> int:
    args = parse_args()
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.omit_list = [x.strip().upper() for x in args.omit.split(",") if x.strip()]

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint does not exist: {args.checkpoint}")
    if args.num_sequences < 1:
        raise SystemExit("--num-sequences must be >= 1")
    if shutil.which(args.mpnn_command) is None and not Path(args.mpnn_command).is_file():
        raise SystemExit(f"Cannot find MPNN command: {args.mpnn_command}")

    refs = None
    if args.reference_fasta:
        args.reference_fasta = args.reference_fasta.expanduser().resolve()
        refs = load_references(args.reference_fasta)
    elif args.fixed_start_sequence is not None:
        if args.redesign_terminus != "n":
            raise SystemExit(
                "--fixed-start-sequence anchors the START of the fixed region, which "
                "only identifies the boundary for --redesign-terminus n (N-terminal "
                "latch). For --redesign-terminus c, use --fixed-end-sequence (an "
                "anchor at the END of the fixed region) or --reference-fasta — a "
                "short start-anchor would silently truncate the fixed region to its "
                "own length and redesign real fixed residues."
            )
        args.fixed_start_sequence = clean_seq(args.fixed_start_sequence)
        if len(args.fixed_start_sequence) < 8:
            print("WARNING: use a longer boundary sequence if possible.", file=sys.stderr)
    else:
        if args.redesign_terminus != "c":
            raise SystemExit(
                "--fixed-end-sequence anchors the END of the fixed region, which only "
                "identifies the boundary for --redesign-terminus c (C-terminal "
                "latch). For --redesign-terminus n, use --fixed-start-sequence or "
                "--reference-fasta."
            )
        args.fixed_end_sequence = clean_seq(args.fixed_end_sequence)
        if len(args.fixed_end_sequence) < 8:
            print("WARNING: use a longer boundary sequence if possible.", file=sys.stderr)

    fasta_dir = args.output_dir / "fasta"
    raw_dir = args.output_dir / "foundry_raw"
    log_dir = args.output_dir / "logs"
    for folder in (args.output_dir, fasta_dir, raw_dir, log_dir):
        folder.mkdir(parents=True, exist_ok=True)

    globber = args.input_dir.rglob if args.recursive else args.input_dir.glob
    structures = sorted({p for g in INPUT_GLOBS for p in globber(g)})
    if not structures:
        raise SystemExit(f"No {INPUT_GLOBS} files found in {args.input_dir}")

    rows: list[dict[str, object]] = []
    failures = 0
    print(f"batch_partial_mpnn version: {SCRIPT_VERSION}")
    print(f"Scanning: {args.input_dir} for {', '.join(INPUT_GLOBS)}")
    print(
        f"Found {len(structures)} .cif.gz or .pdb structure(s); "
        f"generating {args.num_sequences} sequence(s) each."
    )

    for i, structure in enumerate(structures):
        binder_id = structure_stem(structure)
        print(f"[{i + 1}/{len(structures)}] {binder_id}")
        try:
            binder_rows = run_binder(structure, i, args, refs, fasta_dir, raw_dir, log_dir)
            rows.extend(binder_rows)
            print(
                f"  OK: {binder_rows[0]['num_redesigned_residues']} residues redesigned, "
                f"{len(binder_rows)} sequences written."
            )
        except Exception as exc:
            failures += 1
            row = empty_row()
            row.update({
                "status": "ERROR", "error": str(exc), "binder_id": binder_id,
                "structure_file": str(structure.resolve()), "chain": args.chain,
                "temperature": args.temperature, "checkpoint": str(args.checkpoint),
            })
            rows.append(row)
            print(f"  ERROR: {exc}", file=sys.stderr)

    csv_path = args.output_dir / "redesign_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV:    {csv_path}")
    print(f"FASTAs: {fasta_dir}")
    print(f"Logs:   {log_dir}")
    if failures:
        print(f"{failures} binder(s) failed; see CSV/logs.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
