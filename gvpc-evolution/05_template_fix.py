#!/usr/bin/env python3
"""
05_template_fix.py - makes a repacked template readable for atomworks AND for Boltz.

    python 05_template_fix.py raw_repacked.cif -o ..._repacked_fixed.cif
    python 05_template_fix.py ..._fixed.cif --check          # check only

Three errors, all coming from the same step (gemmi rewrites the output of
repack_template.py) and each of which has been found separately before:

1. `label_asym_id` = 'Axp'..'Fxp'.  setup_entities() assigns new subchain names.
   atomworks reads chain_id from label_asym_id, not from auth_asym_id, and this
   cannot be switched by an argument.
   -> LOUD error: ValueError: ID 'F' did not match any entries.

2. `label_seq_id` = '.'.  Loaders then number sequentially from 1.
   -> SILENT error: F95 silently points into a chain numbered 1-33. It was
      only noticed on 20.09. because error 1 happened at the same time.

3. `_entity_poly_seq` is missing entirely.  Then entity.full_sequence is empty,
   and Boltz' parse_polymer runs into `res_name = sequence[j]` with an empty sequence.
   -> MISLEADING error: "IndexError: list index out of range", and the
      traceback names a variant_*.yaml, not the template. One then searches
      the YAMLs instead of the CIF. That was the crash of 22.09.

Point 3 is the most important addition to the two-line repair from the
handover note: that repair fixes 1 and 2 but not 3. Anyone who only adds the two lines to
repack_template.py gets the Boltz crash again at the next repack.

Needs gemmi.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gemmi


def fix(src, dst=None, verbose=True):
    src = Path(src)
    st = gemmi.read_structure(str(src))
    if len(st) == 0:
        sys.exit(f"{src}: gemmi read ZERO models - the file cannot be rescued, "
                 f"the repacking has to write a complete mmCIF.")

    n_atoms_before = sum(len(r) for m in st for c in m for r in c)
    pos_before = [(a.pos.x, a.pos.y, a.pos.z) for m in st for c in m for r in c for a in r]

    st.setup_entities()

    # (1) + (2): label == auth, in both columns
    for model in st:
        for ch in model:
            for res in ch:
                res.subchain = ch.name
                res.label_seq = res.seqid.num
    st.setup_entities()

    # (3) Fill _entity_poly_seq. setup_entities() creates the entities, but
    # full_sequence stays empty if it did not come from the file.
    by_sub = {}
    for model in st:
        for ch in model:
            for res in ch:
                by_sub.setdefault(res.subchain, []).append(res.name)
        break
    n_filled = 0
    for ent in st.entities:
        if ent.full_sequence:
            continue
        names = []
        for sub in ent.subchains:
            if sub in by_sub:
                names = by_sub[sub]
                break
        if names:
            ent.full_sequence = list(names)
            n_filled += 1

    if dst is None:
        dst = src.with_name(src.stem + "_fixed.cif")
    dst = Path(dst)
    st.make_mmcif_document().write_file(str(dst))

    # ---- verify -------------------------------------------------------------
    chk = gemmi.read_structure(str(dst))
    chk.setup_entities()
    n_after = sum(len(r) for m in chk for c in m for r in c)
    pos_after = [(a.pos.x, a.pos.y, a.pos.z) for m in chk for c in m for r in c for a in r]
    dmax = 0.0
    if len(pos_before) == len(pos_after):
        dmax = max(max(abs(a - b) for a, b in zip(p, q))
                   for p, q in zip(pos_before, pos_after))

    if verbose:
        print(f"{src.name} -> {dst.name}")
        print(f"  atoms           {n_atoms_before} -> {n_after}")
        print(f"  max |dxyz|      {dmax:.6f}   (must be 0.000000)")
        print(f"  entity_poly_seq {n_filled} entities filled")
        for m in chk:
            for c in m:
                res = [r for r in c]
                print(f"  {c.name}: {res[0].seqid.num}-{res[-1].seqid.num} "
                      f"({len(res)} residues)")
            break
        empty = [e.name for e in chk.entities if not e.full_sequence]
        print(f"  full_sequence   "
              f"{'all filled' if not empty else 'EMPTY for ' + str(empty)}")
    if n_after != n_atoms_before or dmax > 1e-6:
        sys.exit("ABORT: the coordinates have changed. Do not use this file.")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cif")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--check", action="store_true",
                    help="check only, write nothing")
    a = ap.parse_args()

    if a.check:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import lib_checks as checks
        checks.assert_cif_columns(a.cif)
        return 0
    fix(a.cif, a.out)
    print("\nThat removes all three errors. The two lines for "
          "repack_template.py from the handover note are NOT enough - they cover 1 and 2, "
          "not _entity_poly_seq.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
