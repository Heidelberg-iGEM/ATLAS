"""
lib_checks.py - checks that make silent errors loud.

Two classes of error keep showing up with structure files, and one of them
does not abort:

  loud:   label_asym_id is 'Fxp' instead of 'F'  ->  ValueError on loading
  silent: label_seq_id is empty              ->  the chain is silently renumbered
                                                from 1, and all position
                                                specifications point at the
                                                wrong residues

The first time, the silent error is only noticed because the loud one happens
at the same time. One must not rely on that, so everything is checked after
every load, and the check aborts instead of warning.

assert_structure expects an atomworks or biotite AtomArray. The module
itself imports neither numpy nor gemmi, so that it stays importable from all
four conda envs; assert_cif_columns and assert_pdb_numbering work purely
textually.

    python lib_checks.py template.cif
"""

from __future__ import annotations

from pathlib import Path


class CheckFailed(AssertionError):
    pass


AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
       "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
       "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
       "TYR": "Y", "VAL": "V"}


# --------------------------------------------------------------------------
# After loading a structure
# --------------------------------------------------------------------------

def assert_structure(atom_array, binder="F", targets=("A", "B", "C", "D"),
                     binder_range=(92, 124), n_binder_res=33, label=""):
    """Chains and number range after loading."""
    chains = sorted({str(c) for c in atom_array.chain_id})
    want = sorted({binder, *targets})
    head = f"[checks{' ' + label if label else ''}]"
    print(f"{head} chains: {chains}")

    if chains != want:
        raise CheckFailed(
            f"{head} chains {chains}, expected {want}.\n"
            "  If names like 'Axp' to 'Fxp' appear here, gemmi's "
            "setup_entities() has assigned new subchain names. atomworks reads "
            "label_asym_id, not auth_asym_id. Repair with "
            "05_template_fix.py.")

    b = atom_array[atom_array.chain_id == binder]
    lo, hi = int(b.res_id.min()), int(b.res_id.max())
    n_res = len({int(r) for r in b.res_id})
    print(f"{head} {binder}: {lo}-{hi}, {n_res} residues")

    if (lo, hi) != tuple(binder_range):
        raise CheckFailed(
            f"{head} {binder} is numbered {lo}-{hi}, expected "
            f"{binder_range[0]}-{binder_range[1]}.\n"
            "  Numbering from 1 means: label_seq_id was empty and the "
            "loader renumbered. This error otherwise does NOT abort, "
            "the position specifications then silently point at wrong residues.")
    if n_res != n_binder_res:
        raise CheckFailed(f"{head} {binder} has {n_res} residues, expected "
                          f"{n_binder_res}.")
    return True


def assert_binder_sequence(atom_array, expected, binder="F", label=""):
    """Additionally check the sequence. Catches a swapped template
    that the numbering alone does not reveal."""
    b = atom_array[atom_array.chain_id == binder]
    seen, seq = set(), []
    for rid, rname in zip(b.res_id, b.res_name):
        rid = int(rid)
        if rid not in seen:
            seen.add(rid)
            seq.append(AA3.get(str(rname).upper(), "X"))
    got = "".join(seq)
    head = f"[checks{' ' + label if label else ''}]"
    print(f"{head} {binder}: {got}")
    if got != expected:
        raise CheckFailed(
            f"{head} binder sequence deviates.\n  expected: {expected}\n"
            f"  read    : {got}\n  Either the template was swapped or "
            "the chain is assigned wrongly.")
    return True


# --------------------------------------------------------------------------
# Purely textual, needs no env
# --------------------------------------------------------------------------

def assert_cif_columns(cif_path, binder="F", targets=("A", "B", "C", "D"),
                       binder_range=(92, 124)):
    """Checks directly in the file that the label and auth columns agree.

    Belongs before every run on a newly generated template.
    """
    path = Path(cif_path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    cols, rows, in_loop = [], [], False
    for line in lines:
        s = line.strip()
        if s.startswith("_atom_site."):
            cols.append(s.split(".", 1)[1].split()[0])
            in_loop = True
        elif in_loop:
            if not s or s.startswith(("#", "loop_")):
                break
            rows.append(s.split())
    if not cols or not rows:
        raise CheckFailed(f"[checks] no _atom_site loop in {path}")

    def idx(name):
        return cols.index(name) if name in cols else None

    i_lab, i_auth = idx("label_asym_id"), idx("auth_asym_id")
    i_lseq, i_aseq = idx("label_seq_id"), idx("auth_seq_id")

    lab_chains, auth_chains = set(), set()
    binder_auth = set()
    mismatch_chain = mismatch_seq = empty_lseq = 0

    for r in rows:
        if i_lab is not None and i_lab < len(r):
            lab_chains.add(r[i_lab])
        if i_auth is not None and i_auth < len(r):
            auth_chains.add(r[i_auth])
        if i_lab is not None and i_auth is not None and r[i_lab] != r[i_auth]:
            mismatch_chain += 1
        if i_lseq is not None and i_lseq < len(r):
            if r[i_lseq] in (".", "?"):
                empty_lseq += 1
            elif i_aseq is not None and r[i_lseq] != r[i_aseq]:
                mismatch_seq += 1
        if i_auth is not None and r[i_auth] == binder and i_aseq is not None:
            try:
                binder_auth.add(int(r[i_aseq]))
            except ValueError:
                pass

    print(f"[checks] {path.name}")
    print(f"[checks]   label_asym_id: {sorted(lab_chains)}")
    print(f"[checks]   auth_asym_id : {sorted(auth_chains)}")
    if binder_auth:
        print(f"[checks]   {binder}: {min(binder_auth)}-{max(binder_auth)}")

    want = sorted({binder, *targets})
    problems = []
    if sorted(auth_chains) != want:
        problems.append(f"auth_asym_id {sorted(auth_chains)} != {want}")
    if sorted(lab_chains) != want:
        problems.append(f"label_asym_id {sorted(lab_chains)} != {want} "
                        "(atomworks reads this column)")
    if mismatch_chain:
        problems.append(f"{mismatch_chain} atoms with label_asym_id != auth_asym_id")
    if empty_lseq:
        problems.append(f"{empty_lseq} atoms with empty label_seq_id, loaders "
                        "then silently number from 1")
    if mismatch_seq:
        problems.append(f"{mismatch_seq} atoms with label_seq_id != auth_seq_id")
    if binder_auth and (min(binder_auth), max(binder_auth)) != tuple(binder_range):
        problems.append(f"{binder} {min(binder_auth)}-{max(binder_auth)} "
                        f"!= {binder_range}")
    if problems:
        raise CheckFailed("[checks] template not usable:\n  - "
                          + "\n  - ".join(problems)
                          + "\n  Repair: 05_template_fix.py")

# Second, independent check. The column check cannot tell a file reduced to
# atom_site from a complete mmCIF, Boltz can - but only after the waiting time.
    try:
        import gemmi
    except ImportError:
        print("[checks]   (gemmi not in env, structure cross-check skipped)")
    else:
        st = gemmi.read_structure(str(path))
        if len(st) == 0:
            raise CheckFailed(
                f"[checks] {path.name}: gemmi read ZERO models. The columns "
                "are fine, but the file is not a complete mmCIF.")
        st.setup_entities()
        empty = [e.name for e in st.entities if not e.full_sequence]
        if empty:
            raise CheckFailed(
                f"[checks] {path.name}: _entity_poly_seq missing or empty "
                f"(entities {empty}). Boltz aborts in parse_polymer because of this, "
                "and the traceback names a variant YAML instead of the template.")

    print("[checks]   ok")
    return True


def assert_pdb_numbering(pdb_path, binder="F", targets=("A", "B", "C", "D"),
                         binder_range=(92, 124), target_range=(1, 69)):
    """After the conversion CIF to PDB for AF2.

    21_af2_score.py reads the residue numbers directly from the file. If a
    converter renames chains or renumbers, the hotspots are silently wrong.
    """
    path = Path(pdb_path)
    per_chain = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            per_chain.setdefault(line[21], set()).add(int(line[22:26]))
        except ValueError:
            continue

    print(f"[checks] {path.name}")
    for ch in sorted(per_chain):
        r = per_chain[ch]
        print(f"[checks]   {ch}: {min(r)}-{max(r)}, {len(r)} residues")

    want = sorted({binder, *targets})
    problems = []
    if sorted(per_chain) != want:
        problems.append(f"chains {sorted(per_chain)} != {want}")
    if binder in per_chain:
        r = per_chain[binder]
        if (min(r), max(r)) != tuple(binder_range):
            problems.append(f"{binder} {min(r)}-{max(r)} != {binder_range}")
    for t in targets:
        if t in per_chain:
            r = per_chain[t]
            if (min(r), max(r)) != tuple(target_range):
                problems.append(f"{t} {min(r)}-{max(r)} != {target_range}")
    if problems:
        raise CheckFailed(
            "[checks] The conversion changed chains or numbering:\n  - "
            + "\n  - ".join(problems)
            + "\n  The af2_hotspots in config.yaml refer to this "
              "numbering.")
    print("[checks]   ok")
    return True


# --------------------------------------------------------------------------
# MPNN configuration
# --------------------------------------------------------------------------

def assert_mpnn_keys(cfg_dict, allowed, where):
    """MPNN silently discards unknown keys. A typo such as
    'atomize_side_chian' then runs on quietly with the default and invalidates
    the whole run, without a single line of error message."""
    unknown = set(cfg_dict) - allowed
    print(f"[checks] {where}: {sorted(cfg_dict)}")
    if unknown:
        raise CheckFailed(f"[checks] unknown keys in {where}: "
                          f"{sorted(unknown)}\n  allowed: {sorted(allowed)}")
    return True


def assert_design_constraints(input_cfg):
    """designed_chains and residue-based constraints are mutually exclusive.
    For the tier grading designed_chains has to go."""
    has_chain = bool(input_cfg.get("designed_chains"))
    has_res = bool(input_cfg.get("designed_residues") or input_cfg.get("fixed_residues"))
    if has_chain and has_res:
        raise CheckFailed(
            "[checks] designed_chains together with designed_residues: MPNN "
            "raises 'Cannot mix residue-based and chain-based design "
            "constraints' here.")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage:  python lib_checks.py <template.cif|template.pdb>")
        sys.exit(0)
    p = Path(sys.argv[1])
    if p.suffix.lower() in (".cif", ".mmcif"):
        assert_cif_columns(p)
    else:
        assert_pdb_numbering(p)
