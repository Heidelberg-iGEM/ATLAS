#!/usr/bin/env python3
"""
Stage 5: write input files for Boltz.

    $BOLTZ_PY 30_boltz_yaml.py --config config.yaml --mode constrained

One YAML per sequence, in a freshly created directory. This is not tidiness for
its own sake: boltz predict takes a directory and uses everything in it,
including Jupyter checkpoints and copies of older versions.

The sequences of the target chains come from the template, not from the
registry. That makes every run incidentally a test of whether the chain made it
unchanged from the template to the YAML.

--mode free omits the pocket block. The template stays in, otherwise the target
does not stand and the numbers describe a structure that does not exist.

MSA: boltz.msa_policy decides whether only the binder runs without an MSA
(empty_binder_only) or all chains (empty_all). A designed sequence has no real
MSA, the target does. The setting has to stay the same across all rounds,
otherwise two different Boltz configurations are being compared.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_registry as registry
import lib_checks as checks


def template_sequences(cif_path, chains):
    """Sequence of each target chain directly from the template."""
    import gemmi
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    if len(st) == 0:
        raise SystemExit(
            f"{cif_path}: gemmi read ZERO models. A file reduced to atom_site "
            "fails here, and Boltz dies later with a meaningless "
            "message.")
    by_name = {c.name: c for c in st[0]}
    missing = [c for c in chains if c not in by_name]
    if missing:
        raise SystemExit(f"{cif_path}: chains {missing} missing, "
                         f"present {sorted(by_name)}")
    out = {}
    for c in chains:
        res = [r for r in by_name[c] if r.find_atom("CA", "*")]
        out[c] = gemmi.one_letter_code([r.name for r in res])
    return out


def msa_query(path):
    """First sequence from an MSA file (.a3m or Boltz CSV).

    Only for the cross-check: an MSA whose query is a different sequence
    is accepted by Boltz without complaint, and the profile columns then point
    at the wrong residues.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        import csv as _csv
        with path.open(encoding="utf-8", errors="replace", newline="") as fh:
            reader = _csv.DictReader(fh)
            col = next((c for c in (reader.fieldnames or [])
                        if str(c).strip().lower() in ("sequence", "seq")), None)
            if col is None:
                return "", f"no column 'sequence', found: {reader.fieldnames}"
            for row in reader:
                v = (row.get(col) or "").strip()
                if v:
                    return "".join(c for c in v if c.isupper()), ""
            return "", "CSV without a data row"

    query, started = "", False
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line.startswith(">"):
                if started:
                    break
                started = True
                continue
            if started:
                query += line
    # a3m: lowercase letters are insertions, '-' are gaps
    return "".join(c for c in query if c.isupper()), ("" if query else "no query found")


def write_yaml(path, binder_chain, binder_seq, tseqs, template, pocket,
               msa_policy, target_msa=""):
    binder_empty = msa_policy in ("empty_binder_only", "empty_all")
    target_empty = msa_policy == "empty_all"

    lines = ["version: 1", "sequences:"]
    for cid, seq in tseqs.items():
        lines += ["  - protein:", f"      id: {cid}", f"      sequence: {seq}"]
        if target_empty:
            lines += ["      msa: empty"]
        elif target_msa:
            lines += [f"      msa: {target_msa}"]
    lines += ["  - protein:", f"      id: {binder_chain}",
              f"      sequence: {binder_seq}"]
    if binder_empty:
        lines += ["      msa: empty"]

    ids = ", ".join(tseqs)
    lines += ["templates:", f"  - cif: {template}",
              f"    chain_id: [{ids}]", f"    template_id: [{ids}]"]
    if pocket:
        contacts = ", ".join(f"[{c}, {r}]" for c, r in pocket)
        lines += ["constraints:", "  - pocket:", f"      binder: {binder_chain}",
                  f"      contacts: [{contacts}]"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--loop", default=None)
    ap.add_argument("--mode", default="constrained", choices=["constrained", "free"])
    a = ap.parse_args()

    cfg = registry.load_config(a.config)
    loop = a.loop or cfg["project"]["loop"]
    root = Path(cfg["project"]["root"])
    loop_dir = root / "04_loops" / loop
    tpl = cfg["template"]
    binder, tchains, brange = tpl["binder_chain"], tpl["target_chains"], tpl["binder_range"]
    msa_policy = cfg["boltz"]["msa_policy"]

    cif = registry.resolve(cfg, tpl["cif"])
    if not cif.exists():
        sys.exit(f"Template missing: {cif}")
    checks.assert_cif_columns(cif, binder=binder, targets=tchains, binder_range=brange)

    tseqs = template_sequences(cif, tchains)
    print(f"[boltz] Template {cif.name}: "
          + ", ".join(f"{c} {len(s)} aa" for c, s in tseqs.items()))

    # Precomputed target MSA. Without it Boltz asks an external server for each
    # prediction, and every round gets a different MSA.
    target_msa = str(cfg["boltz"].get("target_msa") or "").strip()
    if target_msa:
        msa_path = registry.resolve(cfg, target_msa)
        if not msa_path.exists():
            sys.exit(f"boltz.target_msa points to {msa_path}, the file is missing. "
                     "Either back up the MSA or leave the key empty.")
        query, err = msa_query(msa_path)
        if err:
            sys.exit(f"{msa_path}: {err}")
        want = next(iter(tseqs.values()))
        if query != want:
            same = sum(1 for x, y in zip(query, want) if x == y)
            sys.exit(f"{msa_path}: the query row is not the target sequence "
                     f"({same}/{min(len(query), len(want))} positions equal). "
                     "The MSA probably belongs to a different template.")
        print(f"[boltz] target MSA {msa_path.name} matches the template")
        target_msa = str(msa_path)
    else:
        print("[boltz] no precomputed target MSA, Boltz asks the server")

    # Prefilter passers plus controls
    rows = registry.join_all(loop_dir)
    sel = registry.to_predict(rows)
    if not sel:
        sys.exit("No prefilter passers. Run stage 4 first.")
    n_ctrl = sum(1 for r in sel if registry.is_control(r))
    print(f"[boltz] {len(sel)} sequences, {n_ctrl} of them controls")

    pocket = None
    if a.mode == "constrained":
        pocket = [(c, int(r)) for c, r in cfg["epitope"]["pocket_contacts"]]
        bad = sorted({c for c, _ in pocket if c not in tchains})
        if bad:
            sys.exit(f"pocket_contacts name chains {bad} that are not in "
                     f"target_chains {tchains}")

    indir = registry.fresh_dir(root / "02_input" / "boltz_input" / f"{loop}_{a.mode}",
                               root / "03_output" / "archive")
    expected_len = brange[1] - brange[0] + 1
    n_written = 0
    for r in sel:
        seq = r["binder_seq"].strip().upper()
        if len(seq) != expected_len:
            print(f"  ! {r['seq_id']}: {len(seq)} residues, expected {expected_len}",
                  file=sys.stderr)
            continue
        write_yaml(indir / f"{r['seq_id']}.yaml", binder, seq, tseqs, cif,
                   pocket, msa_policy, target_msa)
        n_written += 1

    registry.log_stage(loop_dir, f"30_boltz_yaml_{a.mode}",
                       {"n_in": len(rows), "n_written": n_written, "mode": a.mode,
                        "msa_policy": msa_policy, "target_msa": target_msa,
                        "template": str(cif),
                        "pocket_contacts": len(pocket) if pocket else 0})
    print(f"\n[boltz] {n_written} YAMLs -> {indir}")
    print(f"        mode {a.mode}"
          + (f", {len(pocket)} pocket contacts" if pocket else ", no pocket constraint"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
