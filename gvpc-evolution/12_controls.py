#!/usr/bin/env python3
"""
Stage 3: write controls as normal rows into the registry.

    python 12_controls.py --config config.yaml --n-scrambles 5 --seed 2026

This way the controls go through the same stages and the same GPU run as the
designs. That is the point: without controls in the same run, no gate threshold
can be checked.

Created are:
  control_normal_helix   EAAAK repeat, around 60 percent alanine. Deliberately
                         breaks the composition filter and shows whether a
                         structure score falls for it.
  scramble_wt_N          Permutations of the wild type: same composition,
                         different order. Hence blind to any composition
                         criterion and the honest test of whether a score
                         reads binding or only amino acid statistics.
                         Deterministic from --seed, so identical in every round.
  --extra FASTA          own controls, e.g. published variants with a known
                         effect on the measured quantity.

The wild type itself is not created here. It comes as a row with
design_mode 'reference' from stage 2 (--with-t0).

Running along and being ordered are two different things. Scrambles have to be
in the prediction run so that the thresholds can be checked, but they do not
have to be synthesized. That is controlled by --n-order-scrambles.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_registry as registry


def normal_helix(n: int) -> str:
    return ("EAAAK" * (n // 5 + 1))[:n]


def scrambles(native: str, n: int, seed: int) -> list[str]:
    """Permutations without repetition. One that happens to be the wild type
    again is excluded - that would be a silent positive control in the
    negative set."""
    rng = random.Random(seed)
    out, seen = [], {native}
    for _ in range(10000):
        if len(out) >= n:
            break
        s = "".join(rng.sample(native, len(native)))
        if s not in seen:
            seen.add(s)
            out.append(s)
    if len(out) < n:
        sys.exit(f"Only {len(out)} of {n} distinct scrambles found.")
    return out


def read_fasta(path):
    out, name, buf = [], None, []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(buf)))
            name, buf = line[1:].strip(), []
        elif line.strip():
            buf.append(line.strip())
    if name is not None:
        out.append((name, "".join(buf)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--loop", default=None)
    ap.add_argument("--n-scrambles", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-normal-helix", action="store_true")
    ap.add_argument("--n-order-scrambles", type=int, default=0,
                    help="how many scrambles are also ordered; the rest "
                         "only runs along")
    ap.add_argument("--extra", "--shift-control", dest="extra", default=None,
                    metavar="FASTA",
                    help="own control sequences, always ordered")
    ap.add_argument("--show", action="store_true", help="only show")
    a = ap.parse_args()

    cfg = registry.load_config(a.config)
    loop = a.loop or cfg["project"]["loop"]
    loop_dir = Path(cfg["project"]["root"]) / "04_loops" / loop
    native = cfg["template"]["native_binder"]
    n_res = len(native)

    # (name, sequence, note, order)
    entries = []
    if not a.no_normal_helix:
        entries.append(("control_normal_helix", normal_helix(n_res),
                        "EAAAK repeat, negative control for composition", True))
    for i, s in enumerate(scrambles(native, a.n_scrambles, a.seed), 1):
        entries.append((f"scramble_wt_{i}", s,
                        "Permutation of the wild type, same composition",
                        i <= a.n_order_scrambles))
    if a.extra:
        for name, seq in read_fasta(a.extra):
            seq = seq.strip().upper()
            if len(seq) != n_res:
                sys.exit(f"{name}: {len(seq)} residues, expected {n_res}")
            entries.append((f"extra_{name}", seq, "own control", True))

    n_order = sum(1 for e in entries if e[3])
    print(f"{loop}: {len(entries)} controls, {n_order} of them on the order list")
    for name, seq, note, order in entries:
        ala = sum(c == "A" for c in seq) / len(seq)
        print(f"  {'order' if order else 'run only':<10} {name:<24} {seq}  "
              f"Ala {ala:.0%}  {note}")

    rows = registry.read_csv(loop_dir / "registry.csv")
    if not rows:
        sys.exit(f"No registry.csv in {loop_dir}. Run stages 1 and 2 first.")

    # Is the wild type present? Without it the whole measurement series has no reference point.
    wt = [r for r in rows if str(r.get("design_mode", "")).lower().startswith("reference")]
    if wt:
        print(f"\n  Wild type present: {', '.join(r['seq_id'] for r in wt[:3])}")
    else:
        print("\n  ! No reference row in the registry. Missing if stage 2 "
              "ran without --with-t0.")
    if not a.extra:
        print("  Note: no control with a known effect in the pool "
              "(--extra empty).\n"
              "    Usually that is a preparation condition anyway and not a\n"
              "    sequence, but without a known large effect a null result\n"
              "    cannot later be told apart from an assay that is too "
              "coarse.")
    if a.show:
        return 0

    have = {r.get("seq_sha8") or registry.seq_sha8(r["binder_seq"]) for r in rows}
    n = max((int(str(r["seq_id"]).rsplit("-", 1)[-1])
             for r in rows if str(r["seq_id"]).rsplit("-", 1)[-1].isdigit()), default=0)

    added, skipped = [], []
    for name, seq, note, order in entries:
        sha = registry.seq_sha8(seq)
        if sha in have:
            skipped.append(name)
            continue
        have.add(sha)
        n += 1
        added.append({
            "seq_id": registry.make_seq_id(loop, n), "loop": loop,
            "cell": "control", "parent_id": "", "derived_from": "",
            "design_mode": "control", "n_designed": 0, "binder_seq": seq,
            "seq_sha8": sha, "helicity_level": "", "helicity_budget_exposed": "",
            "created": registry.now_iso(),
            "order_control": "true" if order else "false",
            "note": f"{name}: {note}"})

    if skipped:
        print(f"\n  {len(skipped)} already present: {', '.join(skipped)}")
    if not added:
        print("  Nothing to add.")
        return 0

    dest = loop_dir / "registry.csv"
    registry.archive(dest, loop_dir / "archive")
    registry.write_csv(dest, rows + added)
    registry.log_stage(loop_dir, "12_controls",
                       {"n_added": len(added), "n_skipped": len(skipped),
                        "n_scrambles": a.n_scrambles, "seed": a.seed,
                        "extra": a.extra})
    print(f"\n  {len(added)} rows appended -> {dest} "
          f"({len(rows) + len(added)} total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
