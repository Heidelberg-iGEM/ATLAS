#!/usr/bin/env python3
"""
Stage 2: lay the helix palette onto the exposed positions.

    python 11_mpnn_helicity.py --config config.yaml --levels high,native,low --with-t0

Every sequence from stage 2 is fanned out into three variants: the exposed
positions get a more or less helix-friendly occupation. The interface stays as
MPNN built it, so that the two axes of the experimental design do not mix.

Not a second MPNN run but a lookup table. MPNN's route to more helicity is
alanine, and that is exactly what one wants less of on the outside.

The rule preserves the charge class of the native residue: where the wild type
is positive, something positive stays there. Otherwise the axis would change
the electrostatics at the same time and could not be interpreted.

--with-t0 appends three rows with the native interface. The one with palette
'native' is the wild type and therefore runs in the same prediction run as the
designs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_registry as registry

# Helix propensity in kcal/mol, relative to alanine (Pace & Scholtz 1998).
# Smaller means more helical.
PACE = {"A": 0.00, "L": 0.21, "R": 0.21, "M": 0.24, "K": 0.26, "Q": 0.39,
        "E": 0.40, "I": 0.41, "W": 0.49, "S": 0.50, "Y": 0.53, "F": 0.54,
        "H": 0.61, "V": 0.61, "N": 0.65, "T": 0.66, "C": 0.68, "D": 0.69,
        "G": 1.00, "P": 3.16}

PALETTE = set("RKEQDNST")          # only polar residues to the outside
POS, NEG = set("KR"), set("DE")

DEFAULT_CHOICE = {"high": {"+": "R", "-": "E", "0": "Q"},
                  "low":  {"+": "K", "-": "D", "0": "T"}}
DEFAULT_NEUTRAL_ALA = {"high": 0.5, "low": 0.0}

# 'pos'/'neg'/'neutral' are allowed because a bare '-' is parsed unreliably
# as a YAML key.
ALIAS = {"pos": "+", "plus": "+", "positive": "+",
         "neg": "-", "minus": "-", "negative": "-",
         "neutral": "0", "zero": "0", "none": "0"}


def charge_class(aa: str) -> str:
    return "+" if aa in POS else "-" if aa in NEG else "0"


def budget(seq: str, offset: int, positions: list[int]) -> float:
    """Sum of the helix propensities over a set of positions."""
    return round(sum(PACE.get(seq[p - offset], 0.0) for p in positions), 2)


def load_palette(cfg):
    h = cfg.get("helicity") or {}
    known = {"choice", "neutral_ala", "levels"}
    unknown = sorted(set(h) - known)
    if unknown:
        print(f"  Note: helicity.{unknown} is not read and has no effect.")

    choice = {k: dict(v) for k, v in DEFAULT_CHOICE.items()}
    for level, mapping in (h.get("choice") or {}).items():
        if level not in choice:
            raise ValueError(f"helicity.choice[{level}] unknown, allowed: {sorted(choice)}")
        if not isinstance(mapping, dict):
            raise ValueError(f"helicity.choice[{level}] is not a mapping. "
                             "Use block notation (pos: R), not "
                             "inline mappings.")
        for k, v in mapping.items():
            key = ALIAS.get(str(k).strip().lower(), str(k).strip())
            if key not in ("+", "-", "0"):
                raise ValueError(f"helicity.choice[{level}][{k}]: allowed are "
                                 "pos, neg, neutral")
            choice[level][key] = str(v).strip()

    ala = dict(DEFAULT_NEUTRAL_ALA)
    for level, v in (h.get("neutral_ala") or {}).items():
        ala[level] = float(v)
        if not 0.0 <= ala[level] <= 1.0:
            raise ValueError(f"helicity.neutral_ala[{level}] must be between 0 and 1")

    bad = [(lv, k, v) for lv, m in choice.items() for k, v in m.items() if v not in PALETTE]
    if bad:
        raise ValueError(f"helicity.choice contains residues outside the polar "
                         f"palette {sorted(PALETTE)}: {bad}")
    return choice, ala


def ala_positions(native, offset, exposed, frac):
    """Which neutral exposed positions get an alanine.

    Alanine is highly helical and charge-neutral, so it widens the span
    between 'high' and 'low' without violating the charge condition.
    Spread evenly along the helix, otherwise an Ala block forms.
    """
    neutral = [p for p in exposed if charge_class(native[p - offset]) == "0"]
    n = int(round(frac * len(neutral)))
    if n <= 0:
        return set()
    if n >= len(neutral):
        return set(neutral)
    step = len(neutral) / n
    return {neutral[int(i * step)] for i in range(n)}


def pick(native, i, level, pos_abs, choice, ala_set):
    if pos_abs in ala_set:
        return "A"
    return choice[level][charge_class(native[i])]


def apply_palette(seq, native, offset, exposed, level, choice, ala_set):
    """Replace exposed positions. The charge classes come from the
    wild type, not from the design - otherwise the stability axis inherits
    MPNN's charge distribution."""
    if level == "native":
        out = list(seq)
        for p in exposed:
            out[p - offset] = native[p - offset]
        return "".join(out)
    out = list(seq)
    for p in exposed:
        i = p - offset
        out[i] = pick(native, i, level, p, choice, ala_set)
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--loop", default=None)
    ap.add_argument("--levels", default="high,native,low")
    ap.add_argument("--with-t0", action="store_true",
                    help="append three rows with the native interface; the "
                         "variant 'native' is the wild type")
    ap.add_argument("--show", action="store_true",
                    help="only show the levels for the wild type, write nothing")
    a = ap.parse_args()

    cfg = registry.load_config(a.config)
    loop = a.loop or cfg["project"]["loop"]
    offset = cfg["template"]["binder_range"][0]
    native = cfg["template"]["native_binder"]
    exposed = cfg["zones"]["exposed"]
    all_pos = list(range(offset, offset + len(native)))

    levels = [l.strip() for l in a.levels.split(",") if l.strip()]
    choice, neutral_ala = load_palette(cfg)
    ala_sets = {lv: ala_positions(native, offset, exposed, neutral_ala.get(lv, 0.0))
                for lv in levels if lv != "native"}

    # How much of the propensity budget lies on the outside at all?
    # If the share is small, the axis moves little.
    b_exp, b_all = budget(native, offset, exposed), budget(native, offset, all_pos)
    print(f"Wild type: {b_exp:.2f} of {b_all:.2f} kcal/mol lie on the "
          f"{len(exposed)} exposed positions ({b_exp / b_all:.0%})\n")

    for lv in levels:
        s = apply_palette(native, native, offset, exposed, lv, choice,
                          ala_sets.get(lv, set()))
        net = sum(c in POS for c in s) - sum(c in NEG for c in s)
        charged = sum(c in POS or c in NEG for c in s) / len(s)
        ala = sum(s[p - offset] == "A" for p in exposed)
        print(f"  {lv:<7} {s}")
        print(f"          budget {budget(s, offset, exposed):>5.2f}  net {net:+d}  "
              f"charged {charged:.0%}  Ala outside {ala}/{len(exposed)}")
    print("\n  Net charge and charged fraction stay the same across all levels.")

    if a.show:
        return 0

    loop_dir = Path(cfg["project"]["root"]) / "04_loops" / loop
    rows = registry.read_csv(loop_dir / "registry.csv")
    if not rows:
        sys.exit(f"No registry.csv in {loop_dir}. Run stage 1 first.")

    out, n = [], 0
    for r in rows:
        design = r["binder_seq"].strip().upper()
        for lv in levels:
            n += 1
            seq = apply_palette(design, native, offset, exposed, lv, choice,
                                ala_sets.get(lv, set()))
            new = dict(r)
            new.update({
                "seq_id": registry.make_seq_id(loop, n),
                # parent_id belongs to the lineage across rounds and is only passed
                # through here. derived_from is the origin within the round.
                "parent_id": r.get("parent_id", ""),
                "derived_from": r["seq_id"],
                "binder_seq": seq, "seq_sha8": registry.seq_sha8(seq),
                "helicity_level": lv,
                "helicity_budget_exposed": budget(seq, offset, exposed),
                "cell": f"{r.get('design_mode', r.get('cell', '?'))}_{lv}",
                "created": registry.now_iso()})
            out.append(new)

    # T0: native interface, only the outside varies. This is the only cell
    # that isolates the stability effect without touching the binding.
    n_t0 = 0
    if a.with_t0:
        for lv in levels:
            n += 1
            n_t0 += 1
            seq = apply_palette(native, native, offset, exposed, lv, choice,
                                ala_sets.get(lv, set()))
            out.append({
                "seq_id": registry.make_seq_id(loop, n), "loop": loop,
                "cell": f"T0_{lv}", "parent_id": "", "derived_from": "",
                # Only the wild type is the reference and is never selected.
                # T0_high and T0_low are real designs.
                "design_mode": "reference" if lv == "native" else "tier_T0",
                "n_designed": 0, "binder_seq": seq,
                "seq_sha8": registry.seq_sha8(seq), "helicity_level": lv,
                "helicity_budget_exposed": budget(seq, offset, exposed),
                "created": registry.now_iso(),
                "note": "wild type" if lv == "native" else f"WT interface, outside {lv}"})

    dest = loop_dir / "registry.csv"
    registry.archive(dest, loop_dir / "archive")
    registry.write_csv(dest, out)
    registry.log_stage(loop_dir, "11_mpnn_helicity",
                       {"n_in": len(rows), "n_out": len(out), "levels": levels,
                        "choice": choice, "neutral_ala": neutral_ala,
                        "n_t0": n_t0})

    print(f"\n  {len(rows)} designs x {len(levels)} levels = {len(out) - n_t0} sequences")
    if n_t0:
        print(f"  + {n_t0} T0 rows (native interface), one of them the wild type")
    print(f"  {len(out)} rows -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
