#!/usr/bin/env python3
"""
check_controls.py -- answers ONE question before a threshold is set:

    Is bz_shape_rmsd suitable as a gate, and is bz_bsa a clean ranking axis?

Three checks, in this order:
  1. Does the WT (design_mode = reference) pass the candidate threshold?
     The lesson from af_i_pae_m1: threshold 10.0, WT 11.42 -- the positive
     anchor failed its own filter. Any threshold that does this is out.
  2. Do the controls (control_normal_helix, the five scrambles) fail?
     That is the one job the gate still really does.
  3. Where do controls that PASS end up on the bz_bsa ranking?
     That is the catastrophic case and the part one overlooks when just
     looking: a control with a large BSA is harmless as long as it gets stuck
     at the gate. If it passes AND ranks high, the best-ranked pot takes its
     ten places from exactly this kind.

Writes nothing. stdlib only -- runs under any Python of the pipeline.

    python check_controls.py --table ../04_loops/L1/master.csv
    python check_controls.py --table ... --thresholds 1.5,2.0,2.5,3.0 --top 10

WHICH FILE
----------
NOT registry.csv. Stages 7/8 (32/33) do not append their columns there but
write scores_boltz.csv / scores_register.csv next to it; the join only happens
in 40_select_gate.py (lib_registry.join_all). The columns are therefore in
master.csv or gate.csv, and both are created by the first 40_ run -- also with
--open. Without --table the search order is:
    master.csv -> gate.csv -> registry.csv
"""

import argparse
import csv
import os
import sys

METRICS = [
    "bz_shape_rmsd",
    "bz_site_frac",
    "bz_iface_plddt",
    "bz_bsa",
    "reg_chains_hit_seeds",
    "af_i_pae_m1",
    "af_hotspot_rmsd_m1",
]

# design_mode carries the TIER for designs (40_select_gate.tier_of uses the
# column as the cell of the factorial design), not the word "design". Classification
# therefore goes through the prefixes of the non-designs -- just like
# lib_registry.CONTROL_MODES and NO_SELECT in 40_select_gate.py.
CTRL_PREFIXES = ("control", "reference", "probe")

NAME_CANDIDATES = ["control_name", "name", "label", "variant", "seq_name",
                   "note", "remark", "comment", "description"]
FALLBACK_TABLES = ["master.csv", "gate.csv", "registry.csv"]


def fnum(row, col):
    """Value as float, or None if empty/missing/unparseable."""
    v = row.get(col, "")
    if v is None:
        return None
    v = str(v).strip()
    if v == "" or v.upper() in ("NA", "NAN", "NONE", "N/A", "NULL"):
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return None if f != f else f          # drop NaN


def quantiles(vals):
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)

    def q(p):
        i = p * (n - 1)
        lo = int(i)
        hi = min(lo + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (i - lo)

    return (s[0], q(0.25), q(0.50), q(0.75), s[-1])


def mode_of(row):
    return str(row.get("design_mode", "") or "").strip().lower()


def is_ctrl(row):
    return mode_of(row).startswith(CTRL_PREFIXES)


def kind_of(row):
    """reference | control | probe | design"""
    m = mode_of(row)
    for p in CTRL_PREFIXES:
        if m.startswith(p):
            return p
    return "design"


def find_table(explicit):
    if explicit:
        return explicit
    for base in (os.path.join("..", "04_loops", "L1"), "."):
        for name in FALLBACK_TABLES:
            p = os.path.join(base, name)
            if os.path.exists(p):
                return p
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default=None,
                    help="master.csv or gate.csv (default: search "
                         "automatically)")
    ap.add_argument("--registry", dest="table",
                    help="legacy, identical to --table")
    ap.add_argument("--thresholds", default="1.5,2.0,2.5,3.0",
                    help="candidate thresholds for bz_shape_rmsd")
    ap.add_argument("--metric", default="bz_shape_rmsd",
                    help="which column is played through")
    ap.add_argument("--rank-col", default="bz_bsa",
                    help="the ranking axis for check 3")
    ap.add_argument("--top", type=int, default=10,
                    help="size of the best-ranked pot (select.n_best)")
    args = ap.parse_args()

    path = find_table(args.table)
    if not path:
        sys.exit("No table found. Searched for %s in "
                 "../04_loops/L1/ and here.\n"
                 "  master.csv is created by the first run of "
                 "40_select_gate.py (also with --open).\n"
                 "  Otherwise give the path with --table."
                 % ", ".join(FALLBACK_TABLES))

    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except IOError as e:
        sys.exit("Table not readable: %s" % e)
    if not rows:
        sys.exit("%s is empty." % path)

    header = list(rows[0])
    if "design_mode" not in header:
        sys.exit("Column 'design_mode' is missing in %s -- without it it is not "
                 "possible to tell designs from controls." % path)

    present = [m for m in METRICS if m in header]
    missing = [m for m in METRICS if m not in header]
    metric, rank_col = args.metric, args.rank_col

    print("Table: %s  (%d rows)" % (path, len(rows)))
    if missing:
        print("Columns not present: %s" % ", ".join(missing))
    if metric not in header:
        sys.exit("\n'%s' is missing. Either stage 7 (32_boltz_rank.py) has not "
                 "run yet, or\n  the table is registry.csv instead of "
                 "master.csv -- the Boltz columns are\n  in "
                 "scores_boltz.csv and are only joined by 40_select_gate.py." % metric)

    name_col = next((c for c in NAME_CANDIDATES if c in header), None)

    def label(r):
        if name_col:
            v = str(r.get(name_col, "") or "").strip()
            if v:
                return v[:26]
        return str(r.get("seq_id", "?"))

    # ---- groups -----------------------------------------------------------
    groups = {"design": [], "reference": [], "control": [], "probe": []}
    for r in rows:
        groups[kind_of(r)].append(r)

    def scored(rs):
        return [r for r in rs if fnum(r, metric) is not None]

    designs = scored(groups["design"])
    refs = scored(groups["reference"])
    ctrls = scored(groups["control"])
    probes = scored(groups["probe"])
    anchors = refs + ctrls + probes

    print("Scored with %s: %d designs, %d reference, %d control, %d probe"
          % (metric, len(designs), len(refs), len(ctrls), len(probes)))
    print("  design_mode values: %s"
          % ", ".join(sorted({mode_of(r) for r in rows if mode_of(r)})[:12]))
    if not designs:
        print("  ! No designs scored. Either the stage did not run, or "
              "design_mode\n    starts with control/reference/probe for all of them "
              "- then the classification is wrong.")
    if not ctrls and not refs:
        print("  ! WARNING: neither WT nor controls scored. The check runs "
              "empty.\n    Check whether 12_controls.py ran and the controls were in the "
              "SAME Boltz run.")
    print()

    # ---- 1. Distributions -------------------------------------------------
    print("=" * 74)
    print("DISTRIBUTIONS -- designs against anchors and controls")
    print("=" * 74)
    for m in present:
        dq = quantiles([v for r in designs if (v := fnum(r, m)) is not None])
        print()
        print("%s" % m)
        if dq:
            print("  Designs (n=%d)  min %8.3f  q25 %8.3f  med %8.3f  "
                  "q75 %8.3f  max %8.3f"
                  % (len(designs), dq[0], dq[1], dq[2], dq[3], dq[4]))
        else:
            print("  Designs: no values")
        for r in anchors:
            v = fnum(r, m)
            print("  %-10s %-28s %s"
                  % (kind_of(r), label(r),
                     "%8.3f" % v if v is not None else "       -"))

    # ---- 2. Threshold play-through ----------------------------------------
    print()
    print("=" * 74)
    print("CHECK 1+2: does %s keep the WT in and the controls out?" % metric)
    print("=" * 74)
    print()
    print("  %-10s %-16s %-16s %s"
          % ("Threshold", "WT", "Controls", "Designs"))
    print("  " + "-" * 60)

    try:
        thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    except ValueError:
        sys.exit("--thresholds not readable: %r" % args.thresholds)

    # Controls = control + probe; the WT is reference and is listed separately.
    neg = ctrls + probes
    for t in thresholds:
        wt = []
        for r in refs:
            v = fnum(r, metric)
            wt.append("passes" if v is not None and v <= t else "FAILS")
        wt_txt = "/".join(wt) if wt else "no ref"
        c_fail = sum(1 for r in neg
                     if fnum(r, metric) is None or fnum(r, metric) > t)
        d_pass = sum(1 for r in designs
                     if fnum(r, metric) is not None and fnum(r, metric) <= t)
        print("  %-10.2f %-16s %-16s %d of %d"
              % (t, wt_txt, "%d/%d out" % (c_fail, len(neg)),
                 d_pass, len(designs)))

    print()
    print("  Reading: every threshold at which the WT FAILS is out -- the")
    print("  positive anchor has to pass its own filter. Among the")
    print("  remaining ones take the one that removes the most controls.")
    print("  If NO threshold separates the WT from the controls, no other number")
    print("  helps either: a threshold shifts the level, not the order.")
    print("  Same class of error as af_hotspot_rmsd_m1.")

    # ---- 3. Rank of the passers -------------------------------------------
    if rank_col not in header:
        print("\n%s is missing -- check 3 not possible." % rank_col)
        return

    print()
    print("=" * 74)
    print("CHECK 3: are passing controls at the top of the "
          "%s ranking?" % rank_col)
    print("=" * 74)
    print()
    print("  A control with a large %s is harmless as long as it gets stuck at the gate"
          % rank_col)
    print("  -- the ranking axis only applies BEHIND it. The damage")
    print("  only happens when it passes AND ranks high.")

    for t in thresholds:
        pool = [r for r in rows
                if fnum(r, metric) is not None and fnum(r, metric) <= t
                and fnum(r, rank_col) is not None]
        pool.sort(key=lambda r: -fnum(r, rank_col))
        bad = [(i + 1, r) for i, r in enumerate(pool) if is_ctrl(r)
               and kind_of(r) != "reference"]
        wt_rank = [(i + 1, r) for i, r in enumerate(pool)
                   if kind_of(r) == "reference"]

        print()
        print("  Threshold %.2f -- %d rows behind the gate" % (t, len(pool)))
        for rank, r in wt_rank:
            print("    WT at rank %d/%d (%s %.1f)"
                  % (rank, len(pool), rank_col, fnum(r, rank_col)))
        if not bad:
            print("    No control passes. Clean.")
            continue
        for rank, r in bad:
            flag = "   <<< IN THE TOP %d" % args.top if rank <= args.top else ""
            print("    Rank %3d/%3d  %-10s %-28s %s %7.1f%s"
                  % (rank, len(pool), kind_of(r), label(r), rank_col,
                     fnum(r, rank_col), flag))

    print()
    print("  If a control is in the top %d, %s is compromised as a ranking axis"
          % (args.top, rank_col))
    print("  -- then the split %d best-ranked /"
          % args.top)
    print("  13 diverse has to be put on the table again before ordering.")
    print("  If it is further down, it is harmless: the best-ranked pot")
    print("  draws from the top, the diverse one via the distance filter.")
    print()


if __name__ == "__main__":
    main()
