#!/usr/bin/env python3
"""
Stage 10: merge everything, apply the gate, write the order list.

    python 40_select_gate.py --config config.yaml --open    # calibration run
    python 40_select_gate.py --config config.yaml           # live

The earlier stages each put their own scores_*.csv next to the registry.csv.
Only here are they joined over seq_id into master.csv. The join reports how
full each column has become, because a join on the wrong key does not abort,
it silently enters empty strings.

THE GATE IS A COLUMN, NOT A STAGE. It deletes nothing. Every row gets
gate_pass and gate_fail and stays in the table. Adjusting a threshold
therefore costs a second instead of a GPU run - and that is necessary, since
none of the thresholds is calibrated on experimental data.

No weighted formula. Each criterion individually as an exclusion, afterwards a
single ranking axis (bz_bsa). Two confidence measures that correlate with each
other are not a second witness.

Criteria in gate.report_only keep their column but lose the veto.
Useful for columns that did not separate in the calibration run.

Selection is made in two pots:
    best ranked   the highest buried area, regardless of the cell
    diverse       first one per cell of the experimental design, then the variants
                  with the greatest distance to everything already chosen

The candidate is the INTERFACE, not the row: two variants with
the same interface and different helix palette are a paired
comparison and should both be ordered. The minimum distance min_dist applies
between different interfaces.

Output:
    master.csv    everything, unfiltered
    gate.csv      plus gate_pass and gate_fail
    selected.csv  the selected designs
    order.csv     what goes to the lab, including controls and failures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_registry as registry

# (column, direction, config key, plain text)
CRITERIA = [
    ("bz_shape_rmsd",      "<=", "max_shape_rmsd",   "defined structure across the seeds"),
    ("bz_site_frac",       ">=", "min_site_frac",    "hit the right surface"),
    ("bz_iface_plddt",     ">=", "min_iface_plddt",  "model confidence at the interface"),
    ("af_i_pae_m1",        "<=", "max_i_pae",        "does the sequence hold the pose"),
    ("af_hotspot_rmsd_m1", "<=", "max_hotspot_rmsd", "does the binder stay at the epitope"),
]

# Rows that run along and are reported but are never a candidate.
NO_SELECT = ("reference", "probe", "control")


def fnum(row, key):
    v = row.get(key, "")
    if v in ("", None):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f          # drop NaN


def quantiles(values):
    if not values:
        return None
    s = sorted(values)

    def q(p):
        i = p * (len(s) - 1)
        lo, hi = int(i), min(int(i) + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (i - lo)

    return dict(n=len(s), min=s[0], q25=q(.25), med=q(.5), q75=q(.75), max=s[-1])


def cluster(rows, threshold):
    """Greedy by ranking axis: each sequence goes into the first cluster whose
    representative is similar enough to it. Diagnosis only - the distance filter
    further below is what really separates."""
    reps, assign = [], {}
    for r in rows:
        seq = r["binder_seq"]
        for i, rep in enumerate(reps):
            if registry.seq_identity(seq, rep) >= threshold:
                assign[r["seq_id"]] = i
                break
        else:
            reps.append(seq)
            assign[r["seq_id"]] = len(reps) - 1
    return assign, reps


def iface_signature(seq, idx):
    """The residues at the interface positions. Two variants with the same
    signature are identical at the interface, whatever is on the outside."""
    return tuple(seq[i] for i in idx if 0 <= i < len(seq))


def iface_dist(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def far_enough(sig, chosen, k):
    return all(iface_dist(sig, s) >= k for s in chosen)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--loop", default=None)
    ap.add_argument("--open", action="store_true",
                    help="calibration run: all thresholds set to 'passes'")
    ap.add_argument("--allow-missing", dest="require_criteria", action="store_false",
                    default=None,
                    help="allow missing criteria columns, e.g. as long as "
                         "stage 9 has not run")
    ap.add_argument("--n-total", type=int, default=None,
                    help="overrides select.n_total")
    a = ap.parse_args()

    cfg = registry.load_config(a.config)
    sel_cfg = cfg.get("select") or {}

    def opt(key, default):
        v = sel_cfg.get(key)
        return default if v is None else v

    n_total = a.n_total or int(opt("n_total", 30))
    n_controls = int(opt("n_controls", 7))
    min_dist = int(opt("min_dist", 2))
    n_medium = int(opt("n_medium", 2))
    per_cell = int(opt("per_cell", 1))
    n_fail = int(opt("n_fail", 0))
    max_comp_fail = int(opt("max_comp_fail", 2))
    cluster_id = float(opt("cluster_id", 0.70))
    fail_mode = str(opt("fail_mode", "spread"))
    if fail_mode not in ("borderline", "spread", "severe"):
        sys.exit(f"select.fail_mode '{fail_mode}' unknown")
    ft = opt("fail_tiers", "")
    fail_tiers = ([str(x).strip() for x in ft] if isinstance(ft, list)
                  else [x.strip() for x in str(ft).split(",") if x.strip()])
    n_fail_tiers = int(opt("n_fail_tiers", 0)) or None
    lv = opt("helicity_levels", "high,low")
    levels = ",".join(str(x) for x in lv) if isinstance(lv, list) else str(lv)
    allowed_levels = (None if levels.strip().lower() in ("all", "*", "")
                      else {x.strip().lower() for x in levels.split(",") if x.strip()})

    loop = a.loop or cfg["project"]["loop"]
    loop_dir = Path(cfg["project"]["root"]) / "04_loops" / loop
    g = dict(cfg["gate"])
    min_chain_seeds = int(g.get("min_chain_seeds", 4))
    expected_seeds = int(cfg["boltz"]["diffusion_samples"])
    report_only = {str(c).strip() for c in (g.get("report_only") or [])}
    require_criteria = (bool(g.get("require_all_criteria", False))
                        if a.require_criteria is None else a.require_criteria)

    unknown = report_only - {c for c, _, _, _ in CRITERIA}
    if unknown:
        sys.exit(f"gate.report_only names unknown criteria {sorted(unknown)}")

    offset = cfg["template"]["binder_range"][0]
    iface_abs = sorted(cfg["zones"]["interface"])
    iface_idx = [p - offset for p in iface_abs]
    native = cfg["template"]["native_binder"]

    # --- Merge ---------------------------------------------------------------
    rows = registry.join_all(loop_dir)

    exp = registry.attach_experimental(rows, loop_dir.parent)
    if exp["n"]:
        print(f"  {exp['n']} of {exp['n_exp']} lab measurements matched")
    elif exp["n_exp"]:
        print(f"  Warning: {registry.EXPERIMENTAL} has {exp['n_exp']} rows, "
              "but none matches a sequence of this round.")
    else:
        print("  No lab measurements - the thresholds stay uncalibrated.")
    if exp["orphan"]:
        print(f"  Check: {len(exp['orphan'])} measurements without a matching sequence, "
              f"e.g. {', '.join(exp['orphan'][:5])}")

    n_short = 0
    for r in rows:
        seq = (r.get("binder_seq") or "").strip().upper()
        r["identity_to_wt"] = registry.identity_to_wt(seq, native) if seq else ""
        if seq and len(seq) == len(native):
            sig = iface_signature(seq, iface_idx)
            r["_sig"] = sig
            r["iface_motif"] = "".join(sig)
        else:
            r["_sig"], r["iface_motif"] = None, ""
            n_short += seq != ""
    if n_short:
        print(f"  ! {n_short} rows with deviating sequence length, for which "
              "the distance filter is not defined.")

    # Cells of the experimental design: binding axis, plus the helicity level if
    # it is not already part of 'cell'.
    strata = ["cell"]
    hel = registry.find_helicity_col(rows)
    if hel:
        if len({registry.stratum(r, ["cell", hel]) for r in rows}) > \
           len({registry.stratum(r, ["cell"]) for r in rows}):
            strata.append(hel)

    registry.write_csv(loop_dir / "master.csv", rows)
    print(f"\n{loop}: {len(rows)} rows, {len(rows[0])} columns -> master.csv")
    print(f"Cells over {strata}")
    print(f"Distance filter: at least {min_dist} differing interface positions "
          f"out of {len(iface_idx)}")
    print(f"Orderable helicity levels: "
          f"{'all' if allowed_levels is None else ', '.join(sorted(allowed_levels))}\n")

    # --- Distributions --------------------------------------------------------
    print("Distributions")
    hdr = (f'  {"Column":<22}{"n":>5}{"min":>9}{"q25":>9}{"med":>9}{"q75":>9}'
           f'{"max":>9}   Role')
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    present = {}
    for col, _, key, _ in CRITERIA + [("bz_bsa", "", "", ""),
                                      ("identity_to_wt", "", "", "")]:
        qs = quantiles([v for r in rows if (v := fnum(r, col)) is not None])
        present[col] = qs
        role = ("   report only" if col in report_only else
                "   ranking axis" if col == "bz_bsa" else
                "   gate" if key else "")
        if qs is None:
            print(f'  {col:<22}{"missing, criterion skipped":>50}{role}')
        else:
            print(f'  {col:<22}{qs["n"]:>5}{qs["min"]:>9.2f}{qs["q25"]:>9.2f}'
                  f'{qs["med"]:>9.2f}{qs["q75"]:>9.2f}{qs["max"]:>9.2f}{role}')
    print()

    # --- Gate ----------------------------------------------------------------
    fail_count = {}
    for r in rows:
        fails = []
        is_lab = str(r.get("design_mode", "")).lower().startswith("lab")
        r["lab_reference"] = is_lab

        if str(r.get("pass_comp", "")).lower() not in ("true", "1", ""):
            if is_lab:
                r["comp_note"] = "Composition broken, kept as lab variant"
            else:
                fails.append("composition")
        if r.get("dup_of") and not is_lab:
            fails.append(f"duplicate_of_{r['dup_of']}")

        if not a.open:
            for col, direction, key, _ in CRITERIA:
                if col in report_only or present.get(col) is None:
                    continue
                v = fnum(r, col)
                if v is None:
                    fails.append(f"{col}_missing")
                    continue
                thr = float(g[key])
                if (direction == "<=" and v > thr) or (direction == ">=" and v < thr):
                    fails.append(f"{col}={v:g}{'>' if direction == '<=' else '<'}{thr:g}")

            # In how many seeds are all target chains hit?
            hit = fnum(r, "reg_chains_hit_seeds")
            if hit is not None:
                if hit < min_chain_seeds:
                    fails.append(f"chains only in {hit:g}/{expected_seeds} seeds "
                                 f"(min {min_chain_seeds})")
            else:
                nmin = fnum(r, "bz_n_chains_min")
                n_tgt = len(cfg["template"]["target_chains"])
                if nmin is not None and nmin < n_tgt:
                    fails.append(f"only {nmin:g}/{n_tgt} chains in the worst seed")

            # From the free Boltz run, if present: if the template was
            # ignored, all other numbers in the row are worthless.
            trmsd = fnum(r, "bzf_target_rmsd")
            if trmsd is not None and trmsd > 2.0:
                fails.append(f"bzf_target_rmsd={trmsd:g}>2")

        r["gate_pass"] = not fails
        r["gate_fail"] = ",".join(fails)
        for f in fails:
            k = f.split("=")[0].split("(")[0].strip()
            fail_count[k] = fail_count.get(k, 0) + 1

    passers = [r for r in rows if r["gate_pass"]]
    print(f"Gate{' (open, calibration run)' if a.open else ''}: "
          f"{len(passers)}/{len(rows)} pass")
    if fail_count:
        for k, v in sorted(fail_count.items(), key=lambda x: -x[1]):
            print(f"    {k:<34} {v}")

    if not a.open:
        vetoing, missing = [], []
        for col, _, key, what in CRITERIA:
            if col in report_only:
                print(f"  report only: {col} ({key}={g.get(key)})")
            elif present.get(col) is None:
                missing.append(col)
            elif not any(col in r["gate_fail"] for r in rows):
                print(f"  {col} excludes nobody. Worthless as a discriminator, "
                      f"kept as an emergency brake for '{what}'.")
            else:
                vetoing.append(col)
        if missing:
            print(f"  Missing criteria columns: {', '.join(missing)}")
            if require_criteria:
                sys.exit(
                    "\nAborted: gate.require_all_criteria is set and "
                    f"{len(missing)} criteria column(s) are missing. These rows "
                    "would pass the gate without the criterion having been "
                    "checked, and that would produce an order list "
                    "that looks stricter than it is.\n"
                    "Either run the missing stage or explicitly mark it as "
                    "incomplete with --allow-missing.")
        print(f"  Effectively excluding: {', '.join(vetoing) if vetoing else 'none'}")

    # --- Ranking axis and clusters ---------------------------------------------
    ranked = [r for r in passers if fnum(r, "bz_bsa") is not None]
    ranked.sort(key=lambda r: -fnum(r, "bz_bsa"))
    for i, r in enumerate(ranked, 1):
        r["rank_bsa"] = i
    if len(ranked) < len(passers):
        print(f"  ! {len(passers) - len(ranked)} passers without bz_bsa, not rankable")

    assign, reps = cluster(ranked, cluster_id)
    seen_cluster = set()
    for r in ranked:
        r["cluster_id"] = assign[r["seq_id"]]
        if r["cluster_id"] not in seen_cluster:
            seen_cluster.add(r["cluster_id"])
            r["cluster_rep"] = True

    # --- Selection -----------------------------------------------------------
    def selectable(r):
        if str(r.get("design_mode", "")).lower().startswith(NO_SELECT):
            return False
        if r.get("_sig") is None:
            return False
        if allowed_levels is not None:
            lvl = str(r.get("helicity_level", "")).strip().lower()
            if lvl and lvl not in allowed_levels:
                return False
        return True

    n_design = max(0, n_total - n_controls)
    n_best = sel_cfg.get("n_best")
    n_diverse = sel_cfg.get("n_diverse")
    n_best = int(n_best) if n_best is not None else None
    n_diverse = int(n_diverse) if n_diverse is not None else None
    if n_best is None or n_diverse is None or (n_best + n_diverse) != n_design:
        # The best-ranked pot saturates: beyond a certain size it only adds
        # neighbours from the same corner. The gain goes into the diverse pot.
        frac = float(sel_cfg.get("best_frac", 0.43))
        cap = int(sel_cfg.get("n_best_cap", 30))
        n_best = min(cap, max(1, round(n_design * frac)))
        n_diverse = max(0, n_design - n_best)
        print(f"  Pots re-split: {n_design} design slots -> "
              f"{n_best} best ranked + {n_diverse} diverse")

    selected, taken = [], set()
    chosen_sigs, chosen_set = [], set()
    pool = [r for r in ranked if selectable(r)]

    # Per interface the best row per palette. pool is sorted by bsa.
    by_sig = {}
    for r in pool:
        lvl = str(r.get("helicity_level", "")).strip().lower() or "-"
        by_sig.setdefault(r["_sig"], {}).setdefault(lvl, r)
    sig_list = list(by_sig)

    def tier_of(sig):
        return str(next(iter(by_sig[sig].values())).get("design_mode", "") or "?")

    def take_sig(sig, reason, budget):
        """Take one interface together with its palette states.
        Returns how many order slots that cost."""
        if budget <= 0 or sig in chosen_set or not far_enough(sig, chosen_sigs, min_dist):
            return 0
        rows_for = [by_sig[sig][lv] for lv in sorted(by_sig[sig])][:budget]
        if not rows_for:
            return 0
        chosen_sigs.append(sig)
        chosen_set.add(sig)
        for r in rows_for:
            taken.add(r["seq_id"])
            r["selected_reason"] = reason
            selected.append(r)
        return len(rows_for)

    # Pot 1: by buried area, regardless of the cell
    n_got_best = 0
    for sig in sig_list:
        if n_got_best >= n_best:
            break
        n_got_best += take_sig(sig, "best_ranked", n_best - n_got_best)

    # Pot 2: cover cells, then medium strength, then the outliers
    cell_stats = {}
    for r in rows:
        st = cell_stats.setdefault(registry.stratum(r, strata),
                                   {"n": 0, "pass": 0, "sel": 0})
        st["n"] += 1
        st["pass"] += bool(r["gate_pass"])

    n_got_div = 0
    covered = {tier_of(s) for s in chosen_sigs}
    per_tier = {}
    for sig in sig_list:
        if sig not in chosen_set:
            per_tier.setdefault(tier_of(sig), []).append(sig)
    for tier in sorted(per_tier):
        if n_got_div >= n_diverse or tier in covered:
            continue
        for sig in per_tier[tier][:max(1, per_cell)]:
            got = take_sig(sig, f"diverse_cell_{tier}", n_diverse - n_got_div)
            if got:
                n_got_div += got
                covered.add(tier)
                break

    if n_medium and n_got_div < n_diverse:
        rest = [s for s in sig_list if s not in chosen_set]
        used = 0
        for sig in rest[max(0, len(rest) // 2 - n_medium // 2):]:
            if used >= n_medium or n_got_div >= n_diverse:
                break
            got = take_sig(sig, "diverse_mid_bsa", n_diverse - n_got_div)
            n_got_div += got
            used += got

    # Not the next best by bsa, but the one furthest away -
    # otherwise everything comes from the same corner.
    while n_got_div < n_diverse:
        rest = [s for s in sig_list
                if s not in chosen_set and far_enough(s, chosen_sigs, min_dist)]
        if not rest:
            break
        best = max(rest, key=lambda s: (
            min((iface_dist(s, c) for c in chosen_sigs), default=99),
            fnum(next(iter(by_sig[s].values())), "bz_bsa") or 0.0))
        got = take_sig(best, "diverse_span", n_diverse - n_got_div)
        if not got:
            break
        n_got_div += got

    # Always include lab variants: yardstick, not candidate
    for r in rows:
        if r.get("lab_reference") and r["seq_id"] not in taken and selectable(r):
            taken.add(r["seq_id"])
            r["selected_reason"] = "lab_reference"
            selected.append(r)

    for r in selected:
        r.setdefault("selected_reason", "diverse_span")
        r["selected"] = True
        cell_stats.setdefault(registry.stratum(r, strata),
                              {"n": 0, "pass": 0, "sel": 0})["sel"] += 1
    for r in rows:
        for k, v in (("selected", False), ("cluster_id", ""), ("cluster_rep", False),
                     ("rank_bsa", ""), ("selected_reason", ""), ("comp_note", "")):
            r.setdefault(k, v)

    # --- Order list ----------------------------------------------------------
    order, seen_order = [], set()

    def order_add(r, reason):
        if r["seq_id"] in seen_order:
            return
        seen_order.add(r["seq_id"])
        order.append({**r, "order_reason": reason})

    for r in selected:
        order_add(r, r.get("selected_reason", "selected"))

    # Running along and being ordered are two different things: stage 3 sets
    # order_control to false for sequences that are only to be computed.
    n_ctrl = n_ctrl_runonly = 0
    for r in rows:
        dm = str(r.get("design_mode", "")).lower()
        if not dm.startswith(("probe", "control", "reference")):
            continue
        if str(r.get("order_control", "")).strip().lower() in ("false", "0", "no"):
            n_ctrl_runonly += 1
            continue
        order_add(r, "control_wt" if dm.startswith("reference") else "control")
        n_ctrl += 1

    # Failures. Whoever measures only passers cannot validate the gate.
    fail_groups, n_tier_taken = {}, 0

    def fail_key(r):
        return r["gate_fail"].split(",")[0].split("=")[0].split("(")[0].strip()

    def is_fail_candidate(r):
        return (not r["gate_pass"] and r["seq_id"] not in seen_order
                and not str(r.get("design_mode", "")).lower().startswith(NO_SELECT)
                and not fail_key(r).startswith("duplicate"))

    n_fail_left = n_fail
    # Certain tiers first: if one failure group dominates the list, the weak end
    # of the binding axis otherwise never gets its turn - although exactly
    # that is what failures are ordered for.
    if fail_tiers and n_fail_left > 0:
        budget = min(n_fail_left, n_fail_tiers or n_fail_left)
        cand = [r for r in rows if is_fail_candidate(r)
                and str(r.get("design_mode", "")) in fail_tiers and r.get("_sig")]
        cand.sort(key=lambda r: (len(r["gate_fail"].split(",")),
                                 fnum(r, "af_i_pae_m1") or 99.0,
                                 -(fnum(r, "bz_bsa") or 0.0)))
        grouped = {}
        for r in cand:
            grouped.setdefault(r["_sig"], []).append(r)
        for sig, group in grouped.items():
            if n_tier_taken >= budget:
                break
            if sig in chosen_set or not far_enough(sig, chosen_sigs, min_dist):
                continue
            chosen_sigs.append(sig)
            chosen_set.add(sig)
            for r in group:
                if n_tier_taken >= budget:
                    break
                k = f"{r.get('design_mode', '?')}_{fail_key(r)}"
                order_add(r, f"failure_{k}")
                fail_groups[k] = fail_groups.get(k, 0) + 1
                n_tier_taken += 1
        n_fail_left -= n_tier_taken

    if n_fail_left > 0:
        groups = {}
        for r in rows:
            if is_fail_candidate(r):
                groups.setdefault(fail_key(r) or "unknown", []).append(r)
        for grp in groups.values():
            grp.sort(key=lambda r: (len(r["gate_fail"].split(",")),
                                    -(fnum(r, "bz_bsa") or 0)))

        def pull(lst, i):
            if not lst:
                return None
            if fail_mode == "borderline":
                return lst.pop(0)        # narrowly failed, low risk
            if fail_mode == "severe":
                return lst.pop()         # clearly failed
            return lst.pop(0) if i % 2 == 0 else lst.pop()   # spread

        n_taken, i, n_comp = 0, 0, 0
        while n_taken < n_fail_left and any(groups.values()):
            progress = False
            for k in sorted(groups):
                if n_taken >= n_fail_left:
                    break
                if k.startswith("composition") and n_comp >= max_comp_fail:
                    continue
                r = pull(groups[k], i)
                if r is None:
                    continue
                order_add(r, f"failure_{k}")
                fail_groups[k] = fail_groups.get(k, 0) + 1
                n_comp += k.startswith("composition")
                n_taken += 1
                progress = True
            i += 1
            if not progress:
                break

    # Why was a variant not ordered? As a column, not as a deletion.
    sel_sigs = [(r["seq_id"], r["_sig"]) for r in selected if r.get("_sig")]
    n_blocked = 0
    for r in rows:
        sig = r.get("_sig")
        cand = [(iface_dist(sig, s), sid) for sid, s in sel_sigs
                if sid != r["seq_id"]] if sig else []
        if not cand:
            r["iface_mindist_to_selected"] = r["blocked_by"] = ""
            continue
        d, who = min(cand)
        r["iface_mindist_to_selected"] = d
        blocked = (d < min_dist and not r.get("selected")
                   and r["gate_pass"] and selectable(r))
        r["blocked_by"] = who if blocked else ""
        n_blocked += bool(blocked)

    # Upper limit: how many distinguishable interfaces stand behind the gate?
    deck = []
    for r in pool:
        if far_enough(r["_sig"], deck, min_dist):
            deck.append(r["_sig"])

    for r in rows + selected:
        r.pop("_sig", None)

    ORDER_COLS = ["seq_id", "binder_seq", "order_reason", "cell", "design_mode",
                  "helicity_level", "iface_motif", "identity_to_wt", "gate_pass",
                  "gate_fail", "rank_bsa", "cluster_id", "bz_bsa", "bz_shape_rmsd",
                  "bz_site_frac", "bz_iface_plddt", "seq_sha8", "parent_id",
                  "derived_from", "helicity_budget_exposed"]
    cols = [c for c in ORDER_COLS if any(c in r for r in order)]
    cols += [c for c in (order[0] if order else {}) if c not in cols]

    registry.write_csv(loop_dir / "gate.csv", rows)
    registry.write_csv(loop_dir / "selected.csv", selected)
    registry.write_csv(loop_dir / "order.csv", order, fieldnames=cols)
    registry.log_stage(loop_dir, "40_select_gate", {
        "n": len(rows), "n_pass": len(passers), "n_clusters": len(reps),
        "n_selected": len(selected), "open": a.open, "gate": g,
        "report_only": sorted(report_only), "min_chain_seeds": min_chain_seeds,
        "strata": strata, "min_dist": min_dist, "n_total": n_total,
        "n_best": n_best, "n_diverse": n_diverse, "n_controls": n_controls,
        "n_order": len(order), "cells": cell_stats, "fail_tiers": fail_tiers,
        "n_fail": n_fail})

    # --- Report --------------------------------------------------------------
    n_paired = sum(1 for sig in chosen_sigs
                   if sum(1 for r in selected
                          if r.get("iface_motif") == "".join(sig)) > 1)
    print(f"\nClusters at {cluster_id:.0%} identity: {len(reps)} of {len(ranked)}")
    print(f"Interfaces covered: {len(chosen_sigs)}, of which {n_paired} at more "
          "than one palette.")
    print("  Only the paired ones allow the stability comparison at the same "
          "interface.")

    n_lv = len(allowed_levels) if allowed_levels else 3
    print(f"\nUpper limit at distance {min_dist}: {len(deck)} distinguishable "
          f"interfaces behind the gate")
    print(f"  times {n_lv} orderable level(s) = {len(deck) * n_lv} constructs. "
          "More slots bring nothing beyond that.")
    if len(deck) * n_lv < n_total - n_controls:
        print(f"  ! The frame of {n_total} is already above this limit.")
    if n_blocked:
        print(f"{n_blocked} passers were displaced by the distance filter; "
              "blocked_by in gate.csv says by whom.")

    print(f"\nCells ({'|'.join(strata)})")
    print(f'  {"Cell":<20}{"n":>6}{"passes":>10}{"chosen":>10}')
    empty = [k for k, s in cell_stats.items() if s["n"] and not s["pass"]]
    for k in sorted(cell_stats):
        s = cell_stats[k]
        print(f'  {k:<20}{s["n"]:>6}{s["pass"]:>10}{s["sel"]:>10}')
    if empty:
        print(f"\n  ! Cells without a single passer: {', '.join(empty)}")
        print("    This corner of the experimental design is dead. Check before "
              "freezing the thresholds.")

    print(f"\nSelected: {len(selected)} -> selected.csv")
    ids = [v for v in (fnum(r, "identity_to_wt") for r in selected) if v is not None]
    if ids:
        print(f"  identity_to_wt from {min(ids):.2f} to {max(ids):.2f} "
              "(1.00 is the wild type)")
    for r in selected:
        iw = fnum(r, "identity_to_wt")
        print(f'  {r["seq_id"]:<10} {r.get("iface_motif", ""):<12} '
              f'bsa {fnum(r, "bz_bsa") or 0:>7.1f}  '
              f'id {iw if iw is None else f"{iw:.2f}":>5}  {r["selected_reason"]}')

    n_failures = sum(fail_groups.values())
    print(f"\nOrder balance (frame {n_total})")
    print(f'  {"best ranked":<22}{n_best:>6}{n_got_best:>6}')
    print(f'  {"diverse":<22}{n_diverse:>6}{n_got_div:>6}')
    print(f'  {"Controls":<22}{n_controls:>6}{n_ctrl + n_failures:>6}'
          f'   (of which {n_failures} failures)')
    if n_ctrl_runonly:
        print(f'  {"  run only":<22}{"-":>6}{n_ctrl_runonly:>6}')
    print(f'  {"total":<22}{n_total:>6}{len(order):>6}')
    if len(order) > n_total:
        print(f"  ! {len(order) - n_total} slots above the frame.")
    elif len(order) < n_total:
        print(f"  ! {n_total - len(order)} slots unused. Usually the distance filter "
              f"does not find enough distinguishable interfaces.")

    print(f"\nOrder list: {len(order)} -> order.csv")
    if fail_groups:
        print(f"  Failures ({fail_mode}): "
              + ", ".join(f"{k} {v}" for k, v in sorted(fail_groups.items())))
        if fail_tiers and n_tier_taken == 0:
            print("  ! No row drawn from the preferred tiers. Either they "
                  "all pass, or their interfaces are too close to "
                  "already chosen ones.")
    elif not n_fail:
        print("  Note: no failures included (select.n_fail = 0). Without them "
              "the gate cannot be validated.")

    if a.open:
        print("\nCalibration run. Now look at the distributions, enter thresholds in "
              "config.yaml, then rerun without --open.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
