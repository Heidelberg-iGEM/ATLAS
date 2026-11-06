#!/usr/bin/env python3
"""What does the factorial design look like AFTER the composition filter?

The one question before the GPU run: is every cell still occupied? An empty
cell cannot be repaired after Boltz - a column of the experimental design is
missing then, and the effect it was meant to isolate can no longer be
separated.

Usage (from 00_pipeline/, in the boltz env):

    python cells_after_filter.py --config config.yaml --loop L1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_registry as registry


def truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--loop", default="L1")
    ap.add_argument("--with-controls", action="store_true",
                    help="count controls in the factorial design. They are not "
                         "a cell of the experimental design and are otherwise "
                         "reported separately.")
    a = ap.parse_args()

    cfg = registry.load_config(a.config)
    loop_dir = Path(cfg["project"]["root"]) / "04_loops" / a.loop

    reg = registry.read_csv(loop_dir / "registry.csv")
    flt = {r["seq_id"]: r for r in registry.read_csv(loop_dir / "scores_filter.csv")}
    if not reg:
        print("no registry.csv"); return 1
    if not flt:
        print("no scores_filter.csv - run stage 4 first"); return 1

    # Controls are not a cell of the factorial design. Since 12_controls.py they
    # are in registry.csv and would otherwise show up as a cell of their own,
    # flow into the tier sums and distort the "NEW" share.
    ctrl = [r for r in reg if registry.is_control(r)]
    if ctrl and not a.with_controls:
        reg = [r for r in reg if not registry.is_control(r)]

    # Cells in the order in which they appear in the registry
    cells, order = {}, []
    for r in reg:
        c = r.get("cell") or "?"
        if c not in cells:
            cells[c] = {"n": 0, "pass": 0, "new": 0, "dup": 0,
                        "fail": {}, "seqs": set()}
            order.append(c)
        d = cells[c]
        d["n"] += 1
        f = flt.get(r["seq_id"])
        if not f:
            continue
        if f.get("dup_of"):
            d["dup"] += 1
        if truthy(f.get("pass_comp")):
            d["pass"] += 1
            if not f.get("dup_of"):
                d["new"] += 1
                d["seqs"].add((r.get("binder_seq") or "").strip().upper())
        else:
            for k in (f.get("comp_fail") or f.get("fail") or "").split(","):
                k = k.strip()
                if k:
                    d["fail"][k] = d["fail"].get(k, 0) + 1

    print(f"\n{a.loop}: factorial design after the composition filter\n")
    if ctrl and not a.with_controls:
        n_ok = sum(1 for r in ctrl
                   if truthy((flt.get(r["seq_id"]) or {}).get("pass_comp")))
        print(f"  ({len(ctrl)} control rows not in the factorial design; "
              f"{n_ok} of them pass the composition filter.")
        print("   That the others break it is their purpose - "
              "control_normal_helix exists for exactly that.)\n")
    sel_lv = (cfg.get("select") or {}).get("helicity_levels")
    if isinstance(sel_lv, str):
        sel_lv = [x.strip() for x in sel_lv.split(",") if x.strip()]

    def orderable(c):
        """Can the cell be ordered? select.helicity_levels excludes e.g. `native`.
        Such cells still run along and are reported, but do not make it onto the
        order list - a dead cell there is less bad."""
        if not sel_lv:
            return True
        return (c or "").rsplit("_", 1)[-1] in sel_lv

    hdr = f'  {"Cell":<18}{"total":>8}{"passed":>10}{"NEW":>6}{"dupl.":>7}  O  most frequent failure'
    print(hdr)
    print("  " + "-" * (len(hdr) + 12))

    tot_new = 0
    empty, thin = [], []
    for c in order:
        d = cells[c]
        tot_new += d["new"]
        top = ""
        if d["fail"]:
            k, v = max(d["fail"].items(), key=lambda x: -x[1])
            top = f"{k} ({v})"
        mark = ""
        if d["new"] == 0:
            mark = "   <== DEAD"
            empty.append(c)
        elif d["new"] < 3 and registry.tier_of(c) != "T0":
            # T0 has ONE sequence per helicity level by construction (the
            # WT interface). "thin" would not be a finding there, but the plan.
            mark = "   <== thin"
            thin.append(c)
        print(f'  {c:<18}{d["n"]:>8}{d["pass"]:>10}{d["new"]:>6}{d["dup"]:>7}'
              f'  {"x" if orderable(c) else " "}  {top}{mark}')

    print(f'\n  {"total":<18}{len(reg):>8}{"":>10}{tot_new:>6}')
    if sel_lv:
        print(f'  Column O: orderable according to select.helicity_levels '
              f'({", ".join(sel_lv)}). The others run along.')

    # ---- assessment --------------------------------------------------------
    print()
    if empty:
        print(f"  DEAD: {', '.join(empty)}")
        print("    These cells go into the GPU run with zero sequences. The")
        print("    effect they were meant to isolate can no longer be")
        print("    separated afterwards - that cannot be repaired later.")
    if thin:
        print(f"  THIN (<3): {', '.join(thin)}")
        print("    A cell with one or two sequences gives a single value,")
        print("    not a group value. Usable as a data point,")
        print("    not as evidence for an effect.")
    if not empty and not thin:
        print("  All cells occupied. The factorial design holds.")

    # Tier sums: does a whole binding axis drop out?
    print()
    tiers = {}
    for c in order:
        # registry.tier_of instead of c.split("_")[0]: the latter gives the word
        # 'tier' for 'tier_T1_high' and throws all three tiers into one bucket.
        t = registry.tier_of(c)
        tiers[t] = tiers.get(t, 0) + cells[c]["new"]
    print("  per tier (binding axis):  " +
          "   ".join(f"{t}={n}" for t, n in sorted(tiers.items())))
    lev = {}
    for c in order:
        parts = c.split("_")
        l = parts[-1] if len(parts) > 1 else "?"
        lev[l] = lev.get(l, 0) + cells[c]["new"]
    print("  per level (stability axis): " +
          "   ".join(f"{l}={n}" for l, n in sorted(lev.items())))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
