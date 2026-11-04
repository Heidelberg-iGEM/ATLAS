#!/usr/bin/env python3
"""
Stage 4: composition filter and dedup.

    python 15_filter_composition.py --config config.yaml

Five criteria on the amino acid composition, position-dependent over the
zones from config.yaml. Runs before the structure predictions, because those
do not penalize poly-alanine: a helix of almost only alanine gets good
interface scores and would pass every downstream gate.

The thresholds are calibrated on the native binder. If it does not pass them,
something is wrong with the zones or the thresholds, and the stage aborts.

The MEASURED VALUES are written, not just the verdict. A threshold whose
measured value is not in the table cannot be readjusted later.

Controls must not drop out here - control_normal_helix breaks the filter by
construction. pass_comp is still computed for them, because that is exactly
the measured value. Which rows go into the GPU run is decided by
lib_registry.to_predict().

Result: 04_loops/<loop>/scores_filter.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_registry as registry

# Large hydrophobics. Alanine deliberately does not count: poly-alanine and
# poor packing are two different findings and need two rows.
BIGHYD = set("VLIMFWY")
CHARGED = set("DEKR")


def longest_run(seq: str, aa: str = "A") -> int:
    best = cur = 0
    for c in seq:
        cur = cur + 1 if c == aa else 0
        best = max(best, cur)
    return best


def measure(seq: str, offset: int, zones: dict) -> dict:
    """The five measured values plus two that are only reported."""
    exposed = [p - offset for p in zones["exposed"]]
    iface = [p - offset for p in zones["interface"]]
    for name, idx in (("exposed", exposed), ("interface", iface)):
        bad = [i for i in idx if not (0 <= i < len(seq))]
        if bad:
            raise ValueError(
                f"zones.{name} contains positions outside the sequence "
                f"(length {len(seq)}, offset {offset}): {bad}. Either "
                "binder_range is wrong or the sequence has the wrong length.")

    exp_aa = [seq[i] for i in exposed]
    ifa_aa = [seq[i] for i in iface]
    return {
        "comp_ala_exposed": round(sum(a == "A" for a in exp_aa) / len(exp_aa), 4),
        "comp_bighyd_exposed": sum(a in BIGHYD for a in exp_aa),
        "comp_ala_block": longest_run(seq, "A"),
        "comp_charged_total": round(sum(a in CHARGED for a in seq) / len(seq), 4),
        "comp_bighyd_iface": sum(a in BIGHYD for a in ifa_aa),
        "comp_ala_total": round(sum(a == "A" for a in seq) / len(seq), 4),
        "comp_net_charge": sum(a in "KR" for a in seq) - sum(a in "DE" for a in seq),
    }


def judge(m: dict, pf: dict) -> tuple[bool, str]:
    """Each criterion individually as an exclusion, no weighted formula."""
    fails = []
    if m["comp_ala_exposed"] > pf["max_ala_exposed"]:
        fails.append(f"ala_exposed={m['comp_ala_exposed']:.2f}>{pf['max_ala_exposed']}")
    if m["comp_bighyd_exposed"] > pf["max_bighyd_exposed"]:
        fails.append(f"bighyd_exposed={m['comp_bighyd_exposed']}>{pf['max_bighyd_exposed']}")
    if m["comp_ala_block"] > pf["max_ala_block"]:
        fails.append(f"ala_block={m['comp_ala_block']}>{pf['max_ala_block']}")
    if m["comp_charged_total"] < pf["min_charged_total"]:
        fails.append(f"charged={m['comp_charged_total']:.2f}<{pf['min_charged_total']}")
    if m["comp_bighyd_iface"] < pf["min_bighyd_iface"]:
        fails.append(f"bighyd_iface={m['comp_bighyd_iface']}<{pf['min_bighyd_iface']}")
    return (not fails), ",".join(fails)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--loop", default=None)
    ap.add_argument("--no-dedup", action="store_true")
    a = ap.parse_args()

    cfg = registry.load_config(a.config)
    loop = a.loop or cfg["project"]["loop"]
    root = Path(cfg["project"]["root"])
    loop_dir = root / "04_loops" / loop
    zones, pf = cfg["zones"], cfg["prefilter"]
    offset = cfg["template"]["binder_range"][0]
    native = cfg["template"]["native_binder"]

    # Calibration on the native binder. Always runs: if the wild type does not pass
    # its own filter, every verdict below it is worthless.
    nm = measure(native, offset, zones)
    ok, why = judge(nm, pf)
    print("Calibration on the native binder")
    print(f"  {native}")
    for name, val, thr in (
            ("ala_exposed", f"{nm['comp_ala_exposed']:.2f}", f"<= {pf['max_ala_exposed']}"),
            ("bighyd_exposed", nm["comp_bighyd_exposed"], f"<= {pf['max_bighyd_exposed']}"),
            ("ala_block", nm["comp_ala_block"], f"<= {pf['max_ala_block']}"),
            ("charged_total", f"{nm['comp_charged_total']:.2f}", f">= {pf['min_charged_total']}"),
            ("bighyd_iface", nm["comp_bighyd_iface"], f">= {pf['min_bighyd_iface']}")):
        print(f"    {name:<16} {str(val):>6}   threshold {thr}")
    if not ok:
        print(f"\n  The native binder fails its own filter: {why}")
        print("  Either the zones or the thresholds are wrong.")
        return 2
    print("  Passes all five criteria.\n")

    entries = registry.read_csv(loop_dir / "registry.csv")
    if not entries:
        sys.exit(f"No registry.csv in {loop_dir}. Run stage 1 first.")
    print(f"{loop}: {len(entries)} sequences")

    # Dedup against all earlier rounds. An exact repeat must not cost any
    # compute time.
    dups = {}
    if not a.no_dedup:
        previous = [p for p in sorted((root / "04_loops").glob("*/registry.csv"))
                    if p.parent.name != loop]
        dups = registry.find_duplicates(entries, previous)
        if dups:
            print(f"Dedup: {len(dups)} exact repeats from earlier rounds")

    results, n_pass, fail_counter = [], 0, {}
    for e in entries:
        seq = e["binder_seq"].strip().upper()
        m = measure(seq, offset, zones)
        passed, why = judge(m, pf)
        n_pass += passed
        for f in filter(None, why.split(",")):
            key = f.split("=")[0]
            fail_counter[key] = fail_counter.get(key, 0) + 1
        results.append({"seq_id": e["seq_id"], **m, "pass_comp": passed,
                        "comp_fail": why, "dup_of": dups.get(e["seq_id"], "")})

    n_gpu = sum(1 for r in results if r["pass_comp"] and not r["dup_of"])
    print(f"\n  passed   {n_pass}/{len(results)}")
    print(f"  of which new   {n_gpu}   (this is what the GPU sees, plus the controls)")
    if fail_counter:
        print("\n  Failures by criterion:")
        for k, v in sorted(fail_counter.items(), key=lambda x: -x[1]):
            print(f"    {k:<16} {v}")

    out_path = loop_dir / "scores_filter.csv"
    registry.write_csv(out_path, results)
    registry.log_stage(loop_dir, "15_filter_composition",
                       {"n_in": len(results), "n_pass": n_pass, "n_to_gpu": n_gpu,
                        "thresholds": pf})
    print(f"\n  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
