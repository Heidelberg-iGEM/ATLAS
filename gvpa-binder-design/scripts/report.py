#!/usr/bin/env python
"""Campaign report: funnel, distributions, composition, diversity, snapshots.

Numbers in a CSV do not tell you that a run collapsed onto poly-Glu helices, or
that every survivor came from one backbone, or that the gate is being carried by
a single criterion. These plots are how those failure modes become visible --
all of them pass every confidence metric happily.

Deliberately read-only: it never re-scores or re-filters anything, so the report
cannot disagree with `accepted.csv`.

    python scripts/report.py [--top 12]

Writes <run>/report/{report.md, plots/*.png, snapshots/}.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# Validated categorical slots 1-3 (all-pairs clean, light surface). Slot 3 is
# below 3:1 on the surface, so every chart using it carries direct labels.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, CRIT = "#0ca30c", "#d03b3b"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
AAS = "ACDEFGHIKLMNPQRSTVWY"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": "#d8d7d2", "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "legend.frameon": False, "figure.dpi": 130,
    })


def fnum(v, d=math.nan):
    try:
        f = float(v)
        return f if f == f else d
    except (TypeError, ValueError):
        return d


def read(path):
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as fh:
        return list(csv.DictReader(fh))


def sort_bins(vals):
    """Order length bins numerically. Lexical sort puts '110-150' before
    '33-45', which silently scrambles every per-bin chart."""
    def k(v):
        try:
            return (0, int(str(v).split("-")[0]))
        except ValueError:
            return (1, 0)
    return sorted(vals, key=k)


def col(rows, key):
    return np.array([fnum(r.get(key)) for r in rows], float)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


# --------------------------------------------------------------------------

def plot_funnel(stages, out):
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    names = [s[0] for s in stages][::-1]
    vals = [s[1] for s in stages][::-1]
    ax.barh(names, vals, color=C1, height=0.6)
    for y, v in enumerate(vals):
        ax.text(v + max(vals) * 0.015, y, f"{v:,}", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_title("Campaign funnel")
    ax.set_xlabel("designs")
    ax.grid(axis="y", visible=False)
    return save(fig, out / "funnel.png")


def plot_scores(acc, rej, filters, out):
    """Key gated metrics: where accepted and rejected designs actually sit."""
    specs = [("plddt", "binder pLDDT (in complex)", 0.80, ">="),
             ("i_ptm", "interface pTM", 0.50, ">="),
             ("i_pae", "interface PAE (0-1)", 0.35, "<="),
             ("scrmsd_complex", "scRMSD vs RFD3 (A)", 2.0, "<="),
             ("interface_dG", "Rosetta interface dG", 0.0, "<"),
             ("interface_sc", "shape complementarity", 0.60, ">="),
             ("interface_nres", "interface residues", 7, ">="),
             ("interface_unsat_hbonds", "buried unsat H-bonds", 4, "<="),
             ("binder_alone_plddt", "free-binder pLDDT", 0.80, ">=")]
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.2))
    for ax, (key, label, thr, op) in zip(axes.ravel(), specs):
        a, r = col(acc, key), col(rej, key)
        a, r = a[a == a], r[r == r]
        if len(a) + len(r) == 0:
            ax.set_visible(False); continue
        allv = np.concatenate([a, r])
        lo, hi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
        if hi <= lo:
            lo, hi = allv.min() - 0.5, allv.max() + 0.5
        if key in ("interface_nres", "interface_unsat_hbonds"):
            bins = np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5, 1.0)
        else:
            bins = np.linspace(lo, hi, 34)
        if len(r):
            ax.hist(r, bins=bins, color=C2, alpha=0.75, label="rejected")
        if len(a):
            ax.hist(a, bins=bins, color=C1, alpha=0.85, label="accepted")
        ax.axvline(thr, color=CRIT, lw=1.6, ls="--")
        ax.text(thr, ax.get_ylim()[1] * 0.94, f" {op}{thr:g}", color=CRIT,
                fontsize=8, ha="left" if op.startswith(">") else "right")
        ax.set_title(label); ax.set_ylabel("designs")
    h, l = axes[0, 0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Score distributions, with the BindCraft threshold on each",
                 y=1.05, fontsize=11, fontweight="bold")
    return save(fig, out / "scores.png")


def plot_failures(rej, out):
    c = Counter(f.split(":")[0] for r in rej
                for f in (r.get("failed_criteria") or "").split(";") if f)
    if not c:
        return None
    items = c.most_common(14)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, max(2.6, 0.34 * len(items))))
    ax.barh([k for k, _ in items], [v for _, v in items], color=C2, height=0.62)
    mx = max(v for _, v in items)
    for y, (_, v) in enumerate(items):
        ax.text(v + mx * 0.015, y, str(v), va="center", color=INK, fontsize=8)
    ax.set_xlim(0, mx * 1.14)
    ax.set_title("Which criterion rejects a design")
    ax.set_xlabel("designs failing (a design can fail several)")
    ax.grid(axis="y", visible=False)
    return save(fig, out / "failures.png")


def plot_composition(acc, allr, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6),
                                   gridspec_kw={"width_ratios": [2.3, 1]})
    x = np.arange(len(AAS))
    for off, rows, c, lab in ((-0.2, allr, C2, "all scored"), (0.2, acc, C1, "accepted")):
        if not rows:
            continue
        m = [np.nanmean(col(rows, f"aa_{a}")) if len(rows) else 0 for a in AAS]
        ax1.bar(x + off, m, width=0.38, color=c, label=lab)
    ax1.set_xticks(x); ax1.set_xticklabels(list(AAS))
    ax1.set_title("Binder amino-acid composition")
    ax1.set_ylabel("mean fraction"); ax1.legend(); ax1.grid(axis="x", visible=False)

    props = [("frac_hydrophobic", "hydrophobic"), ("frac_charged", "charged"),
             ("frac_aromatic", "aromatic")]
    y = np.arange(len(props))
    for off, rows, c, lab in ((-0.19, allr, C2, "all scored"), (0.19, acc, C1, "accepted")):
        if not rows:
            continue
        v = [np.nanmean(col(rows, k)) for k, _ in props]
        ax2.barh(y + off, v, height=0.36, color=c, label=lab)
        for yy, vv in zip(y + off, v):
            if vv == vv:
                ax2.text(vv + 0.01, yy, f"{vv:.2f}", va="center", fontsize=7.5, color=INK)
    ax2.set_yticks(y); ax2.set_yticklabels([n for _, n in props])
    ax2.set_xlim(0, 0.85); ax2.set_title("Bulk properties")
    ax2.grid(axis="y", visible=False)
    return save(fig, out / "composition.png")


def plot_structure(acc, out):
    """Secondary structure + length. Stacked bars carry direct labels (slot 3
    is below 3:1 on this surface, so color alone may not carry the segment)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    groups = sort_bins({r.get("length_bin", "?") for r in acc})
    hel = [np.nanmean(col([r for r in acc if r.get("length_bin") == g], "binder_helix_pct")) for g in groups]
    sht = [np.nanmean(col([r for r in acc if r.get("length_bin") == g], "binder_sheet_pct")) for g in groups]
    lop = [np.nanmean(col([r for r in acc if r.get("length_bin") == g], "binder_loop_pct")) for g in groups]
    y = np.arange(len(groups))
    left = np.zeros(len(groups))
    for vals, c, lab in ((hel, C1, "helix"), (sht, C2, "sheet"), (lop, C3, "loop")):
        v = np.nan_to_num(np.array(vals, float))
        ax1.barh(y, v, left=left, height=0.6, color=c, label=lab, edgecolor=SURFACE, linewidth=2)
        for yy, (l0, vv) in enumerate(zip(left, v)):
            if vv > 7:
                ax1.text(l0 + vv / 2, yy, f"{vv:.0f}", ha="center", va="center",
                         color="#ffffff", fontsize=7.5, fontweight="bold")
        left += v
    ax1.set_yticks(y); ax1.set_yticklabels(groups)
    ax1.set_xlabel("% of binder")
    ax1.set_title("Secondary structure of accepted binders", pad=22)
    ax1.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.005))
    ax1.grid(axis="y", visible=False)
    ax1.set_xlim(0, 100)

    lens = col(acc, "binder_len"); lens = lens[lens == lens]
    if len(lens):
        ax2.hist(lens, bins=np.arange(30, 156, 6), color=C1)
    ax2.set_title("Accepted binder length"); ax2.set_xlabel("residues"); ax2.set_ylabel("designs")
    ax2.grid(axis="x", visible=False)
    return save(fig, out / "structure.png")


def plot_diversity(short, acc, out):
    if not short:
        return None
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.3))
    sizes = Counter(r.get("cluster") for r in short)
    hist = Counter(sizes.values())
    ks = sorted(hist)
    ax1.bar([str(k) for k in ks], [hist[k] for k in ks], color=C1, width=0.6)
    for i, k in enumerate(ks):
        ax1.text(i, hist[k], str(hist[k]), ha="center", va="bottom", fontsize=8, color=INK)
    ax1.set_title(f"Cluster sizes ({len(sizes)} clusters)")
    ax1.set_xlabel("designs in cluster"); ax1.set_ylabel("clusters")
    ax1.grid(axis="x", visible=False)

    for ax, key, title in ((ax2, "arm", "Shortlist by arm"),
                           (ax3, "length_bin", "Shortlist by length bin")):
        c = Counter(r.get(key, "?") for r in short)
        ks2 = sort_bins(c) if key == "length_bin" else sorted(c)
        ax.bar(ks2, [c[k] for k in ks2], color=C1, width=0.6)
        for i, k in enumerate(ks2):
            ax.text(i, c[k], str(c[k]), ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_title(title); ax.set_ylabel("designs")
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", rotation=30)
    return save(fig, out / "diversity.png")


# --------------------------------------------------------------------------

def snapshots(short, cfg, out, top):
    """Copy the best predicted complexes and write a PyMOL session script."""
    snap = out / "snapshots"
    snap.mkdir(parents=True, exist_ok=True)
    picked = []
    for r in short[:top]:
        src = r.get("af2_complex_pdb") or ""
        if not src or not Path(src).exists():
            continue
        dst = snap / f"rank{int(float(r.get('rank', 0))):02d}_{r['id']}.pdb"
        shutil.copy(src, dst)
        picked.append((dst.name, r))
    if not picked:
        return snap, []

    # Chain labels differ between RFD3 output and AF2 output, and these
    # snapshots are AF2 output: chains are <target subunits in the order passed
    # to colabdesign> followed by the binder, each renumbered from 1. Reusing
    # the RFD3 convention here highlighted residues on the wrong subunit and
    # pointed "binder" at a target chain. Derive the layout from the file.
    ref = common.read_structure(snap / picked[0][0])
    order = [str(c) for c in dict.fromkeys(ref.chain)]
    blen = int(float(picked[0][1].get("binder_len") or 0))
    bcands = [c for c in order if len(ref.ca(c)) == blen]
    binder_ch = bcands[-1] if bcands else order[-1]
    tgt_order = [c for c in order if c != binder_ch]
    # input design chain -> AF2 chain, positionally
    in2af2 = dict(zip(tgt_order and cfg["target"]["design_chains"], tgt_order))
    offs = common.chain_offsets(cfg)

    sel_parts = []
    for h in sorted({h for arm in cfg["arms"].values() for h in arm["hotspots"]}):
        ic, rs = h[0], int(h[1:])
        if ic not in in2af2:
            continue
        sel_parts.append(f"(chain {in2af2[ic]} and resi {rs - offs[ic] + 1})")
    sel = " or ".join(sel_parts)

    lines = [
        "# Top designs as AF2 predicted them. Every model is already in the",
        "# target's frame, so nothing is aligned here -- what you see is where",
        "# AF2 actually placed each binder on the same GvpA rib.",
        "#   pymol view.pml      (pymol lives in the mdplot env)",
        "reinitialize", "bg_color white", "set ray_opaque_background, 0",
    ]
    for name, _ in picked:
        lines.append(f"load {name}, {name[:-4]}")
    lines += [
        "hide everything", "show cartoon",
        f"select target_chains, chain {'+'.join(tgt_order)}",
        "color grey80, target_chains",
        f"select binder, chain {binder_ch}",
        "spectrum count, rainbow, binder and name CA",
    ]
    if sel:
        lines += [f"select hotspots, ({sel})", "show sticks, hotspots",
                  "color salmon, hotspots", "orient hotspots", "zoom hotspots, 12"]
    else:
        lines += ["orient binder", "zoom binder, 10"]
    if len(picked) > 1:
        lines.append("set grid_mode, 1")
    lines.append("deselect")
    (snap / "view.pml").write_text("\n".join(lines) + "\n")

    (snap / "README.txt").write_text(
        "Top designs, as AF2 predicted them (model 0 of the full pass).\n\n"
        "Chains in these AF2 models: the GvpA subunits come FIRST (in the order\n"
        "input A, B, E) and the BINDER IS LAST. Target residues are renumbered\n"
        "from 1, so input hotspot B55 is chain B residue 44 here. Note this is\n"
        "NOT the RFD3 layout, where the binder is chain A.\n\n"
        "  pymol view.pml     # grid view, hotspots as salmon sticks\n\n"
        "All models share the target's coordinate frame, so no alignment is\n"
        "applied -- differences you see in binder placement are real.\n")
    return snap, picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    style()
    cfg = common.load_config(a.config)
    root = Path(cfg["run_root"]) / cfg["campaign"]
    out = root / "report"
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)

    backbones = read(root / "backbones.csv")
    seqs = []
    for f in (sorted(glob.glob(str(root / "sequences.shard*.csv")))
              or [str(root / "sequences.csv")]):
        seqs += read(f)
    gate = read(root / "af2_gate" / "scores.csv")
    merged = read(root / "merged.csv")
    acc = [r for r in merged if r.get("accepted") == "1"]
    rej = [r for r in merged if r.get("accepted") != "1"]
    short = read(root / "shortlist.csv")
    short.sort(key=lambda r: fnum(r.get("rank"), 1e9))

    stages = [("RFD3 backbones", len(backbones)),
              ("passed triage", sum(1 for r in backbones if r.get("pass") == "1")),
              ("MPNN sequences", len(seqs))]
    if gate:
        stages.append(("AF2 gate scored", len(gate)))
    if merged:
        stages += [("fully scored", len(merged)), ("accepted", len(acc))]
    if short:
        stages.append(("shortlisted", len(short)))
    stages = [s for s in stages if s[1] > 0]

    made = []
    if stages:
        made.append(plot_funnel(stages, plots))
    if merged:
        made.append(plot_scores(acc, rej, None, plots))
        f = plot_failures(rej, plots)
        made.append(f) if f else None
        made.append(plot_composition(acc, merged, plots))
    if acc:
        made.append(plot_structure(acc, plots))
    d = plot_diversity(short, acc, plots) if short else None
    made.append(d) if d else None

    snap, picked = snapshots(short, cfg, out, a.top) if short else (out / "snapshots", [])

    # ---------------------------------------------------------------- report
    L = [f"# {cfg['campaign']} — campaign report", "",
         f"Run: `{root}`", "", "## Funnel", "",
         "| stage | designs | of previous |", "|---|---:|---:|"]
    prev = None
    for name, n in stages:
        L.append(f"| {name} | {n:,} | {'—' if prev in (None, 0) else f'{n/prev:.1%}'} |")
        prev = n
    L += ["", "![funnel](plots/funnel.png)", ""]

    if merged:
        L += ["## Acceptance", "",
              f"**{len(acc):,} of {len(merged):,} fully-scored designs pass "
              f"({len(acc)/max(len(merged),1):.1%}).**", ""]
        for key, label in (("arm", "arm"), ("tier", "tier"), ("length_bin", "length bin")):
            groups = (sort_bins({r.get(key, "?") for r in merged}) if key == "length_bin"
                      else sorted({r.get(key, "?") for r in merged}))
            if len(groups) < 2:
                continue
            L += [f"| {label} | scored | accepted | rate |", "|---|---:|---:|---:|"]
            for g in groups:
                sub = [r for r in merged if r.get(key) == g]
                na = sum(1 for r in sub if r.get("accepted") == "1")
                L.append(f"| {g} | {len(sub):,} | {na:,} | {na/max(len(sub),1):.1%} |")
            L.append("")
        L += ["![scores](plots/scores.png)", ""]
        if (plots / "failures.png").exists():
            L += ["### What is doing the rejecting", "",
                  "![failures](plots/failures.png)", ""]

    if acc:
        def stat(key):
            v = col(acc, key); v = v[v == v]
            return f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) else "—"
        L += ["## Composition and structure", "",
              "| property | accepted mean |", "|---|---|",
              f"| hydrophobic fraction | {stat('frac_hydrophobic')} |",
              f"| charged fraction | {stat('frac_charged')} |",
              f"| aromatic fraction | {stat('frac_aromatic')} |",
              f"| GRAVY hydropathy | {stat('hydropathy_gravy')} |",
              f"| net charge | {stat('net_charge')} |",
              f"| helix % | {stat('binder_helix_pct')} |",
              f"| sheet % | {stat('binder_sheet_pct')} |",
              f"| loop % | {stat('binder_loop_pct')} |", "",
              "![composition](plots/composition.png)", "",
              "![structure](plots/structure.png)", ""]

    if short:
        L += ["## Diversity", "",
              f"{len(short)} shortlisted across "
              f"{len(set(r.get('cluster') for r in short))} geometric clusters, "
              f"from {len(set(r.get('design') for r in short))} distinct backbones.", "",
              "![diversity](plots/diversity.png)", "",
              "## Top designs", "",
              "| rank | id | arm | tier | len | composite | i_ptm | i_pae | dG | SC |",
              "|---:|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for r in short[:a.top]:
            g = lambda k: (f"{fnum(r.get(k)):.3g}" if fnum(r.get(k)) == fnum(r.get(k)) else "—")
            L.append(f"| {r.get('rank','')} | `{r['id']}` | {r.get('arm','')} | "
                     f"{r.get('tier','')} | {r.get('binder_len','')} | {g('composite')} | "
                     f"{g('i_ptm')} | {g('i_pae')} | {g('interface_dG')} | {g('interface_sc')} |")
        L += ["", f"Structures for the top {len(picked)}: `report/snapshots/` "
                  f"(`pymol view.pml`).", ""]

    (out / "report.md").write_text("\n".join(L) + "\n")
    print(f"report   : {out/'report.md'}")
    print(f"plots    : {len([m for m in made if m])} -> {plots}")
    print(f"snapshots: {len(picked)} structures -> {snap}")


if __name__ == "__main__":
    main()
