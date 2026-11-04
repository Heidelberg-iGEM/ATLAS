#!/usr/bin/env python
"""Turn the shortlist into an orderable table: protein + codon-optimized ORF.

Untagged, as requested -- start codon, the binder ORF, stop codon, nothing else.
No purification tag, no linker, no cloning overhangs. Add those to the ordered
fragment yourself; keeping them out means the same synthesised fragment can go
into whatever vector you settle on.

Codon choice targets E. coli expression:
  * only common codons (the rare ones -- ATA, AGA/AGG/CGA, CTA, CCC -- are
    dropped outright, since a rare codon in a small ORF is a real expression
    risk rather than a rounding error),
  * global GC held inside the configured window and no 40 bp window outside it,
    because synthesis vendors reject on local GC as well as global,
  * no homopolymer run longer than the configured limit,
  * no internal instance of the listed restriction sites, so standard cloning
    stays available.

The optimiser samples a weighted codon assignment, then repairs violations by
synonymous substitution, restarting if a sequence cannot be repaired. Every
returned ORF is verified to back-translate to the exact input protein.

    python scripts/make_order_list.py [--shortlist ...] [--out <run>/order_list.csv]
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# Common E. coli codons with relative synonymous usage. Rare codons removed.
CODONS = {
    "A": {"GCG": .38, "GCC": .29, "GCA": .22, "GCT": .17},
    "R": {"CGT": .49, "CGC": .51},
    "N": {"AAC": .55, "AAT": .45},
    "D": {"GAT": .63, "GAC": .37},
    "C": {"TGC": .55, "TGT": .45},
    "Q": {"CAG": .65, "CAA": .35},
    "E": {"GAA": .68, "GAG": .32},
    "G": {"GGC": .44, "GGT": .38, "GGG": .18},
    "H": {"CAT": .57, "CAC": .43},
    "I": {"ATT": .55, "ATC": .45},
    "L": {"CTG": .60, "TTG": .16, "CTT": .12, "CTC": .12},
    "K": {"AAA": .76, "AAG": .24},
    "M": {"ATG": 1.0},
    "F": {"TTT": .57, "TTC": .43},
    "P": {"CCG": .60, "CCA": .22, "CCT": .18},
    "S": {"AGC": .33, "TCT": .18, "TCC": .18, "TCG": .16, "AGT": .15},
    "T": {"ACC": .45, "ACG": .28, "ACT": .17, "ACA": .10},
    "W": {"TGG": 1.0},
    "Y": {"TAT": .57, "TAC": .43},
    "V": {"GTG": .38, "GTT": .27, "GTC": .22, "GTA": .13},
}
TABLE = {c: aa for aa, d in CODONS.items() for c in d}
STOP = "TAA"


def translate(dna):
    return "".join(TABLE.get(dna[i:i + 3], "*") for i in range(0, len(dna) - 2, 3))


def gc(s):
    return (s.count("G") + s.count("C")) / len(s) if s else 0.0


def violations(dna, cfg_order):
    """Indices of offending CODONS, so repair can target them."""
    o = cfg_order
    bad = set()
    n = len(dna) // 3

    if not (o["gc_min"] <= gc(dna) <= o["gc_max"]):
        bad.update(range(n))                       # global GC: any codon may help
    w = 40
    for i in ([] if o.get("_skip_window") else range(0, max(1, len(dna) - w + 1), 10)):
        win = dna[i:i + w]
        if len(win) == w and not (o["gc_min"] <= gc(win) <= o["gc_max"]):
            bad.update(range(i // 3, min(n, (i + w) // 3 + 1)))

    for m in re.finditer(r"(A{%d,}|C{%d,}|G{%d,}|T{%d,})" % ((o["max_homopolymer"] + 1,) * 4), dna):
        bad.update(range(m.start() // 3, min(n, m.end() // 3 + 1)))

    for site in (o.get("avoid_sites") or {}).values():
        for m in re.finditer(f"(?={site})", dna):
            bad.update(range(m.start() // 3, min(n, (m.start() + len(site)) // 3 + 1)))
    return bad


def optimize(protein, o, rng, restarts=60, iters=4000):
    for _ in range(restarts):
        codons = [rng.choices(list(CODONS[aa]), weights=list(CODONS[aa].values()))[0]
                  if aa in CODONS else None for aa in protein]
        if any(c is None for c in codons):
            raise ValueError(f"non-standard residue in {protein}")
        for _ in range(iters):
            dna = "".join(codons)
            bad = violations(dna, o)
            if not bad:
                assert translate(dna) == protein
                return dna
            i = rng.choice(sorted(bad))
            alts = [c for c in CODONS[protein[i]] if c != codons[i]]
            if alts:
                codons[i] = rng.choice(alts)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--shortlist", default=None)
    ap.add_argument("--all", action="store_true",
                    help="source accepted.csv instead of shortlist.csv: EVERY design "
                         "passing the BindCraft filters, ranked by composite. The "
                         "shortlist is a diversity-clustered SUBSET of accepted and is "
                         "capped by diversity.target_list_size, so on a high-yield "
                         "campaign it shows only a fraction of the passers.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    cfg = common.load_config(a.config)
    root = Path(cfg["run_root"]) / cfg["campaign"]
    o = cfg["order"]
    rng = random.Random(a.seed)

    sl = Path(a.shortlist or (root / ("accepted.csv" if a.all else "shortlist.csv")))
    # Belt and braces: an order list must never be older than the gate that
    # produced it.
    acc = root / "accepted.csv"
    if sl.exists() and acc.exists() and sl.stat().st_mtime < acc.stat().st_mtime - 1:
        raise SystemExit(
            f"{sl.name} is older than accepted.csv -- it is from a previous run. "
            f"Re-run scripts/diversity.py, or delete {sl}.")
    with open(sl) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"empty shortlist: {sl}")

    if a.all:
        # accepted.csv carries no rank/cluster -- those are added by diversity.py.
        # Rank by composite so the full list is still ordered best-first.
        def _c(r):
            try:
                return -float(r.get("composite") or 0)
            except (TypeError, ValueError):
                return 0.0
        rows.sort(key=_c)
        for i, r in enumerate(rows, 1):
            r.setdefault("cluster", "")
            r["rank"] = i

    out_rows, failed = [], 0
    for r in rows:
        prot = r["seq"].strip().upper()
        orf_prot = ("M" + prot if o.get("add_start_met", True) and not prot.startswith("M")
                    else prot)
        note = ""
        dna = optimize(orf_prot, o, rng)
        if dna is None:
            # Some compositions cannot meet the local-GC window at all -- a
            # poly-Lys stretch tops out at 33% GC however it is coded. Fall
            # back to global GC only and FLAG the row rather than dropping a
            # design or silently shipping an ORF that breaks a stated rule.
            relaxed = dict(o); relaxed["_skip_window"] = True
            dna = optimize(orf_prot, relaxed, rng)
            if dna is None:
                failed += 1
                dna, note = "", "could not optimise"
            else:
                note = "relaxed: local GC window not satisfiable for this composition"
        if dna and o.get("add_stop", True):
            dna += STOP
        out_rows.append({
            "rank": r.get("rank", ""), "id": r["id"], "design": r["design"],
            "arm": r.get("arm", ""), "tier": r.get("tier", ""),
            "cluster": r.get("cluster", ""), "length_bin": r.get("length_bin", ""),
            "binder_len": len(orf_prot), "composite": r.get("composite", ""),
            "protein_seq": orf_prot,
            "dna_orf": dna, "dna_len": len(dna), "dna_gc": round(gc(dna), 3) if dna else "",
            "dna_note": note,
            "i_ptm": r.get("i_ptm", ""), "i_pae": r.get("i_pae", ""),
            "plddt": r.get("plddt", ""), "interface_dG": r.get("interface_dG", ""),
            "interface_sc": r.get("interface_sc", ""),
            "interface_nres": r.get("interface_nres", ""),
            "scrmsd_complex": r.get("scrmsd_complex", ""),
            # Buildability, not affinity: close + outward-facing termini are what
            # a fusion, graft or multivalent display needs. Reported only.
            "termini_dist": r.get("termini_dist", ""),
            "termini_same_side": r.get("termini_same_side", ""),
            "termini_both_away": r.get("termini_both_away", ""),
            "termini_n_to_target": r.get("termini_n_to_target", ""),
            "termini_c_to_target": r.get("termini_c_to_target", ""),
            "binder_alone_plddt": r.get("binder_alone_plddt", ""),
            "binder_rmsd": r.get("binder_rmsd", ""),
            # Reported ALONGSIDE the AF2 numbers, never gating. On this target
            # AF2 i_pae and boltz_complex_plddt correlate at r=+0.10 while both
            # separate real designs from scrambles -- independent evidence, so
            # a design strong in both is better supported than one strong in
            # either alone.
            "boltz_iptm": r.get("boltz_iptm", ""),
            "boltz_confidence": r.get("boltz_confidence", ""),
            "boltz_complex_plddt": r.get("boltz_complex_plddt", ""),
            "boltz_ipde": r.get("boltz_ipde", ""),
            "interface_residues": r.get("interface_residues", ""),
        })

    out = Path(a.out or root / "order_list.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)

    fa = out.with_suffix(".fasta")
    with open(fa, "w") as fh:
        for r in out_rows:
            fh.write(f">{r['id']} rank={r['rank']} tier={r['tier']} arm={r['arm']}\n"
                     f"{r['protein_seq']}\n")

    ok = [r for r in out_rows if r["dna_orf"]]
    print(f"order list : {len(out_rows)} designs")
    print(f"  ORFs ok  : {len(ok)}" + (f"  ({failed} could not be optimised)" if failed else ""))
    if ok:
        print(f"  GC range : {min(r['dna_gc'] for r in ok):.3f}-{max(r['dna_gc'] for r in ok):.3f}")
        print(f"  ORF len  : {min(r['dna_len'] for r in ok)}-{max(r['dna_len'] for r in ok)} bp")
    print(f"\nwrote {out}\n      {fa}")


if __name__ == "__main__":
    main()
