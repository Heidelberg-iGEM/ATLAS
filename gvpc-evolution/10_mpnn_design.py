#!/usr/bin/env python3
"""
Stage 1: generate sequences (LigandMPNN).

    $MPNN_PY 10_mpnn_design.py --config config.yaml --tiers T1,T2,T3 --n-per-cell 45

Only the positions listed in zones.tiers are designed. Everything else stays
native, so the back side of the binder cannot be rebuilt along with the interface.

Result: 04_loops/<loop>/registry.csv, one row per sequence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_registry as registry
import lib_checks as checks

# Keys known to the MPNN engine. It silently discards unknown keys,
# so a typo would run quietly with the default.
ENGINE_KEYS = {"model_type", "is_legacy_weights", "out_directory",
               "write_structures", "write_fasta", "checkpoint_path", "device"}
INPUT_KEYS = {"batch_size", "remove_waters", "name", "designed_chains",
              "designed_residues", "fixed_residues", "fixed_chains",
              "atomize_side_chains", "temperature", "bias", "bias_per_residue",
              "temperature_per_residue", "seed", "symmetry", "omit_AA",
              "omit_AA_per_residue"}


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
    ap.add_argument("--tiers", default="T1,T2,T3")
    ap.add_argument("--n-per-cell", type=int, default=45)
    ap.add_argument("--batch-size", type=int, default=15,
                    help="Sequences per MPNN call; n-per-cell is split into such "
                         "batches with different seeds")
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="0.1 is practically argmax and gives hardly any diversity")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--model-type", default="ligand_mpnn",
                    choices=["ligand_mpnn", "protein_mpnn", "solublempnn"])
    ap.add_argument("--dry-run", action="store_true",
                    help="only plan and check, do not load a model")
    a = ap.parse_args()

    cfg = registry.load_config(a.config)
    loop = a.loop or cfg["project"]["loop"]
    root = Path(cfg["project"]["root"])
    tpl = cfg["template"]
    binder, brange, native = tpl["binder_chain"], tpl["binder_range"], tpl["native_binder"]
    tiers_cfg = cfg["zones"]["tiers"]

    tiers = [t.strip() for t in a.tiers.split(",") if t.strip()]
    unknown = [t for t in tiers if t not in tiers_cfg]
    if unknown:
        sys.exit(f"Unknown tiers {unknown}; config.yaml contains {sorted(tiers_cfg)}")

    cif = registry.resolve(cfg, tpl["cif"])
    if not cif.exists():
        sys.exit(f"Template missing: {cif}")

    # Check the template before loading anything. This is purely textual
    # and needs no env.
    checks.assert_cif_columns(cif, binder=binder, targets=tpl["target_chains"],
                              binder_range=brange)

    out_mpnn = root / "03_output" / "MPNN_output" / loop
    loop_dir = root / "04_loops" / loop

    engine_config = {
        "model_type": a.model_type,
        "is_legacy_weights": True,
        "out_directory": str(out_mpnn),
        "write_structures": False,
        "write_fasta": True,
    }
    checks.assert_mpnn_keys(engine_config, ENGINE_KEYS, "engine_config")

    # One entry per batch. Each batch gets its own seed, otherwise
    # they all produce the same output.
    jobs, seed = [], a.seed0
    for tier in tiers:
        positions = sorted(tiers_cfg[tier])
        out_of_range = [p for p in positions if not (brange[0] <= p <= brange[1])]
        if out_of_range:
            sys.exit(f"{tier}: positions {out_of_range} lie outside "
                     f"{binder} {brange[0]}-{brange[1]}.")
        designed = [f"{binder}{p}" for p in positions]

        left = a.n_per_cell
        while left > 0:
            n = min(a.batch_size, left)
            # designed_residues instead of designed_chains: the two are mutually
            # exclusive in MPNN, and only the residue-based variant can release
            # individual positions.
            ic = {"batch_size": n, "remove_waters": True,
                  "name": f"{loop}_{tier}_s{seed}",
                  "designed_residues": designed, "atomize_side_chains": True,
                  "temperature": a.temperature, "seed": seed}
            checks.assert_mpnn_keys(ic, INPUT_KEYS, f"input_config[{tier}/{seed}]")
            checks.assert_design_constraints(ic)
            jobs.append({"cfg": ic, "tier": tier, "seed": seed, "name": ic["name"]})
            left -= n
            seed += 1

    print(f"\n{loop}: {len(tiers)} Tiers x {a.n_per_cell} = "
          f"{len(tiers) * a.n_per_cell} sequences in {len(jobs)} batches")
    for tier in tiers:
        print(f"  {tier}: {len(tiers_cfg[tier])} positions "
              f"{[f'{binder}{p}' for p in sorted(tiers_cfg[tier])]}")
    print(f"  T={a.temperature}  model={a.model_type}  seeds {a.seed0}..{seed - 1}")
    if a.dry_run:
        print("\n--dry-run: model not loaded.")
        return 0

    # Load the structure and check it against the config.
    from atomworks.io.utils.io_utils import load_any
    from mpnn.inference_engines.mpnn import MPNNInferenceEngine

    atom_array = load_any(str(cif), model=1)
    checks.assert_structure(atom_array, binder=binder, targets=tpl["target_chains"],
                            binder_range=brange, n_binder_res=len(native))
    checks.assert_binder_sequence(atom_array, native, binder=binder)

    out_mpnn.mkdir(parents=True, exist_ok=True)
    model = MPNNInferenceEngine(**engine_config)
    model.run(input_dicts=[j["cfg"] for j in jobs],
              atom_arrays=[atom_array] * len(jobs))

    # Collect the FASTAs. They contain all chains one after another; the binder
    # is the last len(native) residues. Do not cut at fixed boundaries,
    # that breaks as soon as a chain length changes.
    n_targets = len(tpl["target_chains"])
    expected_len = n_targets * tpl["target_range"][1] + len(native)

    rows, n, seen = [], 0, set()
    for job in jobs:
        files = sorted(out_mpnn.glob(f"{job['name']}*.fa")) + \
                sorted(out_mpnn.glob(f"{job['name']}*.fasta"))
        if not files:
            print(f"  ! no FASTA for {job['name']}", file=sys.stderr)
            continue
        for name, seq in read_fasta(files[0]):
            full = seq.strip().upper()
            if len(full) != expected_len:
                print(f"  ! {name}: {len(full)} residues, expected {expected_len}",
                      file=sys.stderr)
                continue
            if name in seen:
                continue
            seen.add(name)
            n += 1
            binder_seq = full[-len(native):]
            rows.append({
                "seq_id": registry.make_seq_id(loop, n), "loop": loop,
                "cell": job["tier"], "parent_id": "",
                "design_mode": f"tier_{job['tier']}",
                "mpnn_model": a.model_type, "mpnn_temp": a.temperature,
                "mpnn_seed": job["seed"], "n_designed": len(job["cfg"]["designed_residues"]),
                "binder_seq": binder_seq, "seq_sha8": registry.seq_sha8(binder_seq),
                "mpnn_record": name, "created": registry.now_iso(),
            })

    if not rows:
        sys.exit("No sequences collected. Check the FASTA output in "
                 f"{out_mpnn} .")

    dest = loop_dir / "registry.csv"
    registry.archive(dest, loop_dir / "archive")
    registry.write_csv(dest, rows)
    registry.log_stage(loop_dir, "10_mpnn_design", {
        "n": len(rows), "tiers": tiers, "temperature": a.temperature,
        "model_type": a.model_type, "template": str(cif),
        "seeds": [a.seed0, seed - 1]})

    # Did the tier restriction hold? The exposed positions are in no tier,
    # so they must not be changed anywhere.
    exposed = cfg["zones"]["exposed"]
    unchanged = sum(1 for r in rows
                    if all(r["binder_seq"][p - brange[0]] == native[p - brange[0]]
                           for p in exposed))
    print(f"\n  {len(rows)} sequences -> {dest}")
    print(f"  Back side unchanged in {unchanged}/{len(rows)}")
    if unchanged != len(rows):
        print("  ! MPNN changed positions outside the tier list.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
