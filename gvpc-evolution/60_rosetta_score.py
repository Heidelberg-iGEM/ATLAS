#!/usr/bin/env python3
"""
Stage 11: interface physics with PyRosetta.

    $ROS_PY 60_rosetta_score.py --config config.yaml --models-dir PREDICTIONS \\
        --rows order --jobs 16

FastRelax on the Boltz models, then a physical energy function at the
interface. The relax is mandatory: raw predictions sometimes have positive
total scores, without relax one measures clashes instead of interfaces. With
backbone constraints, otherwise the pose drifts away.

This stage decides nothing. No ros_ column is among the gate criteria
of stage 10, the run does not change gate.csv, selected.csv or order.csv. It
is an outlier check on a selection that has already been made.

--rows chooses which rows are scored:
    pass      all gate passers. Expensive: with 100 passers times 5 seeds times
              around 5 minutes that is about 2.5 hours on 16 cores.
    selected  only selected.csv - the designs WITHOUT controls. That leaves out
              the comparison group, and a single interface energy is then
              not interpretable.
    order     order.csv: designs, controls and the deliberate failures.
              This is the set that is worth it.
    all       every row from gate.csv.

selected.csv and order.csv only come into being with the live run of stage 10.

Averaging is over all seeds, with standard deviation. The same
sequence difference turns out about as large between two structures of the same
sequence as between different sequences; scoring a single model
mostly measures noise. If the standard deviation is of the order of
the differences between variants, the stage contributes nothing.

dG_separated is NOT a binding affinity, but Rosetta units. That
belongs in every table that is published. Rosetta's dSASA also counts
both sides of the interface and is therefore about twice as large as
the bsa from stage 7 - never in the same column.

One JSON file per structure, joined at the end. The run can therefore be aborted
and resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_registry as registry

COLS = ("delta_unsat", "packstat", "dG_separated", "sc", "dsasa", "nres",
        "score_relaxed", "score_raw")

_TARGETS: list[str] = []
_BINDER = "F"
_RELAX = True


def _init(targets, binder, relax):
    global _TARGETS, _BINDER, _RELAX
    _TARGETS, _BINDER, _RELAX = targets, binder, relax
    from pyrosetta import init
    init("-mute all -ex1 -ex2aro")


def _chain_indices(pose, wanted):
    """Chain indices for the desired PDB chain names, read from the pose.

    Chain 'A' is not automatically index 1, so look it up instead of deriving it.
    """
    info = pose.pdb_info()
    return sorted({pose.chain(i) for i in range(1, pose.total_residue() + 1)
                   if info.chain(i) in wanted})


def score_one(task):
    """(seq_id, seed, path, out_json) -> dict. Runs in the worker process."""
    seq_id, seed, path, out_json = task
    out_json = Path(out_json)
    if out_json.exists():
        try:
            return json.loads(out_json.read_text())
        except Exception:
            out_json.unlink(missing_ok=True)

    try:
        import pyrosetta
        from pyrosetta import pose_from_file, get_fa_scorefxn
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

        pose = pose_from_file(str(path))
        sfx = get_fa_scorefxn()
        rec = {"seq_id": seq_id, "seed": seed, "path": str(path),
               "score_raw": round(sfx(pose), 2)}

        chains = {pose.pdb_info().chain(i) for i in range(1, pose.total_residue() + 1)}
        missing = (set(_TARGETS) | {_BINDER}) - chains
        if missing:
            rec["error"] = f"Chains missing: {sorted(missing)} (present {sorted(chains)})"
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(rec))
            return rec

        if _RELAX:
            from pyrosetta.rosetta.protocols.relax import FastRelax
            from pyrosetta.rosetta.core.scoring import ScoreType
            sfx_relax = get_fa_scorefxn()
            sfx_relax.set_weight(ScoreType.coordinate_constraint, 1.0)
            fr = FastRelax(sfx_relax)
            fr.constrain_relax_to_start_coords(True)
            fr.coord_constrain_sidechains(False)
            fr.ramp_down_constraints(False)
            fr.apply(pose)
        rec["score_relaxed"] = round(sfx(pose), 2)

        # InterfaceAnalyzerMover takes chain indices as std::set<int>.
        fixed = pyrosetta.rosetta.std.set_int_t()
        indices = _chain_indices(pose, set(_TARGETS))
        for c in indices:
            try:
                fixed.add(c)
            except AttributeError:
                fixed.insert(c)
        rec["chain_idx"] = indices

        ia = InterfaceAnalyzerMover(fixed)
        ia.set_pack_separated(True)
        ia.set_compute_packstat(True)
        ia.set_compute_interface_sc(True)
        ia.apply(pose)
        data = ia.get_all_data()

        unsat = getattr(data, "delta_unsat_hbonds", None)
        if unsat is None:
            unsat = getattr(data, "delta_unsatHbonds", float("nan"))

        rec.update({
            "dG_separated": round(ia.get_separated_interface_energy(), 3),
            "dsasa": round(ia.get_interface_delta_sasa(), 1),
            "packstat": round(float(data.packstat), 4),
            "sc": round(float(data.sc_value), 4),
            "delta_unsat": float(unsat),
            "nres": float(getattr(data, "interface_nres", 0) or 0)})
    except Exception as e:
        rec = {"seq_id": seq_id, "seed": seed, "path": str(path),
               "error": f"{type(e).__name__}: {e}"}

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rec))
    return rec


def find_models(base: Path, name: str):
    """All seeds of a variant under <base>/<name>/."""
    d = base / name
    if not d.is_dir():
        return []
    for pattern in (f"{name}_model_*.pdb", "*_model_*.pdb", "*.pdb"):
        hits = sorted(d.glob(pattern))
        if hits:
            out = []
            for i, h in enumerate(hits):
                seed = h.stem.rsplit("_", 1)[-1]
                out.append((int(seed) if seed.isdigit() else i, h))
            return out
    return []


def aggregate(values):
    values = [v for v in values if v is not None and v == v]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return round(mean, 4), 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return round(mean, 4), round(var ** 0.5, 4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--loop", default=None)
    ap.add_argument("--models-dir", required=True,
                    help="directory with <variant>/<variant>_model_*.pdb")
    ap.add_argument("--rows", default="order",
                    choices=["pass", "selected", "order", "all"])
    ap.add_argument("--name-col", default="seq_id",
                    help="column with the directory name of the prediction")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--no-relax", action="store_true",
                    help="only for quick tests; without relax one measures clashes")
    ap.add_argument("--dry-run", action="store_true",
                    help="only match and count, no Rosetta")
    a = ap.parse_args()

    cfg = registry.load_config(a.config)
    loop = a.loop or cfg["project"]["loop"]
    loop_dir = Path(cfg["project"]["root"]) / "04_loops" / loop
    targets = list(cfg["template"]["target_chains"])
    binder = cfg["template"]["binder_chain"]

    # Choose rows
    if a.rows in ("selected", "order"):
        src = loop_dir / f"{a.rows}.csv"
        if not src.exists():
            sys.exit(f"{src.name} is missing. The file only comes into being with the live "
                     "run of stage 10, with --open it is not written.")
        rows = registry.read_csv(src)
    else:
        src = loop_dir / "gate.csv"
        if not src.exists():
            src = loop_dir / "master.csv"
        if not src.exists():
            sys.exit(f"Neither gate.csv nor master.csv in {loop_dir}.")
        rows = registry.read_csv(src)
        if a.rows == "pass":
            rows = [r for r in rows
                    if str(r.get("gate_pass", "")).lower() in ("true", "1")]

    if not rows:
        sys.exit(f"No rows from {src.name} (--rows {a.rows}).")
    print(f"{loop}: {len(rows)} rows from {src.name}")
    if a.rows == "selected":
        print("  Note: without controls delta_unsat has no comparison group. "
              "With --rows order they run along.")

    # Strukturen zuordnen
    base = Path(a.models_dir)
    jsondir = loop_dir / "rosetta_json"
    tasks, missing = [], []
    for r in rows:
        name = r.get(a.name_col) or r["seq_id"]
        models = find_models(base, name)
        if not models:
            missing.append(f"{r['seq_id']} ({base / name})")
            continue
        for seed, path in models:
            tasks.append((r["seq_id"], seed, str(path),
                          str(jsondir / f"{r['seq_id']}_{seed}.json")))

    todo = [t for t in tasks if not Path(t[3]).exists()]
    print(f"  {len(tasks)} structures matched, "
          f"{len(tasks) - len(todo)} already scored, {len(todo)} open")
    if missing:
        print(f"  ! {len(missing)} without structures, e.g.:")
        for m in missing[:5]:
            print(f"      {m}")
        print(f"    If all are missing, --name-col is wrong (currently "
              f"'{a.name_col}').")
    if not tasks:
        sys.exit("Nothing matched. Check --models-dir and --name-col.")

    print(f"  estimated {len(todo) * 5 / max(1, a.jobs):.0f} min on {a.jobs} cores "
          f"({'without' if a.no_relax else 'with'} relax)")
    if a.dry_run:
        return 0

    import multiprocessing as mp

    results = []
    if not todo:
        print("  nothing to compute, aggregating only")
    elif a.jobs > 1:
        with mp.Pool(a.jobs, initializer=_init,
                     initargs=(targets, binder, not a.no_relax)) as pool:
            for i, rec in enumerate(pool.imap_unordered(score_one, todo), 1):
                results.append(rec)
                if i % 10 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}", flush=True)
    else:
        _init(targets, binder, not a.no_relax)
        for i, t in enumerate(todo, 1):
            results.append(score_one(t))
            print(f"  {i}/{len(todo)}", flush=True)

    # Include JSONs that already exist
    done_paths = {r.get("path") for r in results}
    for t in tasks:
        if Path(t[3]).exists() and t[2] not in done_paths:
            try:
                results.append(json.loads(Path(t[3]).read_text()))
            except Exception:
                pass

    by_seq, n_err = {}, 0
    for rec in results:
        if rec.get("error"):
            n_err += 1
            continue
        by_seq.setdefault(rec["seq_id"], []).append(rec)

    out = []
    for sid, recs in by_seq.items():
        row = {"seq_id": sid, "ros_n": len(recs)}
        for col in COLS:
            mean, sd = aggregate([r.get(col) for r in recs])
            if mean is not None:
                row[f"ros_{col}"] = mean
                row[f"ros_{col}_sd"] = sd
        out.append(row)

    dest = loop_dir / "scores_rosetta.csv"
    registry.write_csv(dest, out)
    registry.log_stage(loop_dir, "60_rosetta_score",
                       {"n_rows": len(rows), "n_structures": len(tasks),
                        "n_errors": n_err, "models_dir": a.models_dir,
                        "relax": not a.no_relax, "jobs": a.jobs, "rows": a.rows})

    print(f"\n  {len(out)} Varianten -> {dest}")
    if n_err:
        print(f"  ! {n_err} structures with errors, details as JSON in {jsondir}")
    print("  Only the rows are comparable, not the individual values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
