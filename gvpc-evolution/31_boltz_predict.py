#!/usr/bin/env python3
"""
Stage 6: run Boltz.

    $BOLTZ_PY 31_boltz_predict.py --config config.yaml --mode constrained

Calls boltz predict on the input directory from stage 5 and afterwards puts
every job YAML next to its prediction, so that it can later be traced what it
was computed with.

Two flags are not optional:
  --diffusion_samples  at least 2. bz_shape_rmsd arises from the spread
                       across the seeds and is the first gate criterion; with
                       one sample there is no gate.
  --write_full_pae     otherwise Boltz writes no pae_*.npz and the
                       interface PAE column in stage 7 stays empty.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_registry as registry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--loop", default=None)
    ap.add_argument("--mode", default="constrained", choices=["constrained", "free"])
    ap.add_argument("--diffusion-samples", type=int, default=None)
    ap.add_argument("--devices", default="1")
    ap.add_argument("--dry-run", action="store_true", help="only show the command")
    a = ap.parse_args()

    cfg = registry.load_config(a.config)
    loop = a.loop or cfg["project"]["loop"]
    root = Path(cfg["project"]["root"])
    loop_dir = root / "04_loops" / loop
    bcfg = cfg["boltz"]
    n_samples = a.diffusion_samples or int(bcfg["diffusion_samples"])

    indir = root / "02_input" / "boltz_input" / f"{loop}_{a.mode}"
    outdir = root / "03_output" / "boltz_output" / f"{loop}_{a.mode}"

    yamls = sorted(indir.glob("*.yaml"))
    if not yamls:
        sys.exit(f"No YAMLs in {indir}. Run stage 5 first.")
    stray = [p.name for p in indir.iterdir() if p.suffix != ".yaml"]
    if stray:
        sys.exit(f"Foreign files in {indir}: {stray}\n"
                 "boltz predict takes everything from the directory. Clean up "
                 "or rerun stage 5, which creates it fresh.")
    if a.mode == "constrained" and n_samples < 2:
        sys.exit(f"--diffusion-samples {n_samples}: bz_shape_rmsd arises from "
                 "the spread across the seeds. With fewer than 2 there is no gate.")

    cmd = [str(bcfg.get("bin") or cfg["envs"]["boltz_bin"]), "predict", str(indir),
           "--out_dir", str(outdir),
           "--diffusion_samples", str(n_samples),
           "--devices", str(a.devices),
           "--output_format", "pdb",
           "--write_full_pae"]
    if bcfg.get("use_potentials", True):
        cmd.append("--use_potentials")
    # The MSA server is only needed if the target chains do not already have
    # their MSA in the YAML.
    if bcfg["msa_policy"] == "empty_binder_only" and not str(bcfg.get("target_msa") or "").strip():
        cmd.append("--use_msa_server")

    print(f"[boltz] {len(yamls)} YAMLs, {n_samples} seeds, mode {a.mode}")
    print("  $", " ".join(cmd), flush=True)
    if a.dry_run:
        return 0

    outdir.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"\n[boltz] ended with {rc}. Collecting what has finished; "
              "the missing variants are listed below.",
              file=sys.stderr)

    preds = next(iter(outdir.glob("*/predictions")), None)
    if preds is None:
        sys.exit(f"No predictions directory under {outdir}.")

    missing = []
    for y in yamls:
        d = preds / y.stem
        if d.exists():
            shutil.copy2(y, d / y.name)
        else:
            missing.append(y.stem)

    registry.log_stage(loop_dir, f"31_boltz_predict_{a.mode}",
                       {"n_yaml": len(yamls), "n_missing": len(missing),
                        "missing": missing[:20], "diffusion_samples": n_samples,
                        "returncode": rc, "predictions": str(preds)})

    print(f"\n[boltz] predictions: {preds}")
    print(f"        {len(yamls) - len(missing)}/{len(yamls)} variants computed")
    if missing:
        print(f"        missing ({len(missing)}): {', '.join(missing[:10])}"
              + (" ..." if len(missing) > 10 else ""))
        print("        Usually a template parse error; Boltz then skips the "
              "YAML with 'Failed to process ... Skipping'.")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
