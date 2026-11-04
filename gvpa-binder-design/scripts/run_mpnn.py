#!/usr/bin/env python
"""Design binder sequences on triaged backbones with ProteinMPNN.

Uses `mpnn.inference.MPNNInferenceEngine` directly rather than shelling out to
the `mpnn` console script once per backbone. The CLI takes a single
`--structure_path`, so a 1900-backbone campaign would pay the model-load cost
1900 times; the engine's `run(input_dicts=[...])` accepts a list, so the
checkpoint is loaded once per process instead.

Only chain A (the binder) is designed -- the three GvpA chains are context and
must come back unchanged, which is asserted rather than assumed.

    python scripts/run_mpnn.py [--shard i --n-shards N] [--limit N]

Writes <run>/seqs/<shard>/*.fa and <run>/sequences.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

CHUNK = 40  # backbones handed to the engine at once; bounds peak memory


def parse_fasta(path):
    """Yield (header, sequence) from an MPNN .fa file."""
    name, buf = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf)
                name, buf = line[1:], []
            elif line:
                buf.append(line)
    if name is not None:
        yield name, "".join(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    a = ap.parse_args()

    cfg = common.load_config(a.config)
    root = Path(cfg["run_root"]) / cfg["campaign"]
    mp = cfg["mpnn"]

    with open(root / "backbones.csv") as fh:
        bb = [r for r in csv.DictReader(fh) if r.get("pass") == "1"]
    bb.sort(key=lambda r: r["design"])
    if a.limit:
        bb = bb[: a.limit]
    mine = bb[a.shard :: a.n_shards]
    if not mine:
        print(f"shard {a.shard}: nothing to do")
        return

    out_dir = root / "seqs" / f"shard{a.shard:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from mpnn.inference import MPNNInferenceEngine

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = MPNNInferenceEngine(
        model_type="protein_mpnn",
        checkpoint_path=str(mp.get("checkpoint")
                            or Path.home() / "proteinmpnn_v_48_020.pt"),
        is_legacy_weights=True,
        out_directory=str(out_dir),
        write_fasta=True,
        write_structures=False,     # the RFD3 backbone is the structure of record
        device=device,
    )

    # The engine validates `omit` against a THREE-LETTER vocabulary, so a
    # config written as "C" has to be expanded or it fails at validation time.
    omit = [common.ONE_TO_THREE.get(c, c) for c in str(mp.get("omit", "")).replace(",", "")]

    ckpt = str(mp.get("checkpoint") or Path.home() / "proteinmpnn_v_48_020.pt")
    if not Path(ckpt).exists():
        raise SystemExit(f"MPNN checkpoint not found: {ckpt}")
    print(f"shard {a.shard}/{a.n_shards}: {len(mine)} backbones on {device}", flush=True)
    print(f"  weights: {Path(ckpt).name}", flush=True)
    rows = []
    # Resume: a backbone whose FASTA already holds the full set of designs was
    # finished by an earlier session. Only COMPLETE files are skipped, so the
    # append-bug guard below still holds.
    want_n = mp["n_sequences"]
    def complete(design):
        fa = out_dir / f"{design}.fa"
        if not fa.exists():
            return False
        return sum(1 for _ in parse_fasta(fa)) >= want_n
    pre = len(mine)
    mine = [r for r in mine if not complete(r["design"])]
    if len(mine) < pre:
        print(f"resume: {pre - len(mine)} backbones already designed, "
              f"{len(mine)} left", flush=True)

    for start in range(0, len(mine), CHUNK):
        chunk = mine[start : start + CHUNK]
        inputs = [
            dict(structure_path=r["pdb"], name=r["design"], seed=abs(hash(r["design"])) % (2**31),
                 batch_size=mp["n_sequences"], number_of_batches=1,
                 designed_chains=[cfg["target"]["binder_chain"]],
                 omit=omit, temperature=mp["temperature"])
            for r in chunk
        ]
        # MPNN APPENDS to an existing <name>.fa. Re-running this stage (a resume,
        # a retry, --from 03) would therefore stack a second set of designs into
        # the same file under the SAME ids -- b0_d0..d7 twice, different
        # sequences. Downstream keys records by id, so one would silently
        # overwrite the other and metrics would be attributed to the wrong
        # sequence. Clear the target FASTAs first.
        for r in chunk:
            (out_dir / f"{r['design']}.fa").unlink(missing_ok=True)
        try:
            engine.run(input_dicts=inputs)
        except Exception as e:
            print(f"  ENGINE-ERR chunk at {start}: {type(e).__name__}: {e}", flush=True)
            continue

        for r in chunk:
            fa = out_dir / f"{r['design']}.fa"
            if not fa.exists():
                print(f"  MISSING fasta for {r['design']}", flush=True)
                continue
            blen = int(r["binder_len"])
            for hdr, seq in parse_fasta(fa):
                sid = hdr.split(",")[0].strip()
                rec = ""
                if "sequence_recovery=" in hdr:
                    rec = hdr.split("sequence_recovery=")[1].strip()
                binder = seq[:blen]
                if len(seq) < blen:
                    print(f"  SHORT seq for {sid}: {len(seq)} < {blen}", flush=True)
                    continue
                rows.append({
                    "id": sid, "design": r["design"], "cell": r["cell"], "arm": r["arm"],
                    "length_bin": r["length_bin"], "binder_len": blen,
                    "seq": binder, "target_seq": seq[blen:],
                    "mpnn_seq_recovery": rec, "pdb": r["pdb"],
                })
        print(f"  {min(start+CHUNK, len(mine))}/{len(mine)} backbones", flush=True)

    # Ids must be unique: everything downstream (manifests, AF2 scores, the
    # merged table) is keyed by id, so a collision silently mixes designs.
    seen = {}
    for r in rows:
        if r["id"] in seen and seen[r["id"]] != r["seq"]:
            raise SystemExit(
                f"duplicate sequence id {r['id']} with differing sequences -- "
                f"stale FASTAs in {out_dir}. Delete that directory and re-run.")
        seen[r["id"]] = r["seq"]

    # The target must be context, never redesigned. If MPNN ever mutated it the
    # AF2 complex would be scored against a different protein than the one the
    # binder was designed for, and nothing downstream would notice.
    if rows:
        ref = rows[0]["target_seq"]
        bad = sum(1 for r in rows if r["target_seq"] != ref)
        if bad:
            raise SystemExit(f"target sequence differs on {bad}/{len(rows)} rows -- "
                             f"designed_chains is not confining MPNN to the binder")
        print(f"target context identical across all {len(rows)} sequences ({len(ref)} aa) -- ok")

    # Gather from ALL complete FASTAs in this shard, so a resumed session
    # re-emits the rows earlier sessions produced instead of truncating them.
    have = {r["id"] for r in rows}
    for r in bb:
        fa = out_dir / f"{r['design']}.fa"
        if not fa.exists():
            continue
        blen = int(r["binder_len"])
        for hdr, seq in parse_fasta(fa):
            sid = hdr.split(",")[0].strip()
            if sid in have or len(seq) < blen:
                continue
            rec = hdr.split("sequence_recovery=")[1].strip() if "sequence_recovery=" in hdr else ""
            rows.append({"id": sid, "design": r["design"], "cell": r["cell"],
                         "arm": r["arm"], "length_bin": r["length_bin"],
                         "binder_len": blen, "seq": seq[:blen],
                         "target_seq": seq[blen:], "mpnn_seq_recovery": rec,
                         "pdb": r["pdb"]})
            have.add(sid)

    out = root / (f"sequences.shard{a.shard:03d}.csv" if a.n_shards > 1 else "sequences.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "design", "cell", "arm", "length_bin",
                                           "binder_len", "seq", "mpnn_seq_recovery", "pdb"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} sequences -> {out}")


if __name__ == "__main__":
    main()
