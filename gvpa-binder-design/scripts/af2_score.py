#!/usr/bin/env python
"""AF2 initial-guess scoring -- the orthogonal, discriminative gate.

MUST run in the BindCraft env (colabdesign + jax) with its lib on
LD_LIBRARY_PATH; see env.sh.

Two modes, because the full BindCraft metric set on ~15k sequences is not
affordable:

  --mode gate   complex prediction, one model, loose thresholds. Cheap triage
                that decides which sequences earn the expensive work.
  --mode full   complex prediction on BindCraft's two validation models PLUS a
                free-binder prediction, giving the whole BindCraft AF2 metric
                set and every RMSD.

Protocol (Bennett 2023). A de novo binder has no homologs, so vanilla
single-sequence AF2 folds real designs and composition-matched scrambles alike
to pLDDT ~35 -- an invalid test. Initial guess supplies the design's OWN
backbone as template with the binder's sequence and side chains masked
(rm_binder=False, rm_binder_seq=True, rm_binder_sc=True), and asks AF2 to
re-derive the fold from the threaded sequence.

Units, which differ from AF2 convention and have bitten this project before:
colabdesign normalises to 0-1, so plddt > 0.80 (not 70) and i_pae * 31 ~ A
(Bennett's < 10 A cut is i_pae < 0.32). In protocol="binder" the reported
`plddt` is masked to the BINDER ONLY, not the whole complex.

The three RMSDs, matched to BindCraft's definitions:
  hotspot_rmsd   binder CA, predicted vs designed, IN THE TARGET FRAME (the
                 complex is superposed on target CAs, then the binder is
                 compared with no further alignment). "Did the binder stay
                 docked where it was designed to sit." BindCraft's
                 Hotspot_RMSD, whose name is misleading.
  scrmsd_complex binder CA after superposing on the binder itself: pure fold
                 agreement with the RFD3 backbone. Not a BindCraft metric.
  binder_rmsd    free-binder CA vs the designed backbone, aligned. BindCraft's
                 Binder_RMSD, and the one that is meaningless for a chain that
                 only folds on binding (hence tier 2).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

NAN = float("nan")


def _binder_and_target(pdb_path, binder_len):
    """Split a predicted complex into (binder CA, target CA).

    colabdesign writes the target first and the binder last; the binder is
    identified by length rather than by a hardcoded chain id so a relabelled
    output cannot silently swap the two.
    """
    st = common.read_structure(pdb_path)
    order = list(dict.fromkeys(st.chain))
    last = order[-1]
    b = st.ca(last)
    if len(b) != binder_len:
        cands = [c for c in order if len(st.ca(c)) == binder_len]
        if not cands:
            return None, None
        last = cands[-1]
        b = st.ca(last)
    t = np.concatenate([st.ca(c) for c in order if c != last]) if len(order) > 1 else np.zeros((0, 3))
    return b, t


def target_contact_mask(pdb, cfg, cutoff=10.0):
    """Which target residues the DESIGN places near the binder (in target order).

    Needed because colabdesign's i_pae is the mean PAE over EVERY binder-target
    residue pair. With a three-subunit rib the binder engages one subunit while
    PAE to the two distant subunits saturates near 31 A, so the mean is
    dominated by residues that were never meant to be contacted: i_pae sat at
    ~0.86 even for predictions whose binder was 6 A from its designed pose.

    Bennett's <10 A interface-PAE criterion was defined against a single
    compact target. Restricting the average to the residues the design actually
    approaches restores that meaning.
    """
    st = common.read_structure(pdb)
    bx = st.xyz[st.chain == cfg["target"]["binder_chain"]]
    mask = []
    for c in cfg["target"]["chain_map"].values():
        for rs, _, idx in st.residues(c):
            d = np.linalg.norm(st.xyz[idx][:, None, :] - bx[None, :, :], axis=2).min()
            mask.append(d <= cutoff)
    return np.array(mask, dtype=bool)


def interface_pae(aux_pae, t_mask, binder_len):
    """Mean PAE (Angstroms) between the binder and the contacted target residues."""
    p = np.asarray(aux_pae, dtype=float)
    p = (p + p.T) / 2.0
    tL = len(t_mask)
    if p.shape[0] < tL + binder_len or not t_mask.any():
        return float("nan")
    sub = p[tL:tL + binder_len, :tL][:, t_mask]
    return float(sub.mean())


def restore_chain_identity(model, pdb, cfg):
    """Tell AF2-multimer the target is several chains, because it is.

    colabdesign's binder protocol assumes a SINGLE contiguous target: in
    af/prep.py it sets `_lengths = [target_len, binder_len]` and derives
    asym_id/sym_id/entity_id from that. With a three-subunit GvpA rib as the
    target, AF2-multimer is therefore told that three physically separate
    55-residue chains are one continuous 165-residue polypeptide, and it tries
    to fold them as one. The interface never forms.

    Measured on this campaign before the fix: i_ptm 0.07-0.12 and i_pae
    0.78-0.89 (~24 A) on every design, while binder pLDDT stayed 0.64-0.83 --
    AF2 folding the binder happily while seeing nothing docked.

    Only the chain-identity features are rewritten. `_target_len` and
    `_binder_len` are untouched, so the loss masks (target = first tL, binder =
    last bL) and every reported metric keep their meaning. Identical subunits
    share an entity_id, which is what marks them as copies of one entity.
    """
    tchains = list(cfg["target"]["chain_map"].values())
    if len(tchains) <= 1:
        return False
    st = common.read_structure(pdb)
    tlens = [len(st.ca(c)) for c in tchains]
    blen = int(model._binder_len)
    lens = tlens + [blen]
    if sum(lens) != sum(model._lengths):
        raise RuntimeError(
            f"chain-identity fix: lengths {lens} (sum {sum(lens)}) do not match "
            f"colabdesign's {model._lengths} (sum {sum(model._lengths)}) for {pdb}")
    asym = np.concatenate([[n] * L for n, L in enumerate(lens)])
    ent = np.concatenate([[0] * L for L in tlens] + [[1] * blen])
    model._lengths = lens
    model._inputs.update({"asym_id": asym, "sym_id": asym, "entity_id": ent})
    return True


def _superpose(mobile, ref):
    """Rotation+translation taking `mobile` onto `ref` (Kabsch)."""
    mc, rc = mobile.mean(0), ref.mean(0)
    V, S, Wt = np.linalg.svd((mobile - mc).T @ (ref - rc))
    d = np.sign(np.linalg.det(V @ Wt))
    U = V @ np.diag([1.0, 1.0, d]) @ Wt
    return U, mc, rc


def rmsds(pred_pdb, ref_pdb, cfg, binder_len):
    """hotspot_rmsd, scrmsd_complex, target_rmsd for one predicted complex."""
    out = {"hotspot_rmsd": NAN, "scrmsd_complex": NAN, "target_rmsd": NAN}
    pb, pt = _binder_and_target(pred_pdb, binder_len)
    if pb is None:
        return out
    ref = common.read_structure(ref_pdb)
    rb = ref.ca(cfg["target"]["binder_chain"])
    rt = np.concatenate([ref.ca(c) for c in cfg["target"]["chain_map"].values()])
    if len(pb) != len(rb) or len(pt) != len(rt):
        return out

    # Fold agreement: binder onto binder.
    out["scrmsd_complex"] = round(common.kabsch_rmsd(pb, rb), 3)
    # Docking agreement: put the prediction in the designed target's frame,
    # then measure the binder WITHOUT realigning it.
    U, mc, rc = _superpose(pt, rt)
    out["target_rmsd"] = round(float(np.sqrt((((pt - mc) @ U + rc - rt) ** 2).sum(1).mean())), 3)
    pb_f = (pb - mc) @ U + rc
    out["hotspot_rmsd"] = round(float(np.sqrt(((pb_f - rb) ** 2).sum(1).mean())), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["gate", "full"], default="gate")
    ap.add_argument("--models", type=int, nargs="+", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--save-pdb-dir", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="skip manifest rows already present in --out")
    a = ap.parse_args()

    cfg = common.load_config(a.config)
    af = cfg["af2"]
    data_dir = a.data_dir or af["params_dir"]
    models = a.models if a.models is not None else (
        af["gate_models"] if a.mode == "gate" else af["full_models"])
    recycles = af["num_recycles"]
    # The MULTIMER path silently ignores use_templates: with templates on and
    # off it returned bit-identical scores, and the binder landed 46 A from its
    # designed pose. The monomer model actually applies the initial guess
    # (i_ptm 0.14 -> 0.54, binder displacement 46 A -> 6-11 A) and is what
    # Bennett 2023 used. Overridable, but do not turn it back on casually.
    use_multimer = bool(cfg["af2"].get("use_multimer", False))
    pae_cut = float(cfg["af2"].get("interface_pae_contact_cutoff", 10.0))
    save_dir = Path(a.save_pdb_dir) if a.save_pdb_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    with open(a.manifest) as fh:
        man = list(csv.DictReader(fh))
    if not man:
        raise SystemExit(f"empty manifest: {a.manifest}")

    # Resume support. A time-boxed interactive session gets killed mid-stage,
    # and without this the whole shard restarts from zero. Rows that already
    # carry a score for the first model are complete and are skipped; partial
    # output is preserved and appended to.
    done = {}
    if a.resume and Path(a.out).exists():
        key = f"m{models[0]}_plddt"
        with open(a.out) as fh:
            for r in csv.DictReader(fh):
                if _scored(r.get(key, "")):
                    done[r["id"]] = r
        if done:
            before = len(man)
            man = [r for r in man if r["id"] not in done]
            print(f"resume: {len(done)} already scored, {len(man)} of {before} left",
                  flush=True)
    if not man and a.mode != "full":
        print("resume: nothing left to score")
        _write(list(done.values()), a.out)
        return
    if not man:
        # full mode still falls through: pass B below is selected independently
        # and may have free-binder rows to backfill for this shard.
        print("resume: complex pass complete, checking free-binder pass")

    from colabdesign import clear_mem, mk_afdesign_model

    tchains = ",".join(cfg["target"]["chain_map"].values())
    bchain = cfg["target"]["binder_chain"]

    # Rows are keyed by id so the two passes below can write into the same
    # record without a merge step.
    rec = {r["id"]: {"id": r["id"], "design": r["design"], "seq": r["seq"],
                     "binder_len": int(r["binder_len"]), "pdb": r["pdb"]}
           for r in man}
    rec.update(done)   # carry finished rows through to the output

    # ---------------------------------------------------------------- pass A
    # Complex prediction. Grouped by backbone so the template and params are
    # prepped once per backbone and reused across its 8 sequences -- setup
    # dominates, prediction itself is seconds.
    groups = {}
    for r in man:
        groups.setdefault(r["pdb"], []).append(r)

    _announced = False
    done = 0
    for pdb, grp in groups.items():
        try:
            clear_mem()
            cx = mk_afdesign_model(protocol="binder", use_multimer=use_multimer,
                                   use_templates=True, data_dir=data_dir)
            cx.prep_inputs(pdb_filename=pdb, target_chain=tchains, binder_chain=bchain,
                           rm_binder=False, rm_binder_seq=True, rm_binder_sc=True)
            fixed = restore_chain_identity(cx, pdb, cfg)
            t_mask = target_contact_mask(pdb, cfg, pae_cut)
            if fixed and not _announced:
                print(f"chain-identity fix applied: target modelled as "
                      f"{len(cfg['target']['chain_map'])} chains, not one", flush=True)
                _announced = True
        except Exception as e:
            print(f"PREP-ERR {pdb}: {type(e).__name__}: {e}", flush=True)
            for r in grp:
                rec[r["id"]]["error"] = "prep"
            continue

        for r in grp:
            d, blen = rec[r["id"]], int(r["binder_len"])
            for m in models:
                try:
                    try:
                        cx.predict(seq=r["seq"], num_recycles=recycles, models=[m], verbose=False)
                    except Exception:
                        # GPU-level failures here are usually transient
                        # (cuSolver workspace, allocator contention). One retry
                        # costs seconds; a NaN costs the design.
                        cx.predict(seq=r["seq"], num_recycles=recycles, models=[m], verbose=False)
                    log = cx.aux["log"]
                    # NOTE: in the binder protocol `plddt` is BINDER-ONLY.
                    d[f"m{m}_plddt"] = float(log.get("plddt", NAN))
                    d[f"m{m}_ptm"] = float(log.get("ptm", NAN))
                    d[f"m{m}_i_ptm"] = float(log.get("i_ptm", NAN))
                    d[f"m{m}_pae"] = float(log.get("pae", NAN))
                    # colabdesign's own i_pae, averaged over ALL target
                    # residues, is reported for traceability but is not what
                    # the gate uses on a multi-subunit target.
                    d[f"m{m}_i_pae_all"] = float(log.get("i_pae", NAN))
                    ipae_A = interface_pae(cx.aux.get("pae"), t_mask, blen)
                    d[f"m{m}_i_pae_A"] = round(ipae_A, 3)
                    # Normalised the same way colabdesign does (/31), so the
                    # BindCraft threshold of 0.35 keeps its meaning.
                    d[f"m{m}_i_pae"] = round(ipae_A / 31.0, 4) if ipae_A == ipae_A else NAN
                    if save_dir:
                        q = save_dir / f"{r['id']}_m{m}.pdb"
                        cx.save_pdb(str(q))
                        d[f"m{m}_complex_pdb"] = str(q)
                        for k, v in rmsds(str(q), pdb, cfg, blen).items():
                            d[f"m{m}_{k}"] = v
                except Exception as e:
                    print(f"PRED-ERR {r['id']} m{m}: {type(e).__name__}: {e}", flush=True)
            done += 1
            if done % 25 == 0:
                _write(list(rec.values()), a.out)
                print(f"  complex {done}/{len(man)}", flush=True)

    # ---------------------------------------------------------------- pass B
    # Free-binder prediction, full mode only. Run as a SEPARATE pass grouped by
    # length: the hallucination model is length-bound and building it needs
    # clear_mem(), which evicts the complex model. Interleaving the two would
    # rebuild both for every sequence.
    if a.mode == "full":
        # Pass A's --resume filter strips finished rows out of `man`, so driving
        # pass B off `man` silently skips the free-binder prediction for every
        # row an earlier session already scored -- and the next resume skips it
        # again, permanently. Select on the alone score itself instead, so an
        # interrupted run backfills what it missed.
        akey = f"m{models[0]}_binder_alone_plddt"
        todo = [d for d in rec.values() if not _scored(d.get(akey, ""))]
        by_len = {}
        for d in todo:
            by_len.setdefault(int(d["binder_len"]), []).append(d)
        done = 0
        print(f"alone: {len(rec) - len(todo)} already scored, {len(todo)} to score",
              flush=True)
        for blen, grp in sorted(by_len.items()):
            try:
                clear_mem()
                al = mk_afdesign_model(protocol="hallucination", use_templates=False,
                                       initial_guess=False, data_dir=data_dir)
                al.prep_inputs(length=blen)
            except Exception as e:
                print(f"ALONE-PREP-ERR len={blen}: {type(e).__name__}: {e}", flush=True)
                continue
            for d in grp:
                try:
                    al.set_seq(d["seq"])
                    ref_b = common.read_structure(d["pdb"]).ca(bchain)
                    for m in models:
                        al.predict(models=[m], num_recycles=recycles, verbose=False)
                        alog = al.aux["log"]
                        d[f"m{m}_binder_alone_plddt"] = float(alog.get("plddt", NAN))
                        d[f"m{m}_binder_alone_ptm"] = float(alog.get("ptm", NAN))
                        d[f"m{m}_binder_alone_pae"] = float(alog.get("pae", NAN))
                        if save_dir:
                            q = save_dir / f"{d['id']}_alone_m{m}.pdb"
                            al.save_pdb(str(q))
                            d[f"m{m}_alone_pdb"] = str(q)
                            ab = common.read_structure(str(q))
                            ac = ab.ca(list(dict.fromkeys(ab.chain))[0])
                            d[f"m{m}_binder_rmsd"] = (
                                round(common.kabsch_rmsd(ac, ref_b), 3)
                                if len(ac) == len(ref_b) else NAN)
                except Exception as e:
                    print(f"ALONE-ERR {d['id']}: {type(e).__name__}: {e}", flush=True)
                done += 1
                if done % 25 == 0:
                    _write(list(rec.values()), a.out)
                    print(f"  alone {done}/{len(todo)}", flush=True)

    rows = list(rec.values())
    _write(rows, a.out)

    # A NaN here is a lost design, not a bad one. Silent partial failure is the
    # worst outcome, so the rate is always reported and a large one is fatal.
    m0 = models[0]
    bad = [r for r in rows if r.get(f"m{m0}_plddt") != r.get(f"m{m0}_plddt")
           or f"m{m0}_plddt" not in r]
    frac = len(bad) / max(len(rows), 1)
    print(f"wrote {len(rows)} rows -> {a.out}")
    print(f"failed predictions: {len(bad)}/{len(rows)} ({frac:.1%})")
    if frac > 0.02:
        print("  ".join(str(r["id"]) for r in bad[:10]))
        raise SystemExit(
            f"{frac:.1%} of AF2 predictions failed -- above the 2% tolerance. "
            f"These become NaN and are dropped by the gate, biasing the campaign "
            f"silently. Check the log for cuSolver/OOM errors before continuing.")


def _scored(v):
    """True if a CSV cell holds a real score. Blank means never run; the string
    "nan" means the prediction ran and failed, which is final, not retryable."""
    return v not in ("", None) and v == v


def _write(rows, path):
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); cols.append(k)
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
