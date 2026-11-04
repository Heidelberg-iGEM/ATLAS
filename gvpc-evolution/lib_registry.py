"""
lib_registry.py - the shared bookkeeping of the pipeline.

Hard rule: standard library only. This module is imported from four
different conda envs. Any import of numpy, pandas or gemmi makes the pipeline
unusable in at least one of them.

Basic principle: one CSV per round, one row per sequence, every stage appends
columns. Nothing is parsed back out of file names, nothing is deleted.

    python lib_registry.py        # self-test
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
# If PyYAML is available in the active env, it is used. Otherwise the
# mini parser below takes over, so that the config is also readable from an env
# in which PyYAML is missing. It covers exactly what occurs in config.yaml:
# nested mappings via indentation, scalars, inline lists and
# block lists. No claim to complete YAML.

_INT = re.compile(r"^-?\d+$")
_FLOAT = re.compile(r"^-?\d*\.\d+([eE][-+]?\d+)?$")


def _scalar(token: str):
    t = token.strip()
    if t.startswith("#"):
        return None
    if "#" in t and not t.startswith(('"', "'")):
        t = t.split("#", 1)[0].strip()
    if not t:
        return None
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        return t[1:-1]
    low = t.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~"):
        return None
    if _INT.match(t):
        return int(t)
    if _FLOAT.match(t):
        return float(t)
    return t


def _parse_inline_list(s: str):
    """[a, b, [c, d]] -> nested list."""
    s = s.strip()
    out, cur, parents, buf = [], None, [], ""
    cur = out
    for i, ch in enumerate(s):
        if ch == "[":
            if i > 0:
                new = []
                cur.append(new)
                parents.append(cur)
                cur = new
        elif ch == "]":
            if buf.strip():
                cur.append(_scalar(buf))
            buf = ""
            if parents:
                cur = parents.pop()
            else:
                break            # everything after the closing bracket is a comment
        elif ch == ",":
            if buf.strip():
                cur.append(_scalar(buf))
            buf = ""
        else:
            buf += ch
    if buf.strip():
        cur.append(_scalar(buf))
    return out


def _mini_yaml(text: str) -> dict:
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            lines.append((len(raw) - len(raw.lstrip(" ")), s))

    def gather_list(i, start):
        """Collect a multi-line inline list until the brackets balance."""
        acc = start
        i += 1
        while acc.count("[") != acc.count("]") and i < len(lines):
            acc += " " + lines[i][1]
            i += 1
        return _parse_inline_list(acc), i

    def block(i, indent):
        if lines[i][1].startswith("- "):
            out = []
            while i < len(lines) and lines[i][0] == indent and lines[i][1].startswith("- "):
                item = lines[i][1][2:].strip()
                if item.startswith("["):
                    val, i = gather_list(i, item)
                    out.append(val)
                    continue
                out.append(_scalar(item))
                i += 1
            return out, i

        out = {}
        while i < len(lines) and lines[i][0] == indent:
            ind, s = lines[i]
            if ":" not in s:
                raise ValueError(f"Line without ':' is not supported: {s!r}")
            key, _, rest = s.partition(":")
            key, rest = key.strip(), rest.strip()
            if rest.startswith("["):
                out[key], i = gather_list(i, rest)
            elif rest == "" or rest.startswith("#"):
                i += 1
                if i < len(lines) and lines[i][0] > ind:
                    if lines[i][1].startswith("["):
                        out[key], i = gather_list(i, lines[i][1])
                    else:
                        out[key], i = block(i, lines[i][0])
                else:
                    out[key] = None
            else:
                out[key] = _scalar(rest)
                i += 1
        return out, i

    if not lines:
        return {}
    return block(0, lines[0][0])[0]


def load_config(path) -> dict:
    path = Path(path)
    try:
        import yaml
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        return _mini_yaml(path.read_text(encoding="utf-8"))


def resolve(cfg: dict, relpath: str) -> Path:
    """Resolve a relative path from the config against project.root."""
    p = Path(relpath)
    return p if p.is_absolute() else Path(cfg["project"]["root"]) / p


# --------------------------------------------------------------------------
# IDs, hashes, sequence distance
# --------------------------------------------------------------------------

def seq_sha8(seq: str) -> str:
    """Short hash of the sequence. Basis for dedup and for assigning
    lab measurements."""
    return hashlib.sha256(seq.strip().upper().encode("ascii")).hexdigest()[:8]


def make_seq_id(loop: str, n: int) -> str:
    return f"{loop}-{n:04d}"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def seq_identity(a: str, b: str) -> float:
    """Position-wise identity, 0 to 1. For unequal lengths it is cut to the
    shorter one, which is itself already a hint of an error."""
    a, b = a.strip().upper(), b.strip().upper()
    n = min(len(a), len(b))
    return sum(x == y for x, y in zip(a[:n], b[:n])) / n if n else 0.0


def identity_to_wt(seq: str, native: str) -> float:
    """1.00 is the wild type, smaller means more experimental. Makes visible whether
    a selection covers a range or collapses into one corner."""
    return round(seq_identity(seq, native), 4)


# The helicity level can appear under different names depending on the
# spelling; the first one found is taken.
HELICITY_COLS = ("helicity_level", "helicity", "hel_level", "cell_helicity",
                 "helicity_tier", "hel")


def find_helicity_col(rows: list[dict]) -> str | None:
    if not rows:
        return None
    for c in HELICITY_COLS:
        if c in rows[0] and any(str(r.get(c, "")).strip() for r in rows):
            return c
    return None


def stratum(row: dict, cols: list[str]) -> str:
    """Cell key of the experimental design, e.g. 'T2|high'."""
    return "|".join(str(row.get(c, "") or "?") for c in cols)


# Not \b: an underscore is a word character, so \bT1\b would not match in "tier_T1_high".
_TIER = re.compile(r"(?<![A-Za-z0-9])T\d+(?![A-Za-z0-9])")


def tier_of(cell: str) -> str:
    """Tier from a cell name: 'tier_T1_high' -> 'T1'.

    Do not use cell.split('_')[0], that returns 'tier' and throws all tiers
    into one pot.
    """
    m = _TIER.search(str(cell or ""))
    return m.group(0) if m else str(cell or "?").split("_")[0]


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------

def read_csv(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Write to a .tmp first, then rename. An aborted job must not leave
    half a table behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})
    os.replace(tmp, path)


def attach(base_rows, new_cols: dict[str, dict], key: str = "seq_id"):
    """Append columns, join via seq_id. Missing IDs stay empty - that
    a stage did not run on all rows is the normal case."""
    return [{**row, **new_cols.get(row[key], {})} for row in base_rows]


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------
# Rows that are not design candidates but have to go through all prediction
# stages: the wild type (reference), probes (probe) and the controls from
# stage 3 (control).
CONTROL_MODES = ("control", "reference", "probe")


def is_control(row: dict) -> bool:
    return str(row.get("design_mode", "")).lower().startswith(CONTROL_MODES)


def to_predict(rows: list[dict]) -> list[dict]:
    """Which rows go into a prediction run.

    Prefilter passers without duplicates, plus all controls regardless of the
    prefilter. The EAAAK control breaks the composition filter by
    construction; exactly that is why it is the negative control, and exactly
    that is why the filter must not remove it before the GPU.
    """
    out, seen = [], set()
    for r in rows:
        if r.get("dup_of") and not is_control(r):
            continue
        passed = str(r.get("pass_comp", "")).lower() in ("true", "1")
        if (passed or is_control(r)) and r["seq_id"] not in seen:
            seen.add(r["seq_id"])
            out.append(r)
    return out


# --------------------------------------------------------------------------
# Lab measurements
# --------------------------------------------------------------------------
EXPERIMENTAL = "experimental.csv"


def read_experimental(loops_dir) -> dict[str, dict]:
    """Measurements from 04_loops/experimental.csv, {seq_sha8: {column: value}}.

    Spans rounds and is maintained by hand. Keyed by the hash of the
    sequence, not by seq_id: IDs are reassigned every round, the sequence
    stays. A measurement belongs to the sequence, not to the row.

    Required column seq_sha8 or binder_seq; everything else is free.
    """
    path = Path(loops_dir) / EXPERIMENTAL
    out = {}
    for r in read_csv(path):
        sha = (r.get("seq_sha8") or "").strip()
        seq = (r.get("binder_seq") or "").strip().upper()
        if not sha and seq:
            sha = seq_sha8(seq)
        if not sha:
            continue
        if seq and seq_sha8(seq) != sha:
            raise ValueError(
                f"{path}: seq_sha8 {sha} does not match binder_seq "
                f"({seq_sha8(seq)}). A wrongly assigned measurement is "
                "worse than none.")
        out[sha] = {k: v for k, v in r.items()
                    if k not in ("seq_sha8", "binder_seq") and v not in ("", None)}
    return out


def attach_experimental(rows: list[dict], loops_dir) -> dict:
    """Attach measurements via the sequence hash.

    The return value contains orphan: hashes from experimental.csv for which there is
    no sequence in this round. Without this list such a measurement
    is simply silently missing from every evaluation.
    """
    exp = read_experimental(loops_dir)
    if not exp:
        return {"n": 0, "n_exp": 0, "orphan": []}
    seen, n = set(), 0
    for r in rows:
        sha = r.get("seq_sha8") or (seq_sha8(r["binder_seq"]) if r.get("binder_seq") else "")
        vals = exp.get(sha)
        if vals:
            r.update(vals)
            r["has_experiment"] = True
            seen.add(sha)
            n += 1
        else:
            r.setdefault("has_experiment", False)
    return {"n": n, "n_exp": len(exp), "orphan": sorted(set(exp) - seen)}


# --------------------------------------------------------------------------
# Join
# --------------------------------------------------------------------------
EMPTYISH = ("", "nan", "NaN", "NA", "N/A", "None", "null")


def n_filled(rows: list[dict], col: str) -> int:
    return sum(1 for r in rows if str(r.get(col, "")).strip() not in EMPTYISH)


def join_all(loop_dir, key: str = "seq_id", report: bool = True) -> list[dict]:
    """Join registry.csv and all scores_*.csv into one table.

    The order of the files does not matter, that is the point of the registry.

    The report is not decoration: a join on the wrong key does not abort,
    it enters empty strings everywhere. The pipeline then runs through cleanly and
    delivers plausible numbers on a subset that nobody noticed. That is why
    for each source file it is printed how many rows hit and
    how full each new column has become.
    """
    loop_dir = Path(loop_dir)
    base = read_csv(loop_dir / "registry.csv")
    if not base:
        raise FileNotFoundError(f"no registry.csv in {loop_dir}")
    if key not in base[0]:
        raise KeyError(f"registry.csv has no column '{key}'")
    base_ids = {r[key] for r in base}
    warn = []

    if report:
        print(f"Join   registry.csv: {len(base)} rows, {len(base_ids)} IDs")
        if len(base_ids) != len(base):
            warn.append(f"registry.csv has duplicate {key}")

    for scorefile in sorted(loop_dir.glob("scores_*.csv")):
        rows = read_csv(scorefile)
        if not rows:
            if report:
                print(f"  {scorefile.name:<24} EMPTY")
            warn.append(f"{scorefile.name} is empty - did the stage really run?")
            continue
        if key not in rows[0]:
            raise KeyError(f"{scorefile.name} has no column '{key}'. Without a "
                           "key there is no join, only empty columns.")
        idx = {r[key]: {k: v for k, v in r.items() if k != key}
               for r in rows if (r.get(key) or "").strip()}
        hit = base_ids & set(idx)
        orphan = sorted(set(idx) - base_ids)

        collisions = {k for r in base for k in idx.get(r[key], {})
                      if k in r and r[k] != ""}
        if collisions:
            raise ValueError(f"{scorefile.name} overwrites occupied columns "
                             f"{sorted(collisions)}. Prefix the column names "
                             "(af_, bz_, reg_, ros_).")

        newcols = [k for k in rows[0] if k != key]
        base = attach(base, idx, key=key)

        if report:
            print(f"  {scorefile.name:<24}{len(rows):>5} rows, "
                  f"{len(hit):>5} hit, {len(orphan):>4} without partner")
            for c in newcols:
                f = n_filled(base, c)
                print(f"      {c:<28}{f:>5}/{len(base)}"
                      + ("   <-- EMPTY" if f == 0 else ""))
                if f == 0:
                    warn.append(f"{scorefile.name}: column '{c}' is completely empty.")
        if not hit:
            warn.append(f"{scorefile.name}: not a single row hit. "
                        f"Example there: {list(idx)[:3]}, "
                        f"in registry.csv: {sorted(base_ids)[:3]}")
        elif orphan:
            warn.append(f"{scorefile.name}: {len(orphan)} IDs without partner, "
                        f"e.g. {orphan[:3]}. Was the registry rewritten in "
                        "the meantime?")

    if report:
        if warn:
            print("\n  Check:")
            for w in warn:
                print(f"    ! {w}")
        else:
            print("  all source files joined cleanly.")
        print()
    return base


def find_duplicates(rows: list[dict], previous_registries=()) -> dict[str, str]:
    """{seq_id: original_seq_id} for everything that already exists.

    Across rounds: over a few hundred sequences exact
    repeats reliably occur, and a duplicate must not cost any GPU time.
    """
    seen = {}
    for prev in previous_registries:
        for r in read_csv(prev):
            seen.setdefault(r.get("seq_sha8") or seq_sha8(r["binder_seq"]), r["seq_id"])
    dups = {}
    for r in rows:
        sha = r.get("seq_sha8") or seq_sha8(r["binder_seq"])
        if sha in seen:
            dups[r["seq_id"]] = seen[sha]
        else:
            seen[sha] = r["seq_id"]
    return dups


# --------------------------------------------------------------------------
# Processes, files, provenance
# --------------------------------------------------------------------------

def run_in_env(python_bin, script, args, cwd=None, dry_run=False) -> int:
    """Start a stage in another conda env.

    Always with a return-value check: a silently failed
    subprocess is the most expensive class of error in this pipeline.
    """
    import subprocess
    cmd = [str(python_bin), str(script), *[str(a) for a in args]]
    print("  $", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    rc = subprocess.run(cmd, cwd=cwd).returncode
    if rc != 0:
        raise RuntimeError(f"Stage failed (exit {rc}): {' '.join(cmd)}")
    return rc


def archive(path, archive_dir):
    """Move the old state away instead of leaving it next to it.

    Old runs are the comparison group and must not be deleted, but
    also must not sit in a directory that a stage searches.
    """
    path = Path(path)
    if not path.exists():
        return None
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / f"{path.name}_{datetime.now():%Y%m%d_%H%M%S}"
    shutil.move(str(path), str(dest))
    print(f"  archived: {path.name} -> {dest}")
    return dest


def fresh_dir(path, archive_dir=None) -> Path:
    """Empty input directory. boltz predict takes a directory and
    uses everything in it - including leftovers of earlier runs."""
    path = Path(path)
    if path.exists():
        if archive_dir:
            archive(path, archive_dir)
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_stage(loop_dir, stage: str, info: dict) -> None:
    """One JSON line per stage run into provenance.jsonl. Later answers
    'which template was this run with?' without anyone having to reconstruct it."""
    loop_dir = Path(loop_dir)
    loop_dir.mkdir(parents=True, exist_ok=True)
    rec = {"stage": stage, "when": now_iso(), "argv": sys.argv, **info}
    with (loop_dir / "provenance.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    assert seq_sha8("ACDEF") == seq_sha8("acdef ")
    assert make_seq_id("L1", 42) == "L1-0042"
    assert identity_to_wt("ABCDE", "ABCDE") == 1.0
    assert identity_to_wt("ABCDE", "ABCDX") == 0.8
    assert stratum({"cell": "T2", "helicity_level": "high"},
                   ["cell", "helicity_level"]) == "T2|high"
    assert find_helicity_col([{"cell": "T1", "helicity_level": "low"}]) == "helicity_level"
    assert find_helicity_col([{"cell": "T1"}]) is None
    assert tier_of("tier_T1_high") == "T1"
    assert tier_of("T0_native") == "T0"
    print("Self-test ok, loaded third-party modules:",
          [m for m in sys.modules if m in ("numpy", "pandas", "gemmi")] or "none")
