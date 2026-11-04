#!/usr/bin/env python3
"""
fix_svg_upload.py — make SVGs acceptable to uploaders that sniff the file type
from the first bytes (e.g. the iGEM Registry media upload).

Such uploaders expect the file to begin with "<svg". Files written by matplotlib,
Inkscape, R/ggplot (svglite) and similar tools instead begin with an XML
declaration and often a DOCTYPE:

    <?xml version="1.0" encoding="utf-8" standalone="no"?>
    <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "...svg11.dtd">
    <svg ...>

Dropping that preamble is safe: the XML declaration is optional for UTF-8
documents, and the SVG 1.1 DOCTYPE is explicitly discouraged by the W3C.

Usage
-----
    python3 fix_svg_upload.py FOLDER                # report only (default)
    python3 fix_svg_upload.py FOLDER --write        # fix in place, keep .bak
    python3 fix_svg_upload.py FOLDER --write --no-backup
    python3 fix_svg_upload.py FOLDER --write --out fixed/   # write copies
    python3 fix_svg_upload.py FOLDER -r             # recurse into subfolders
    python3 fix_svg_upload.py FOLDER --strip-metadata

Safety
------
* Every file is parsed with an XML parser before and after the change; if the
  result does not parse, or the root element is not <svg>, the file is left
  untouched and reported as SKIPPED.
* Nothing inside <svg>...</svg> is modified (unless --strip-metadata is given,
  which removes only the <metadata> block: author/date/licence records that
  carry no visual information).
* Without --write the script changes nothing at all.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SVG_ROOT = "{http://www.w3.org/2000/svg}svg"
METADATA_RE = re.compile(r"[ \t]*<metadata>.*?</metadata>[ \t]*\n?", re.DOTALL)


def read_text(path: Path) -> tuple[str, bool]:
    """Return (text, had_bom). A UTF-8 BOM also breaks sniffing, so it is detected
    from the raw bytes rather than from the decoded string."""
    raw = path.read_bytes()
    had_bom = raw.startswith(b"\xef\xbb\xbf")
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(enc), had_bom
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", raw, 0, 1, "undecodable")


def parses_as_svg(text: str) -> bool:
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return False
    return root.tag in (SVG_ROOT, "svg")


def transform(text: str, strip_metadata: bool, had_bom: bool = False) -> tuple[str, list[str]]:
    """Return (new_text, list_of_changes_applied)."""
    changes: list[str] = []
    new = text.lstrip("﻿")
    if had_bom or new != text:
        changes.append("BOM")

    idx = new.find("<svg")
    if idx == -1:
        return text, []  # not an SVG we understand; caller skips it
    if idx > 0:
        preamble = new[:idx]
        new = new[idx:]
        if "<?xml" in preamble:
            changes.append("XML declaration")
        if "<!DOCTYPE" in preamble:
            changes.append("DOCTYPE")
        other = re.sub(r"<\?xml.*?\?>|<!DOCTYPE.*?>|<!--.*?-->", "", preamble, flags=re.DOTALL)
        if other.strip():
            changes.append("other leading content")

    if strip_metadata:
        stripped, n = METADATA_RE.subn("", new)
        if n:
            new = stripped
            changes.append(f"{n} <metadata> block{'s' if n > 1 else ''}")

    return new, changes


def process(path: Path, args: argparse.Namespace) -> str:
    """Returns a status string: OK / FIXED / WOULD-FIX / SKIPPED."""
    try:
        text, had_bom = read_text(path)
    except Exception as exc:  # unreadable / undecodable
        print(f"  SKIPPED  {path.name}: cannot read ({exc})")
        return "SKIPPED"

    if not parses_as_svg(text):
        print(f"  SKIPPED  {path.name}: does not parse as SVG (left untouched)")
        return "SKIPPED"

    new, changes = transform(text, args.strip_metadata, had_bom)
    if not changes:
        print(f"  OK       {path.name}: already starts with <svg>")
        return "OK"

    if not parses_as_svg(new):
        print(f"  SKIPPED  {path.name}: result would not parse (left untouched)")
        return "SKIPPED"

    detail = ", ".join(changes)
    if not args.write:
        print(f"  WOULD FIX {path.name}: remove {detail}")
        return "WOULD-FIX"

    if args.out:
        target = args.out / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        target = path
        if not args.no_backup:
            backup = path.with_suffix(path.suffix + ".bak")
            if backup.exists():
                print(f"  SKIPPED  {path.name}: {backup.name} already exists")
                return "SKIPPED"
            backup.write_bytes(path.read_bytes())

    target.write_text(new, encoding="utf-8")
    where = f" -> {target}" if args.out else ""
    print(f"  FIXED    {path.name}: removed {detail}{where}")
    return "FIXED"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="folder containing .svg files")
    ap.add_argument("--write", action="store_true", help="actually modify files (default: report only)")
    ap.add_argument("--no-backup", action="store_true", help="do not keep a .bak copy when fixing in place")
    ap.add_argument("--out", type=Path, default=None, help="write fixed copies here instead of in place")
    ap.add_argument("-r", "--recursive", action="store_true", help="also search subfolders")
    ap.add_argument("--strip-metadata", action="store_true",
                    help="also remove the <metadata> RDF block (author/date/licence, no visual effect)")
    args = ap.parse_args()

    if not args.folder.is_dir():
        print(f"error: {args.folder} is not a folder", file=sys.stderr)
        return 2

    pattern = "**/*.svg" if args.recursive else "*.svg"
    files = sorted(p for p in args.folder.glob(pattern) if p.is_file())
    if not files:
        print(f"No .svg files found in {args.folder}")
        return 0

    print(f"{len(files)} SVG file(s) in {args.folder}"
          f"{'' if args.write else '  [dry run — nothing will be modified]'}\n")

    counts: dict[str, int] = {}
    for f in files:
        status = process(f, args)
        counts[status] = counts.get(status, 0) + 1

    print("\n" + ", ".join(f"{n} {status.lower()}" for status, n in sorted(counts.items())))
    if counts.get("WOULD-FIX"):
        print("Re-run with --write to apply these changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
