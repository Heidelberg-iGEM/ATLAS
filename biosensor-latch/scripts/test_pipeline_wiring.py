#!/usr/bin/env python3
"""Self-check for run_cage_pipeline.py's orchestration glue.

Runs WITHOUT FlexCraft, foundry, AF2, a GPU or any model weights -- it only
exercises the stdlib wiring: argument validation, the step-6 gate subset, and
the shape of the commands that get shelled out. Nothing is executed.

    python3 scripts/test_pipeline_wiring.py

Catches a broken edit or a bad script extraction before you queue a cluster
job. It does NOT check that the pipeline produces sensible designs.
"""

from __future__ import annotations

import csv
import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cage_pipeline as rcp  # noqa: E402

BASE_ARGV = [
    "--target_count", "1", "--run_name", "t/run01",
    "--foundry_dir", "/tmp/foundry", "--rfd3_yaml", "x.yaml",
    "--mpnn_checkpoint", "/tmp/mpnn.pt", "--redesign_terminus", "c",
    "--fixed_end_sequence", "AAAA",
    "--gfp_binder_pdb", "complex.pdb", "--gfp_binder_chain", "B", "--gfp_chain", "A",
    "--hotspots", "10,14",
]

PARTNER_ARGV = [
    "--run_partners",
    "--partner1_sequence", "MGKV", "--partner1_target", "gfp",
    "--partner1_hotspots", "201",
]


def parse(extra: list[str]):
    return rcp.build_arg_parser().parse_args(BASE_ARGV + extra)


def expect_exit(extra: list[str], needle: str) -> None:
    """validate() must reject this combination, with a message mentioning `needle`."""
    try:
        with redirect_stdout(io.StringIO()):
            rcp.validate(parse(extra))
    except SystemExit as exit_error:
        message = str(exit_error)
        assert needle in message, f"wrong rejection for {extra}: {message!r} lacks {needle!r}"
        return
    raise AssertionError(f"validate() accepted {extra} but should have refused it")


def test_defaults_unchanged() -> None:
    """Without --run_partners the pipeline must look exactly like the 5-step version."""
    args = parse([])
    rcp.validate(args)
    assert args.run_partners is False
    assert args.partners_required is False
    assert args.partner1_sequence is None


def test_partner_validation() -> None:
    rcp.validate(parse(PARTNER_ARGV))  # the complete, valid case

    expect_exit(["--partners_required"], "--run_partners")
    expect_exit(["--run_partners"], "--partner1_sequence")
    expect_exit(["--run_partners", "--partner1_sequence", "MGKV"], "--partner1_target")
    expect_exit(
        ["--run_partners", "--partner1_sequence", "MGKV", "--partner1_target", "gfp"],
        "--partner1_hotspots",
    )
    # Gating on a profile with no partner-metric thresholds would pass everything.
    expect_exit(PARTNER_ARGV + ["--partners_required"], "every construct would pass")
    rcp.validate(parse(PARTNER_ARGV + ["--partners_required",
                                       "--partners_filter_json", "filters/partners_test3_target_bound.json"]))
    # A second partner needs its own target + hotspots, and a distinct chain.
    expect_exit(PARTNER_ARGV + ["--partner2_sequence", "MGKV"], "--partner2_target")
    expect_exit(PARTNER_ARGV + ["--partner2_sequence", "MGKV", "--partner2_target", "binder"],
                "--partner2_hotspots")
    expect_exit(PARTNER_ARGV + ["--partner2_sequence", "MGKV", "--partner2_target", "binder",
                                "--partner2_hotspots", "14", "--partner2_chain", "G"], "must differ")


def test_gate_subset_keeps_columns_and_selects_rows() -> None:
    """Step 6 must see only the winners, with every monomer column intact."""
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "score_all.csv"
        fields = ["task_id", "status", "redesign_terminus", "binder_start_residue", "predicted_pdb"]
        with source.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for i in range(4):
                writer.writerow({"task_id": f"t{i}", "status": "success", "redesign_terminus": "c",
                                 "binder_start_residue": 67, "predicted_pdb": f"/p/t{i}.pdb"})

        out = Path(tmp) / "partners_gate.csv"
        kept = rcp.write_gate_subset(source, {"t1", "t3"}, out)

        assert kept == 2, kept
        written_fields, rows = rcp.read_csv_rows(out)
        assert written_fields == fields, written_fields
        assert [r["task_id"] for r in rows] == ["t1", "t3"]
        assert rows[0]["binder_start_residue"] == "67"
        # A task_id that passed nothing must not leak in.
        assert rcp.write_gate_subset(source, set(), Path(tmp) / "empty.csv") == 0


def test_partner_command_shape() -> None:
    """The shelled-out command must carry both partners and the subset gate."""
    args = parse(PARTNER_ARGV + [
        "--partner1_label", "GFP", "--partner1_chain", "G",
        "--partner2_sequence", "MSTV", "--partner2_target", "binder",
        "--partner2_hotspots", "14,18", "--partner2_label", "GvpC",
        "--partners_filter_json", "filters/partners_test3_target_bound.json",
        "--partners_seed", "41",
    ])
    args.dry_run = True
    rcp.validate(args)

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        rcp.run_partners(args, "batch_000", Path("/tmp/b"), Path("/tmp/b/partners_gate.csv"),
                         Path("/repo"), Path("/repo/results"))
    cmd = buffer.getvalue()

    assert "score_gfp_fusion_partners" in cmd
    assert "--gate_status_filter all" in cmd, "the subset CSV is already the selection"
    assert "--partner1_sequence MGKV" in cmd and "--partner1_target gfp" in cmd
    assert "--partner1_hotspots 201" in cmd and "--partner1_label GFP" in cmd
    assert "--partner2_sequence MSTV" in cmd and "--partner2_label GvpC" in cmd
    assert "--seed 41" in cmd
    assert "--filter_profile custom_json" in cmd

    # One partner only (Test 3) must not emit partner2 flags at all.
    solo = parse(PARTNER_ARGV)
    solo.dry_run = True
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        rcp.run_partners(solo, "batch_000", Path("/tmp/b"), Path("/tmp/b/g.csv"),
                         Path("/repo"), Path("/repo/results"))
    solo_cmd = buffer.getvalue()
    assert "--partner2" not in solo_cmd, solo_cmd
    # No filter JSON -> must warn that 'success' is meaningless.
    assert "WARNING" in solo_cmd


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} wiring checks passed.")


if __name__ == "__main__":
    main()
