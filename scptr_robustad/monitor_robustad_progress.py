#!/usr/bin/env python3
"""
Monitor RobusTAD progress for the GSE129029 all-cells run.

This script does not depend on the worker process writing logs. It infers
progress directly from files created under scptr/robustad_scptr_data.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path("/Users/PSLabMonisha/Desktop/tads")
SCPTR_ROOT = PROJECT_ROOT / "scptr"
METADATA_FILE = SCPTR_ROOT / "GSE129029_metadata.csv"
OUTPUT_ROOT_BASE = SCPTR_ROOT / "robustad_scptr_data"
DEFAULT_SUBSET_FILE = SCPTR_ROOT / "robustad_subset_10_per_stage.csv"
STAGES = ["1-cell", "2-cell", "4-cell", "8-cell", "64-cell"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live progress monitor for RobusTAD all-cells run.")
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Refresh interval in seconds. Default: 5",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=32,
        help="Progress bar width. Default: 32",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one snapshot and exit.",
    )
    parser.add_argument(
        "--targets-file",
        type=Path,
        default=None,
        help="Optional CSV/TSV file listing the exact GSM subset being processed.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=250,
        help="Resolution label in kb under robustad_scptr_data. Default: 250",
    )
    return parser.parse_args()


def output_root(resolution_kb: int) -> Path:
    return OUTPUT_ROOT_BASE / str(resolution_kb)


def load_expected_counts(targets_file: Path | None) -> tuple[int, Counter]:
    if targets_file is None and DEFAULT_SUBSET_FILE.exists():
        targets_file = DEFAULT_SUBSET_FILE
    with METADATA_FILE.open() as handle:
        rows = list(csv.DictReader(handle))
    if targets_file is not None:
        with targets_file.open() as handle:
            reader = csv.DictReader(handle)
            target_gsms = {row["GSM"] for row in reader if row.get("GSM")}
        rows = [row for row in rows if row["GSM"] in target_gsms]
    by_stage = Counter(row["CELL_STAGE"] for row in rows)
    return len(rows), by_stage


def collect_stage_stats(stage: str, root: Path) -> dict[str, int]:
    stage_dir = root / stage
    if not stage_dir.exists():
        return {
            "mcool": 0,
            "boundaries": 0,
            "tads": 0,
            "panels": 0,
        }

    return {
        "mcool": len(list(stage_dir.glob("*/" + "*.mcool"))),
        "boundaries": len(list(stage_dir.glob("*/robustad_boundaries.bed"))),
        "tads": len(list(stage_dir.glob("*/robustad_tads_all.bed"))),
        "panels": len(list(stage_dir.glob("robustad_reference_panel_*.pkl"))),
    }


def load_summary_stats(root: Path) -> tuple[int, int]:
    summary = root / "run_summary.tsv"
    if not summary.exists():
        return 0, 0

    ok = 0
    failed = 0
    with summary.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["status"] == "ok":
                ok += 1
            elif row["status"] == "failed":
                failed += 1
    return ok, failed


def bar(done: int, total: int, width: int) -> str:
    total = max(total, 1)
    filled = min(width, int(width * done / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def clear_screen() -> None:
    print("\033[2J\033[H", end="")


def render(width: int, targets_file: Path | None, resolution_kb: int) -> str:
    root = output_root(resolution_kb)
    total_expected, expected_by_stage = load_expected_counts(targets_file)
    effective_targets = targets_file if targets_file is not None else (
        DEFAULT_SUBSET_FILE if DEFAULT_SUBSET_FILE.exists() else None
    )
    ok, failed = load_summary_stats(root)

    total_tads = 0
    total_mcool = 0
    total_boundaries = 0
    panel_count = 0
    stage_lines = []

    for stage in STAGES:
        stats = collect_stage_stats(stage, root)
        expected = expected_by_stage[stage]
        total_tads += stats["tads"]
        total_mcool += stats["mcool"]
        total_boundaries += stats["boundaries"]
        panel_count += stats["panels"]
        stage_lines.append(
            f"{stage:7} {bar(stats['tads'], expected, width)} "
            f"{stats['tads']:>3}/{expected:<3} "
            f"mcool={stats['mcool']:<3} boundaries={stats['boundaries']:<3}"
        )

    lines = [
        "RobusTAD Progress",
        "",
        f"Resolution          {resolution_kb} kb",
        f"Completed TAD calls   {bar(total_tads, total_expected, width)} {total_tads}/{total_expected}",
        f"Summary OK / failed   {ok}/{failed}",
        f"mcool / boundaries    {total_mcool}/{total_boundaries}",
        f"Stage panels created  {panel_count}/5",
        "",
        "Per stage:",
        *stage_lines,
        "",
        f"Targets file: {effective_targets if effective_targets else 'all metadata rows'}",
        f"Output root: {root}",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    while True:
        clear_screen()
        print(render(args.width, args.targets_file, args.resolution), flush=True)
        if args.once:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
