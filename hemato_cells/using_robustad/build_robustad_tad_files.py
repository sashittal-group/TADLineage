#!/usr/bin/env python3
"""
Materialize RobusTAD TAD calls as hemato-style tads.tad files.

Reads robustad_predictions/<NODE>/chr<C>/<res>/<NODE>_chr<C>_<res>_tads_all_with_bins.tsv
(start_bin/end_bin columns) and writes using_robustad/<NODE>/chr<C>/<res>/tads.tad
in the same two-column "start<TAB>end" format used by hemato_cells/<NODE>/.../tads.tad,
so the existing check_intermediate_*.py / tad_utilities.read_tad_intervals machinery
can read it unchanged.

Usage (from anywhere):
    python hemato_cells/using_robustad/build_robustad_tad_files.py
    python hemato_cells/using_robustad/build_robustad_tad_files.py --tad-kind hq
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ROBUSTAD_ROOT = PROJECT_ROOT / "robustad_predictions"

NODES = ["CLP", "CMP", "G", "GMP", "LT", "MEP", "MK", "MKP", "MPP"]
CHRS = [str(i) for i in range(1, 20)]
RESOLUTIONS = ["100kb", "250kb", "500kb", "1mb", "2mb"]


def source_path(node: str, chrom: str, res: str, tad_kind: str) -> Path:
    chrom_label = f"chr{chrom}"
    return (
        ROBUSTAD_ROOT
        / node
        / chrom_label
        / res
        / f"{node}_{chrom_label}_{res}_tads_{tad_kind}_with_bins.tsv"
    )


def dest_path(node: str, chrom: str, res: str) -> Path:
    return SCRIPT_DIR / node / f"chr{chrom}" / res / "tads.tad"


def convert_one(node: str, chrom: str, res: str, tad_kind: str) -> bool:
    src = source_path(node, chrom, res, tad_kind)
    if not src.exists():
        return False

    intervals = set()
    with src.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"start_bin", "end_bin"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{src} missing columns: {sorted(missing)}")

        for row in reader:
            try:
                a = int(float(row["start_bin"]))
                b = int(float(row["end_bin"]))
            except (TypeError, ValueError):
                continue
            if a >= b:
                continue
            intervals.add((a, b))

    dst = dest_path(node, chrom, res)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as fh:
        for a, b in sorted(intervals):
            fh.write(f"{a}\t{b}\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize robustad_predictions TAD calls as hemato-style tads.tad files."
    )
    parser.add_argument(
        "--tad-kind",
        choices=["all", "hq"],
        default="all",
        help="Which RobusTAD call set to use as the base TAD structure (default: all).",
    )
    args = parser.parse_args()

    written = 0
    skipped = []
    for node in NODES:
        for chrom in CHRS:
            for res in RESOLUTIONS:
                if convert_one(node, chrom, res, args.tad_kind):
                    written += 1
                else:
                    skipped.append(f"{node}/chr{chrom}/{res}")

    print(f"Wrote {written} tads.tad files under {SCRIPT_DIR}")
    if skipped:
        print(f"Skipped {len(skipped)} missing node/chrom/resolution combos, e.g.: {skipped[:5]}")


if __name__ == "__main__":
    main()
