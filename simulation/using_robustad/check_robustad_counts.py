#!/usr/bin/env python3
"""Check that RobusTAD assembled enough TADs at scored internal nodes.

RobusTAD analogue of ``using_ontad/check_ontad_counts.py``. Counts rows in each
node's ``robustad/<node>_tads_all_with_bins.tsv`` and fails if a scored internal
node (A/B/C) has fewer than ``--min-tads`` calls.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SIM_ROOT = SCRIPT_DIR.parent          # simulation/ (shared config.py + data/)
sys.path.insert(0, str(SIM_ROOT))

import config


def count_called_tads(path: Path) -> int:
    count = 0
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            try:
                a = int(float(row["start_bin"]))
                b = int(float(row["end_bin"]))
            except (TypeError, ValueError, KeyError):
                continue
            if a < b:
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=SIM_ROOT / "data")
    parser.add_argument("--tad-kind", choices=["all", "hq"], default="all")
    parser.add_argument("--min-tads", type=int, default=20)
    args = parser.parse_args()

    failures = []
    for region_dir in sorted(args.outdir.glob("region_*")):
        for node in config.SCORE_NODES:
            tad_path = region_dir / "robustad" / f"{node}_tads_{args.tad_kind}_with_bins.tsv"
            if not tad_path.exists():
                failures.append((region_dir.name, node, "missing"))
                continue
            n_tads = count_called_tads(tad_path)
            if n_tads < args.min_tads:
                failures.append((region_dir.name, node, n_tads))

    if failures:
        examples = ", ".join(f"{region}:{node}={count}" for region, node, count in failures[:10])
        raise SystemExit(
            f"RobusTAD call-count check failed for {len(failures)} node(s); "
            f"required >= {args.min_tads}. Examples: {examples}"
        )

    print(f"All scored internal nodes have at least {args.min_tads} RobusTAD-called TADs.")


if __name__ == "__main__":
    main()
