#!/usr/bin/env python3
"""Check that OnTAD produced enough TADs at scored internal nodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SIM_ROOT = SCRIPT_DIR.parent          # simulation/ (shared generation + data)
sys.path.insert(0, str(SIM_ROOT))     # shared config.py lives here

import config


def count_called_tads(path: Path) -> int:
    count = 0
    with path.open() as handle:
        for line_no, line in enumerate(handle):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            level = int(fields[2]) if len(fields) >= 3 and fields[2].lstrip("-").isdigit() else None
            if line_no == 0 and level == 0:
                continue
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=SIM_ROOT / "data")
    parser.add_argument("--min-tads", type=int, default=20)
    args = parser.parse_args()

    failures = []
    for region_dir in sorted(args.outdir.glob("region_*")):
        for node in config.SCORE_NODES:
            tad_path = region_dir / "ontad" / f"{node}.tad"
            if not tad_path.exists():
                failures.append((region_dir.name, node, "missing"))
                continue
            n_tads = count_called_tads(tad_path)
            if n_tads < args.min_tads:
                failures.append((region_dir.name, node, n_tads))

    if failures:
        examples = ", ".join(f"{region}:{node}={count}" for region, node, count in failures[:10])
        raise SystemExit(
            f"OnTAD call-count check failed for {len(failures)} node(s); "
            f"required >= {args.min_tads}. Examples: {examples}"
        )

    print(f"All scored internal nodes have at least {args.min_tads} OnTAD-called TADs.")


if __name__ == "__main__":
    main()
