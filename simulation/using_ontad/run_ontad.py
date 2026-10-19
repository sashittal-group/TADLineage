#!/usr/bin/env python3
"""Run OnTAD on simulated matrices to produce the *called* hierarchies.

These OnTAD calls are the observed TAD hierarchies used by the experiment:
  * internal nodes (A, B, C)      -> evaluation ground truth
  * root + leaves (D, E, F, G)    -> reconstruction inputs

Reads the matrices written by generate_data.py and writes, per region:
  region_XX/ontad/<node>.tad      OnTAD nested-TAD calls (start end level ...)

By default this runs all regions and nodes. Snakemake calls it with
--region-index and --node so each job owns exactly one output file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SIM_ROOT = SCRIPT_DIR.parent          # simulation/ (shared generation + data)
REPO_ROOT = SIM_ROOT.parent           # repo root (holds OnTAD/)
sys.path.insert(0, str(SIM_ROOT))     # shared config.py lives here

import config

DEFAULT_ONTAD = REPO_ROOT / "OnTAD" / "OnTAD" / "src" / "OnTAD"


def run_ontad(ontad_bin: Path, matrix_file: Path, out_prefix: Path,
              penalty: float, maxsz: int, minsz: int, lsize: int,
              ldiff: float, use_log2: bool, timeout: int) -> bool:
    """Call OnTAD on one matrix; writes ``<out_prefix>.tad``."""
    # OnTAD wants single-space-collapsed rows.
    cleaned = matrix_file.with_suffix(matrix_file.suffix + ".cleaned")
    with open(matrix_file) as src, open(cleaned, "w") as dst:
        for line in src:
            stripped = " ".join(line.split())
            if stripped:
                dst.write(stripped + "\n")

    cmd = [
        str(ontad_bin), str(cleaned),
        "-o", str(out_prefix),
        "-penalty", str(penalty),
        "-maxsz", str(maxsz),
        "-minsz", str(minsz),
        "-lsize", str(lsize),
        "-ldiff", str(ldiff),
    ]
    if use_log2:
        cmd.append("-log2")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    cleaned.unlink(missing_ok=True)
    if result.returncode != 0:
        print(f"  OnTAD failed for {out_prefix}: {result.stderr[:300]}", flush=True)
        return False
    if not out_prefix.with_suffix(".tad").exists():
        print(f"  OnTAD did not write expected output: {out_prefix}.tad", flush=True)
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=SIM_ROOT / "data")
    p.add_argument("--ontad-bin", type=Path, default=DEFAULT_ONTAD)
    p.add_argument("--region-index", type=str, default=None,
                   help="run exactly this region index/name, e.g. 7, 007, or region_007")
    p.add_argument("--node", choices=config.NODE_ORDER, default=None,
                   help="run exactly this node within the selected region(s)")
    p.add_argument("--penalty", type=float, default=0.1)
    p.add_argument("--maxsz", type=int, default=200)
    p.add_argument("--minsz", type=int, default=3)
    p.add_argument("--lsize", type=int, default=5)
    p.add_argument("--ldiff", type=float, default=1.96)
    p.add_argument("--log2", action="store_true", help="pass -log2 to OnTAD")
    p.add_argument("--timeout", type=int, default=300)
    args = p.parse_args()

    if not args.ontad_bin.exists():
        raise SystemExit(f"OnTAD binary not found: {args.ontad_bin}")

    if args.region_index is None:
        region_dirs = sorted(args.outdir.glob("region_*"))
    else:
        idx = args.region_index
        if idx.startswith("region_"):
            region_name = idx
        else:
            region_name = f"region_{int(idx):03d}"
        region_dirs = [args.outdir / region_name]

    if not region_dirs:
        raise SystemExit(f"No region_* directories under {args.outdir}; run generate_data.py first")

    missing = [str(rd) for rd in region_dirs if not rd.exists()]
    if missing:
        raise SystemExit(f"Region directory not found: {missing[0]}")

    nodes = [args.node] if args.node else config.NODE_ORDER

    total = 0
    failures = 0
    for region_dir in region_dirs:
        ontad_dir = region_dir / "ontad"
        ontad_dir.mkdir(exist_ok=True)
        for node in nodes:
            matrix_file = region_dir / "matrices" / f"{node}.matrix"
            if not matrix_file.exists():
                raise SystemExit(f"Matrix not found: {matrix_file}")
            out_prefix = ontad_dir / node
            if run_ontad(args.ontad_bin, matrix_file, out_prefix,
                         args.penalty, args.maxsz, args.minsz, args.lsize,
                         args.ldiff, args.log2, args.timeout):
                total += 1
            else:
                failures += 1
        print(f"{region_dir.name}: OnTAD done", flush=True)

    if failures:
        raise SystemExit(f"OnTAD failed for {failures} matrix/matrices")

    print(f"\nOnTAD produced {total} call files under {args.outdir}")


if __name__ == "__main__":
    main()
