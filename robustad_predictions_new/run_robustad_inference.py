#!/usr/bin/env python3
"""Run RobusTAD LMCC boundary refinement and TAD assembly for one sample/chromosome."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cooler
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--resolution", type=int, default=5000)
    parser.add_argument("--resolution-label", default="5kb")
    parser.add_argument("--mcool-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--robustad-root", type=Path, required=True)
    parser.add_argument("--mplconfigdir", type=Path, required=True)
    parser.add_argument("--minW", type=int, default=50, help="RobusTAD min window in kb.")
    parser.add_argument("--maxW", type=int, default=600, help="RobusTAD max window in kb.")
    parser.add_argument("--mind", type=int, default=50, help="Minimum boundary distance in kb.")
    parser.add_argument("--ratio", type=float, default=1.2)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--mintad", type=int, default=50_000, help="Minimum TAD size in bp.")
    parser.add_argument("--maxtad", type=int, default=3_000_000, help="Maximum TAD size in bp.")
    parser.add_argument("--delta", type=float, default=0.1)
    return parser.parse_args()


def robustad_env(robustad_root: Path, mplconfigdir: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{robustad_root}{os.pathsep}{existing}" if existing else str(robustad_root)
    )
    mplconfigdir.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mplconfigdir)
    return env


def robustad_cmd() -> list[str]:
    return [sys.executable, "-m", "robustad.robustad"]


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def normalize_boundary_chrom(boundary_file: Path, chrom_label: str) -> None:
    df = pd.read_csv(boundary_file, sep="\t", header=None)
    df[0] = chrom_label
    df.to_csv(boundary_file, sep="\t", header=False, index=False)


def resolve_chrom_label(mcool_file: Path, resolution: int, requested: str) -> str:
    clr = cooler.Cooler(f"{mcool_file}::/resolutions/{resolution}")
    chromnames = set(clr.chromnames)
    candidates = [requested]
    if requested.startswith("chr"):
        candidates.append(requested[3:])
    else:
        candidates.append(f"chr{requested}")

    for candidate in candidates:
        if candidate in chromnames:
            return candidate
    raise ValueError(
        f"Could not find chromosome {requested!r} at {resolution} bp in {mcool_file}. "
        f"Available examples: {sorted(chromnames)[:8]}"
    )


def main() -> None:
    args = parse_args()
    mcool_file = args.mcool_dir / f"GSE152918_{args.sample}.mcool"
    if not mcool_file.exists():
        raise FileNotFoundError(
            f"Missing input for sample {args.sample}: {mcool_file}. "
            "Remove this sample from SAMPLES or add the .mcool file."
        )
    if not args.panel.exists():
        raise FileNotFoundError(f"Missing RobusTAD reference panel: {args.panel}")

    chrom_label = resolve_chrom_label(mcool_file, args.resolution, args.chrom)
    outdir = args.output_root / args.sample / args.chrom / args.resolution_label
    outdir.mkdir(parents=True, exist_ok=True)

    boundary_prefix = outdir / "boundary"
    boundary_file = outdir / "boundary.bed"
    tad_prefix = outdir / "tads"

    env = robustad_env(args.robustad_root, args.mplconfigdir)

    run(
        robustad_cmd()
        + [
            "lmcc",
            "--resol",
            str(args.resolution),
            "--chr",
            chrom_label,
            "--panel",
            str(args.panel),
            "--minW",
            str(args.minW),
            "--maxW",
            str(args.maxW),
            "--mind",
            str(args.mind),
            "--ratio",
            str(args.ratio),
            "--alpha",
            str(args.alpha),
            str(mcool_file),
            str(boundary_prefix),
        ],
        env,
    )
    normalize_boundary_chrom(boundary_file, chrom_label)

    run(
        robustad_cmd()
        + [
            "assembly",
            "--resol",
            str(args.resolution),
            "--chr",
            chrom_label,
            "--mintad",
            str(args.mintad),
            "--maxtad",
            str(args.maxtad),
            "--ratio",
            str(args.ratio),
            "--delta",
            str(args.delta),
            str(mcool_file),
            str(boundary_file),
            str(tad_prefix),
        ],
        env,
    )


if __name__ == "__main__":
    main()
