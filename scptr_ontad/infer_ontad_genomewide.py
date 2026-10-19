#!/usr/bin/env python3
"""Infer OnTAD calls for all requested chromosomes, then optionally merge them."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from merge_ontad_genomewide import (
    infer_one_cell_genomewide,
    load_chrom_layout,
    resolution_to_bp,
)


DEFAULT_BASE = Path("../scptr/cell_differentiation")
DEFAULT_ONTAD = Path("../OnTAD/OnTAD/src/OnTAD")
DEFAULT_GENOME_SIZES = Path("../genome.chrom.sizes")
STAGES = ["1-cell", "2-cell", "4-cell", "8-cell", "64-cell"]
CHROMS = [f"chr{i}" for i in range(1, 20)] + ["chrX"]


def normalize_chrom(value: str) -> str:
    value = str(value)
    return value if value.startswith("chr") else f"chr{value}"


def parse_contact_file(contact_file: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(
            contact_file,
            sep="\t",
            header=None,
            dtype={1: str, 2: int, 4: str, 5: int},
            usecols=[1, 2, 4, 5],
        )
    except Exception as exc:
        print(f"  Error reading {contact_file}: {exc}", flush=True)
        return None

    df.columns = ["chr1", "pos1", "chr2", "pos2"]
    df["chr1"] = df["chr1"].map(normalize_chrom)
    df["chr2"] = df["chr2"].map(normalize_chrom)
    df = df[df["chr1"].isin(CHROMS) & df["chr2"].isin(CHROMS)]
    return df if len(df) else None


def build_chrom_matrix(contacts: pd.DataFrame, chrom: str, resolution_bp: int) -> np.ndarray | None:
    cis = contacts[(contacts["chr1"] == chrom) & (contacts["chr2"] == chrom)].copy()
    if cis.empty:
        return None

    cis["bin1"] = cis["pos1"] // resolution_bp
    cis["bin2"] = cis["pos2"] // resolution_bp
    binned = cis.groupby(["bin1", "bin2"]).size().reset_index(name="count")
    max_bin = int(max(binned["bin1"].max(), binned["bin2"].max())) + 1
    matrix = np.zeros((max_bin, max_bin), dtype=np.float32)

    for row in binned.itertuples(index=False):
        i = int(row.bin1)
        j = int(row.bin2)
        count = float(row.count)
        matrix[i, j] += count
        if i != j:
            matrix[j, i] += count

    return matrix


def run_ontad(ontad_bin: Path, matrix_file: Path, output_prefix: Path, penalty: float, timeout: int) -> bool:
    cleaned_file = matrix_file.with_suffix(matrix_file.suffix + ".cleaned")
    with open(matrix_file) as src, open(cleaned_file, "w") as dst:
        for line in src:
            stripped = " ".join(line.split())
            if stripped:
                dst.write(stripped + "\n")

    cmd = [str(ontad_bin), str(cleaned_file), "-o", str(output_prefix), "-penalty", str(penalty)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"    OnTAD failed for {output_prefix}: {result.stderr[:500]}", flush=True)
        return False

    try:
        os.remove(matrix_file)
        os.remove(cleaned_file)
    except OSError:
        pass
    return True


def process_cell(
    contact_file: Path,
    stage: str,
    base_dir: Path,
    ontad_bin: Path,
    resolutions: list[str],
    chromosomes: list[str],
    penalty: float,
    timeout: int,
    overwrite: bool,
) -> int:
    gsm = contact_file.name.split("_")[0]
    print(f"Processing {stage}/{gsm}", flush=True)
    contacts = parse_contact_file(contact_file)
    if contacts is None:
        return 0

    written = 0
    for resolution in resolutions:
        resolution_bp = resolution_to_bp(resolution)
        for chrom in chromosomes:
            out_dir = base_dir / stage / gsm / chrom / resolution
            tad_file = out_dir / "tads.tad"
            if tad_file.exists() and not overwrite:
                continue

            matrix = build_chrom_matrix(contacts, chrom, resolution_bp)
            if matrix is None:
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            matrix_file = out_dir / "hic_matrix.txt"
            np.savetxt(matrix_file, matrix, fmt="%d", delimiter=" ")

            if run_ontad(ontad_bin, matrix_file, out_dir / "tads", penalty, timeout):
                written += int(tad_file.exists())
                print(f"  {chrom}/{resolution}: wrote {tad_file}", flush=True)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--ontad-bin", type=Path, default=DEFAULT_ONTAD)
    parser.add_argument("--genome-sizes", type=Path, default=DEFAULT_GENOME_SIZES)
    parser.add_argument("--resolutions", nargs="+", default=["1000kb"])
    parser.add_argument("--chromosomes", nargs="+", default=CHROMS)
    parser.add_argument("--stages", nargs="+", default=STAGES)
    parser.add_argument("--penalty", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--merge-after", action="store_true")
    parser.add_argument("--merge-output-base", type=Path, default=Path("results_updated/scptr_merged_all_chr"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ontad_bin.exists():
        raise SystemExit(f"OnTAD binary not found: {args.ontad_bin}")

    chromosomes = [normalize_chrom(chrom) for chrom in args.chromosomes]
    inferred = 0
    for stage in args.stages:
        stage_dir = args.base_dir / stage
        if not stage_dir.exists():
            continue
        for contact_file in sorted(stage_dir.glob("*.txt")):
            inferred += process_cell(
                contact_file,
                stage,
                args.base_dir,
                args.ontad_bin,
                args.resolutions,
                chromosomes,
                args.penalty,
                args.timeout,
                args.overwrite,
            )

    print(f"OnTAD files inferred: {inferred}", flush=True)

    if args.merge_after:
        for resolution in args.resolutions:
            offsets, matrix_size = load_chrom_layout(args.genome_sizes, resolution_to_bp(resolution))
            genomewide = 0
            for stage in args.stages:
                stage_dir = args.base_dir / stage
                if not stage_dir.exists():
                    continue
                for contact_file in sorted(stage_dir.glob("*.txt")):
                    if contact_file.name.startswith("GSM"):
                        genomewide += infer_one_cell_genomewide(
                            contact_file,
                            stage,
                            args.merge_output_base,
                            resolution,
                            offsets,
                            matrix_size,
                            args.ontad_bin,
                            args.penalty,
                            args.timeout,
                            keep_matrix=False,
                        )
            complete = args.merge_output_base / resolution / "_complete.txt"
            complete.parent.mkdir(parents=True, exist_ok=True)
            complete.write_text(f"resolution\t{resolution}\nsamples_written\t{genomewide}\n")
            print(f"Genomewide full-matrix OnTAD wrote {genomewide} samples for {resolution}", flush=True)


if __name__ == "__main__":
    main()
