#!/usr/bin/env python3
"""Build genomewide OnTAD calls from full-cell Hi-C contact matrices."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


CHROMS = [f"chr{i}" for i in range(1, 20)] + ["chrX"]
STAGES = ["1-cell", "2-cell", "4-cell", "8-cell", "64-cell"]
DEFAULT_ONTAD = Path("../OnTAD/OnTAD/src/OnTAD")


def resolution_to_bp(resolution: str) -> int:
    label = str(resolution).lower()
    if label.endswith("kb"):
        return int(label[:-2]) * 1000
    return int(label)


def load_chrom_offsets(genome_sizes_path: Path, resolution_bp: int) -> dict[str, int]:
    offsets, _ = load_chrom_layout(genome_sizes_path, resolution_bp)
    return offsets


def load_chrom_layout(genome_sizes_path: Path, resolution_bp: int) -> tuple[dict[str, int], int]:
    sizes = {}
    with open(genome_sizes_path) as handle:
        for line in handle:
            cols = line.strip().split()
            if len(cols) >= 2:
                sizes[cols[0]] = int(cols[1])

    offsets = {}
    cursor = 0
    for chrom in CHROMS:
        if chrom not in sizes:
            raise KeyError(f"{chrom} is missing from {genome_sizes_path}")
        offsets[chrom] = cursor
        cursor += int(math.ceil(sizes[chrom] / resolution_bp)) + 1
    return offsets, cursor


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


def build_genomewide_matrix(
    contacts: pd.DataFrame,
    resolution_bp: int,
    offsets: dict[str, int],
    matrix_size: int,
) -> np.ndarray:
    contacts = contacts.copy()
    contacts["bin1"] = contacts["chr1"].map(offsets) + contacts["pos1"] // resolution_bp
    contacts["bin2"] = contacts["chr2"].map(offsets) + contacts["pos2"] // resolution_bp
    binned = contacts.groupby(["bin1", "bin2"]).size().reset_index(name="count")

    if binned.empty:
        return np.zeros((matrix_size, matrix_size), dtype=np.int32)

    observed_size = int(max(binned["bin1"].max(), binned["bin2"].max())) + 1
    matrix_size = max(matrix_size, observed_size)
    matrix = np.zeros((matrix_size, matrix_size), dtype=np.int32)
    for row in binned.itertuples(index=False):
        i = int(row.bin1)
        j = int(row.bin2)
        count = int(row.count)
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
        print(f"  OnTAD failed for {output_prefix}: {result.stderr[:500]}", flush=True)
        return False
    return True


def infer_one_cell_genomewide(
    contact_file: Path,
    stage: str,
    output_base: Path,
    resolution: str,
    offsets: dict[str, int],
    matrix_size: int,
    ontad_bin: Path,
    penalty: float,
    timeout: int,
    keep_matrix: bool,
) -> int:
    sample = contact_file.name.split("_")[0]
    out_dir = output_base / resolution / stage / sample / "all" / resolution
    tad_file = out_dir / "tads.tad"

    contacts = parse_contact_file(contact_file)
    if contacts is None:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_file = out_dir / "genomewide_hic_matrix.txt"
    matrix = build_genomewide_matrix(contacts, resolution_to_bp(resolution), offsets, matrix_size)
    np.savetxt(matrix_file, matrix, fmt="%d", delimiter=" ")

    if not run_ontad(ontad_bin, matrix_file, out_dir / "tads", penalty, timeout):
        return 0

    cleaned_file = matrix_file.with_suffix(matrix_file.suffix + ".cleaned")
    if not keep_matrix:
        for path in (matrix_file, cleaned_file):
            try:
                os.remove(path)
            except OSError:
                pass

    if not tad_file.exists():
        return 0

    meta_path = out_dir / "genomewide_ontad_metadata.tsv"
    with open(meta_path, "w") as handle:
        handle.write("input_scope\tfull_genomewide_hic_matrix\n")
        handle.write(f"source_contact_file\t{contact_file}\n")
        handle.write(f"resolution\t{resolution}\n")
        handle.write(f"chromosomes\t{','.join(CHROMS)}\n")
        handle.write(f"matrix_bins\t{matrix.shape[0]}\n")
        handle.write(f"contact_rows\t{len(contacts)}\n")
        handle.write(f"ontad_penalty\t{penalty}\n")

    return 1


def merge_one_cell(*args, **kwargs) -> int:
    raise RuntimeError(
        "Per-chromosome TAD merging is no longer supported here. "
        "Use infer_one_cell_genomewide to run OnTAD on the full genomewide matrix."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-base",
        type=Path,
        default=Path("../scptr/cell_differentiation"),
        help="Input root with stage/*.txt contact files.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("results_updated/scptr_merged_all_chr"),
        help="Output root for full-matrix all-chromosome OnTAD files.",
    )
    parser.add_argument("--ontad-bin", type=Path, default=DEFAULT_ONTAD)
    parser.add_argument("--genome-sizes", type=Path, default=Path("../genome.chrom.sizes"))
    parser.add_argument("--resolution", default="1000kb")
    parser.add_argument("--stages", nargs="+", default=STAGES)
    parser.add_argument("--penalty", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--keep-matrix", action="store_true")
    parser.add_argument("--complete-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ontad_bin.exists():
        raise SystemExit(f"OnTAD binary not found: {args.ontad_bin}")

    resolution_bp = resolution_to_bp(args.resolution)
    offsets, matrix_size = load_chrom_layout(args.genome_sizes, resolution_bp)

    written = 0
    seen = 0
    for stage in args.stages:
        stage_dir = args.input_base / stage
        if not stage_dir.exists():
            continue
        for contact_file in sorted(stage_dir.glob("*.txt")):
            if not contact_file.name.startswith("GSM"):
                continue
            seen += 1
            print(f"Running genomewide OnTAD for {stage}/{contact_file.name.split('_')[0]}", flush=True)
            written += infer_one_cell_genomewide(
                contact_file,
                stage,
                args.output_base,
                args.resolution,
                offsets,
                matrix_size,
                args.ontad_bin,
                args.penalty,
                args.timeout,
                args.keep_matrix,
            )

    if seen == 0:
        raise SystemExit(f"No input contact files found under {args.input_base}")
    if written == 0:
        raise SystemExit(
            "Genomewide OnTAD did not produce any tads.tad files; "
            "check the OnTAD binary, timeout, input contacts, and log output."
        )

    if args.complete_out is not None:
        args.complete_out.parent.mkdir(parents=True, exist_ok=True)
        args.complete_out.write_text(
            f"resolution\t{args.resolution}\n"
            f"samples_seen\t{seen}\n"
            f"samples_written\t{written}\n"
        )

    print(f"Resolution: {args.resolution}")
    print(f"Samples scanned: {seen}")
    print(f"Full-matrix genomewide OnTAD files written: {written}")
    print(f"Output root: {args.output_base}")


if __name__ == "__main__":
    main()
