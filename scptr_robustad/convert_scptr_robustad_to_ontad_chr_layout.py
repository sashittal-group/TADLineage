#!/usr/bin/env python3
"""
Convert RobusTAD SCPTR outputs from base-pair coordinates to bin-index format
and write per-chromosome files in a cell_differentiation-like folder layout.

Input layout (existing):
  scptr_robustad/robustad_scptr_data/<resolution_kb>/<stage>/<gsm>/
    - robustad_tads_all.bed
    - robustad_boundaries.bed

Output layout (new):
  scptr_robustad/cell_differentiation_robustad_ontad/<stage>/<gsm>/<chr>/<resolution_kb>kb/
    - tads.tad
    - boundaries.bed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/Users/PSLabMonisha/Desktop/tads")
INPUT_ROOT = PROJECT_ROOT / "scptr_robustad" / "robustad_scptr_data"
OUTPUT_ROOT = PROJECT_ROOT / "scptr_robustad" / "cell_differentiation_robustad_ontad"

TADS_COLUMNS = ["chrom", "startbp", "endbp", "score", "nestedScore", "Hq", "level"]
BOUNDARY_COLUMNS = ["chrom", "startbp", "endbp", "score", "nestedScore", "Hq", "level"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert robustad scptr data to OnTAD-like per-chromosome bin format."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=INPUT_ROOT,
        help="Root folder containing robustad_scptr_data.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Destination root for converted per-chromosome files.",
    )
    parser.add_argument(
        "--resolutions",
        nargs="*",
        default=["250", "500", "1000"],
        help="Resolution folders (in kb) to process.",
    )
    return parser.parse_args()


def read_tads_all(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=0)
    rename_map = {}
    if "chr" in df.columns and "chrom" not in df.columns:
        rename_map["chr"] = "chrom"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Keep only expected columns when extras are present.
    available = [c for c in TADS_COLUMNS if c in df.columns]
    df = df[available].copy()

    for col in ["startbp", "endbp", "score", "nestedScore", "Hq", "level"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "chrom" not in df.columns or "startbp" not in df.columns or "endbp" not in df.columns:
        raise ValueError(f"Missing required columns in {path}")

    for col in ["score", "nestedScore", "Hq", "level"]:
        if col not in df.columns:
            df[col] = pd.NA

    return df.dropna(subset=["chrom", "startbp", "endbp"]).copy()


def read_boundaries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=BOUNDARY_COLUMNS)
    for col in ["startbp", "endbp", "score", "nestedScore", "Hq", "level"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["chrom", "startbp", "endbp"]).copy()


def to_bins(df: pd.DataFrame, bp: int) -> pd.DataFrame:
    out = df.copy()
    out["startbp"] = out["startbp"].astype(int)
    out["endbp"] = out["endbp"].astype(int)

    # OnTAD-style 1-based bin indexing.
    out["start_bin"] = (out["startbp"] // bp).astype(int) + 1
    out["end_bin"] = ((out["endbp"] - 1) // bp).astype(int) + 1
    return out


def write_per_chrom_tads(df: pd.DataFrame, out_base: Path, res_label: str) -> int:
    written = 0
    for chrom, part in df.groupby("chrom", sort=True):
        chrom_dir = out_base / str(chrom) / f"{res_label}kb"
        chrom_dir.mkdir(parents=True, exist_ok=True)
        out_file = chrom_dir / "tads.tad"

        write_df = part[["start_bin", "end_bin", "level", "score", "nestedScore", "Hq"]].copy()
        write_df = write_df.sort_values(["start_bin", "end_bin", "level"], kind="mergesort")
        write_df.to_csv(out_file, sep="\t", header=False, index=False)
        written += 1
    return written


def write_per_chrom_boundaries(df: pd.DataFrame, out_base: Path, res_label: str) -> int:
    written = 0
    for chrom, part in df.groupby("chrom", sort=True):
        chrom_dir = out_base / str(chrom) / f"{res_label}kb"
        chrom_dir.mkdir(parents=True, exist_ok=True)
        out_file = chrom_dir / "boundaries.bed"

        write_df = part[["start_bin", "end_bin", "score", "nestedScore", "Hq", "level"]].copy()
        write_df = write_df.sort_values(["start_bin", "end_bin"], kind="mergesort")
        write_df.to_csv(out_file, sep="\t", header=False, index=False)
        written += 1
    return written


def process_one_cell(cell_dir: Path, out_root: Path, res_kb: int) -> tuple[int, int]:
    tads_path = cell_dir / "robustad_tads_all.bed"
    boundaries_path = cell_dir / "robustad_boundaries.bed"
    if not tads_path.exists() or not boundaries_path.exists():
        return 0, 0

    stage = cell_dir.parent.name
    gsm = cell_dir.name
    out_base = out_root / stage / gsm

    bp = res_kb * 1000

    tads = read_tads_all(tads_path)
    tads = to_bins(tads, bp)
    tads_written = write_per_chrom_tads(tads, out_base, str(res_kb))

    boundaries = read_boundaries(boundaries_path)
    boundaries = to_bins(boundaries, bp)
    boundaries_written = write_per_chrom_boundaries(boundaries, out_base, str(res_kb))

    # Keep metadata near the sample root when present.
    meta_src = cell_dir / "cell_metadata.tsv"
    if meta_src.exists():
        meta_dst = out_base / "cell_metadata.tsv"
        meta_dst.parent.mkdir(parents=True, exist_ok=True)
        meta_dst.write_text(meta_src.read_text())

    return tads_written, boundaries_written


def main() -> None:
    args = parse_args()
    input_root = args.input_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    total_cells = 0
    total_tad_files = 0
    total_boundary_files = 0

    for res in args.resolutions:
        res_dir = input_root / str(res)
        if not res_dir.exists():
            continue

        res_kb = int(res)
        stage_dirs = [p for p in sorted(res_dir.iterdir()) if p.is_dir()]
        for stage_dir in stage_dirs:
            for cell_dir in sorted(stage_dir.iterdir()):
                if not cell_dir.is_dir() or not cell_dir.name.startswith("GSM"):
                    continue
                tads_written, boundaries_written = process_one_cell(cell_dir, output_root, res_kb)
                if tads_written or boundaries_written:
                    total_cells += 1
                    total_tad_files += tads_written
                    total_boundary_files += boundaries_written

    print(f"Processed cells: {total_cells}")
    print(f"Wrote per-chromosome tads.tad files: {total_tad_files}")
    print(f"Wrote per-chromosome boundaries.bed files: {total_boundary_files}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
