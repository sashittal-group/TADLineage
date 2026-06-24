#!/usr/bin/env python3
"""Add bin-number columns to RobusTAD boundary and TAD outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


BASE = Path(__file__).resolve().parent / "robustad_tads"
RES_TO_BP = {
    "5kb": 5_000,
    "10kb": 10_000,
    "25kb": 25_000,
    "50kb": 50_000,
    "100kb": 100_000,
    "250kb": 250_000,
    "500kb": 500_000,
    "1mb": 1_000_000,
    "2.5mb": 2_500_000,
}


def resolution_bp(path: Path) -> int:
    label = path.parent.name.lower()
    if label not in RES_TO_BP:
        raise ValueError(f"Unknown resolution folder {path.parent}; add it to RES_TO_BP")
    return RES_TO_BP[label]


def empty_boundary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "chr",
            "start",
            "end",
            "bin",
            "leftScore",
            "leftBoundary",
            "rightScore",
            "rightBoundary",
        ]
    )


def empty_tad_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "chr",
            "start",
            "end",
            "start_bin",
            "end_bin",
            "score",
            "nestedScore",
            "Hq",
            "level",
        ]
    )


def convert_boundary(path: Path, bp: int) -> Path:
    try:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=[
                "chr",
                "start",
                "end",
                "leftScore",
                "leftBoundary",
                "rightScore",
                "rightBoundary",
            ],
        )
    except EmptyDataError:
        df = empty_boundary_frame()

    for col in ["start", "end", "leftScore", "leftBoundary", "rightScore", "rightBoundary"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["start", "end"]).copy()
    if not df.empty:
        df["start"] = df["start"].astype(int)
        df["end"] = df["end"].astype(int)
        df["bin"] = (df["start"] // bp).astype(int)
    else:
        df = empty_boundary_frame()

    out = path.with_name(path.stem + "_with_bins.tsv")
    df[
        [
            "chr",
            "start",
            "end",
            "bin",
            "leftScore",
            "leftBoundary",
            "rightScore",
            "rightBoundary",
        ]
    ].to_csv(out, sep="\t", index=False)
    return out


def convert_tads(path: Path, bp: int, has_header: bool) -> Path:
    try:
        df = pd.read_csv(
            path,
            sep="\t",
            header=0 if has_header else None,
            names=None
            if has_header
            else ["chr", "start", "end", "score", "nestedScore", "Hq", "level"],
        )
    except EmptyDataError:
        df = empty_tad_frame()

    rename_map = {
        "chrom": "chr",
        "startbp": "start",
        "endbp": "end",
    }
    df = df.rename(columns={key: val for key, val in rename_map.items() if key in df.columns})

    for col in ["chr", "start", "end", "score", "nestedScore", "Hq", "level"]:
        if col not in df.columns:
            df[col] = pd.NA
    for col in ["start", "end", "score", "nestedScore", "Hq", "level"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["start", "end"]).copy()
    if not df.empty:
        df["start"] = df["start"].astype(int)
        df["end"] = df["end"].astype(int)
        df["start_bin"] = (df["start"] // bp).astype(int)
        df["end_bin"] = ((df["end"] - 1) // bp).astype(int)
    else:
        df = empty_tad_frame()

    out = path.with_name(path.stem + "_with_bins.tsv")
    df[
        [
            "chr",
            "start",
            "end",
            "start_bin",
            "end_bin",
            "score",
            "nestedScore",
            "Hq",
            "level",
        ]
    ].to_csv(out, sep="\t", index=False)
    return out


def convert_file(path: Path) -> Path | None:
    bp = resolution_bp(path)
    if path.name == "boundary.bed":
        return convert_boundary(path, bp)
    if path.name == "tads_all.bed":
        return convert_tads(path, bp, has_header=True)
    if path.name == "tads_hq.bed":
        return convert_tads(path, bp, has_header=False)
    return None


def main() -> None:
    written = []
    for path in sorted(BASE.glob("*/*/*/*.bed")):
        if path.name.endswith("_with_bins.tsv"):
            continue
        out = convert_file(path)
        if out is not None:
            written.append(out)

    print(f"Wrote {len(written)} files")
    for out in written:
        print(out)


if __name__ == "__main__":
    main()
