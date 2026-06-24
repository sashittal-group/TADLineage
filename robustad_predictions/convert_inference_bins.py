#!/usr/bin/env python3
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent / "ontad_panel" / "inference"
RES_TO_BP = {
    "100kb": 100_000,
    "250kb": 250_000,
    "500kb": 500_000,
    "1mb": 1_000_000,
    "2mb": 2_500_000,
}


def convert_boundary(path: Path, bp: int) -> Path:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "leftScore", "leftBoundary", "rightScore", "rightBoundary"],
    )
    for col in ["start", "end", "leftScore", "leftBoundary", "rightScore", "rightBoundary"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["start", "end"]).copy()
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["bin"] = (df["start"] // bp).astype(int)

    out = path.with_name(path.stem + "_with_bins.tsv")
    df[["chr", "start", "end", "bin", "leftScore", "leftBoundary", "rightScore", "rightBoundary"]].to_csv(
        out, sep="\t", index=False
    )
    return out


def convert_tads(path: Path, bp: int, has_header: bool) -> Path:
    df = pd.read_csv(
        path,
        sep="\t",
        header=0 if has_header else None,
        names=["chr", "start", "end", "score", "nestedScore", "Hq", "level"],
    )

    rename_map = {}
    if "chrom" in df.columns:
        rename_map["chrom"] = "chr"
    if "startbp" in df.columns:
        rename_map["startbp"] = "start"
    if "endbp" in df.columns:
        rename_map["endbp"] = "end"
    if rename_map:
        df = df.rename(columns=rename_map)

    for col in ["start", "end", "score", "nestedScore", "Hq", "level"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["start", "end"]).copy()
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["start_bin"] = (df["start"] // bp).astype(int)
    df["end_bin"] = ((df["end"] - 1) // bp).astype(int)

    for col in ["score", "nestedScore", "Hq", "level"]:
        if col not in df.columns:
            df[col] = pd.NA

    out = path.with_name(path.stem + "_with_bins.tsv")
    df[["chr", "start", "end", "start_bin", "end_bin", "score", "nestedScore", "Hq", "level"]].to_csv(
        out, sep="\t", index=False
    )
    return out


def main() -> None:
    written = []
    for path in sorted(BASE.glob("*/*/*.bed")):
        res = path.parent.name
        bp = RES_TO_BP[res]

        if path.name.endswith("_boundary.bed"):
            written.append(convert_boundary(path, bp))
        elif path.name.endswith("_tads_all.bed"):
            written.append(convert_tads(path, bp, has_header=True))
        else:
            written.append(convert_tads(path, bp, has_header=False))

    print(f"Wrote {len(written)} files")
    for out in written:
        print(out)


if __name__ == "__main__":
    main()
