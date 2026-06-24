#!/usr/bin/env python3

from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent
RES250 = 250000


def convert_boundary(path: Path) -> Path:
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
    df["bin250kb"] = (df["start"] // RES250).astype(int)

    out_path = path.with_name(path.stem + "_with_bins.tsv")
    df[
        [
            "chr",
            "start",
            "end",
            "bin250kb",
            "leftScore",
            "leftBoundary",
            "rightScore",
            "rightBoundary",
        ]
    ].to_csv(out_path, sep="\t", index=False)
    return out_path


def convert_tads(path: Path) -> Path:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "score", "nestedScore", "Hq", "level"],
    )

    for col in ["start", "end", "score", "nestedScore", "Hq", "level"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["start", "end"]).copy()
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)

    df["start_bin250kb"] = (df["start"] // RES250).astype(int)
    df["end_bin250kb"] = ((df["end"] - 1) // RES250).astype(int)

    out_path = path.with_name(path.stem + "_with_bins.tsv")
    df[
        [
            "chr",
            "start",
            "end",
            "start_bin250kb",
            "end_bin250kb",
            "score",
            "nestedScore",
            "Hq",
            "level",
        ]
    ].to_csv(out_path, sep="\t", index=False)
    return out_path


def main() -> None:
    targets = [
        BASE / "CMP" / "chr4" / "250kb" / "CMP_chr4_250kb_boundary.bed",
        BASE / "CMP" / "chr4" / "250kb" / "CMP_chr4_250kb_tads_all.bed",
        BASE / "CMP" / "chr4" / "250kb" / "CMP_chr4_250kb_tads_hq.bed",
        BASE / "MPP" / "chr4" / "250kb" / "MPP_chr4_250kb_boundary.bed",
        BASE / "MPP" / "chr4" / "250kb" / "MPP_chr4_250kb_tads_all.bed",
        BASE / "MPP" / "chr4" / "250kb" / "MPP_chr4_250kb_tads_hq.bed",
    ]

    for target in targets:
        if "boundary" in target.name:
            out = convert_boundary(target)
        else:
            out = convert_tads(target)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
