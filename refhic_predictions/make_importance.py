from __future__ import annotations

import argparse
import csv
from pathlib import Path


def pick_input_tsv(folder: Path) -> Path:
    # Prefer the expected sample/chrom-specific file name.
    sample = folder.parent.name
    chrom = folder.name
    expected = folder / f"{sample}_refhic_{chrom}_BoundaryScore_250kb.tsv"
    if expected.exists():
        return expected

    candidates = sorted(folder.glob("*_BoundaryScore_250kb.tsv"))
    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(f"No '*_BoundaryScore_250kb.tsv' found in {folder}")

    raise RuntimeError(
        "Found multiple '*_BoundaryScore_250kb.tsv' files. "
        f"Please keep one file in {folder} or rename to expected pattern."
    )


def build_importance(folder: Path, normalize: bool = True) -> Path:
    folder = folder.resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    input_tsv = pick_input_tsv(folder)
    output_csv = folder / "importance.csv"

    # First pass: collect all scores and compute min/max for normalization
    rows_data = []
    min_score = float("inf")
    max_score = float("-inf")

    with input_tsv.open("r", newline="") as src:
        reader = csv.DictReader(src, delimiter="\t")

        required = {"bin250kb", "lScore", "rScore"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing required columns in {input_tsv.name}: {sorted(missing)}"
            )

        for row in reader:
            boundary = row["bin250kb"]
            score = float(row["lScore"]) + float(row["rScore"])
            rows_data.append((boundary, score))
            min_score = min(min_score, score)
            max_score = max(max_score, score)

    # Second pass: normalize and write
    with output_csv.open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=["Boundary", "Score"])
        writer.writeheader()

        score_range = max_score - min_score
        for boundary, score in rows_data:
            if normalize and score_range > 0:
                normalized = (score - min_score) / score_range
            else:
                normalized = score
            writer.writerow({"Boundary": boundary, "Score": normalized})

    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create importance.csv from *_BoundaryScore_250kb.tsv in a folder like MPP/chr4"
        )
    )
    parser.add_argument("folder", help="Path to sample/chrom folder, e.g. MPP/chr4")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip normalization (keep raw lScore+rScore values)",
    )
    args = parser.parse_args()

    out = build_importance(Path(args.folder), normalize=not args.no_normalize)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
