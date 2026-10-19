#!/usr/bin/env python3
"""Bin the raw contact files into sparse per-chromosome matrices, on mm10 and with the diagonal.

Two things about the shared matrices under ``schiccluster_scptr/work/`` are wrong for scHiCPTR,
and this script writes the alternatives so both can be measured.

**The genome build.** ``genome.chrom.sizes`` at the repository root holds mm9 lengths, but these
data are mm10: the largest coordinate observed in the raw files is 195,371,443 on chr1,
182,011,794 on chr2 and 170,673,392 on chrX, and the last two exceed the mm9 lengths of those
chromosomes. ``convert_contact_file`` clamps a coordinate past the end of a chromosome into the
last bin, so binning mm10 data on mm9 piles 264 kb of chr2 and 4.0 Mb of chrX into a single bin
each, on the eleven chromosomes where mm10 is the longer assembly. A method that reads the
contact map locally is barely affected. scHiCPTR is not such a method: its imputation runs a
random walk with restart over the whole chromosome, so one bin carrying the contacts of sixteen
distorts the stationary distribution everywhere on that chromosome.

**The diagonal.** ``convert_contact_file`` also drops the ``bin1 == bin2`` entries. scHiCPTR
convolves the contact map *before* the random walk deletes the diagonal, so the diagonal counts
are smeared into the near-diagonal bins and do carry into the embedding. Dropping them is
therefore not neutral for scHiCPTR the way it is for a method that never looks at the diagonal.

``--genome`` and ``--keep-diagonal`` select each, and ``run_schicptr_scptr.py --matrix-source``
reads the result, so the shared matrices and the corrected ones can be compared directly.
Everything else -- the intra-chromosomal filter, the clamp itself -- is what
``convert_contact_file`` does.

Usage:
    python prepare_matrices.py --res 250000 --genome mm10 --keep-diagonal
    python prepare_matrices.py --res 250000 --genome mm10
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
SOURCE_DIR = PROJECT_ROOT / "scptr" / "cell_differentiation"
WORK_DIR = ROOT / "work"
CHROMS = [f"chr{i}" for i in range(1, 20)] + ["chrX"]

# GRCm38/mm10, the assembly these contact files are mapped to
MM10 = {
    "chr1": 195471971, "chr2": 182113224, "chr3": 160039680, "chr4": 156508116,
    "chr5": 151834684, "chr6": 149736546, "chr7": 145441459, "chr8": 129401213,
    "chr9": 124595110, "chr10": 130694993, "chr11": 122082543, "chr12": 120129022,
    "chr13": 120421639, "chr14": 124902244, "chr15": 104043685, "chr16": 98207768,
    "chr17": 94987271, "chr18": 90702639, "chr19": 61431566, "chrX": 171031299,
}


def chrom_sizes(genome: str) -> dict[str, int]:
    """``mm10`` is the assembly of the data; ``mm9`` is what the shared matrices were binned on."""
    if genome == "mm10":
        return dict(MM10)
    sizes = {}
    with (PROJECT_ROOT / "genome.chrom.sizes").open() as handle:
        for line in handle:
            fields = line.split()
            if len(fields) >= 2:
                sizes[fields[0]] = int(fields[1])
    return sizes


def resolution_label(resolution: int) -> str:
    return f"{resolution // 1000}kb" if resolution % 1000 == 0 else f"{resolution}bp"


def convert(contact_path: Path, out_paths: dict[str, Path], resolution: int,
            max_bin: dict[str, int], keep_diagonal: bool) -> None:
    """One cell's raw pairs into one sparse matrix per chromosome."""
    counts: dict[str, Counter] = {chrom: Counter() for chrom in out_paths}
    with contact_path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 6:
                continue
            chrom1, pos1, chrom2, pos2 = fields[1], fields[2], fields[4], fields[5]
            if chrom1 != chrom2 or chrom1 not in counts:
                continue
            bin1 = min(int(pos1) // resolution, max_bin[chrom1])
            bin2 = min(int(pos2) // resolution, max_bin[chrom1])
            if bin1 == bin2 and not keep_diagonal:
                continue
            if bin2 < bin1:
                bin1, bin2 = bin2, bin1
            counts[chrom1][(bin1, bin2)] += 1

    for chrom, out_path in out_paths.items():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as handle:
            for (bin1, bin2), count in sorted(counts[chrom].items()):
                handle.write(f"{bin1}\t{bin2}\t{count}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--res", type=int, default=500_000)
    parser.add_argument("--chrom", nargs="+", default=CHROMS)
    parser.add_argument("--genome", default="mm10", choices=["mm10", "mm9"],
                        help="mm10 is the assembly of the data; mm9 is what the shared "
                             "matrices were binned on")
    parser.add_argument("--keep-diagonal", action="store_true",
                        help="retain the bin1 == bin2 entries the shared matrices drop")
    args = parser.parse_args()

    sizes = chrom_sizes(args.genome)
    chroms = list(args.chrom)
    max_bin = {chrom: sizes[chrom] // args.res for chrom in chroms}
    label = resolution_label(args.res)
    tag = args.genome + ("_diag" if args.keep_diagonal else "")

    cells = [(path.stem, path) for stage in sorted(SOURCE_DIR.glob("*-cell"))
             for path in sorted(stage.glob("*.txt"))]
    print(f"{len(cells)} cells, {len(chroms)} chromosomes at {label}, "
          f"{args.genome}, diagonal {'kept' if args.keep_diagonal else 'dropped'}", flush=True)
    for i, (cell_id, contact_path) in enumerate(cells, 1):
        out_paths = {chrom: WORK_DIR / label / chrom / f"cell_matrix_{tag}" / f"{cell_id}_{chrom}.txt"
                     for chrom in chroms}
        if all(p.exists() for p in out_paths.values()):
            continue
        convert(contact_path, out_paths, args.res, max_bin, args.keep_diagonal)
        if i % 100 == 0:
            print(f"  {i}/{len(cells)}", flush=True)
    print(f"wrote matrices under {WORK_DIR / label}/<chrom>/cell_matrix_{tag}/")


if __name__ == "__main__":
    main()
