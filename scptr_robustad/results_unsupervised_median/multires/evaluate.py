#!/usr/bin/env python3
"""Post-hoc ARI/NMI of every fit at every bin size; the only script here that reads stage labels.

Writes ari_summary.tsv with one row per (resolution, chromosome); chromosome "all" is the
genome-wide fit. 1 Mb rows come from ../nussinov_median (nussinov.py median, same EM).
At 250/500 kb the genome-wide fit is genome_graph1mb (cell similarity from the 1 Mb binning,
hierarchies at the bin size); the fit with its own-bin graph (genome/) is printed for reference.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from evaluate import truth  # noqa: E402
from extract import CHROMS  # noqa: E402

FOLDERS = {
    "250kb": (HERE / "250000/genome_graph1mb", HERE / "250000/chromosome_only"),
    "500kb": (HERE / "500000/genome_graph1mb", HERE / "500000/chromosome_only"),
    "1000kb": (HERE.parent / "nussinov_median/run", HERE.parent / "nussinov_median/per_chromosome"),
}


def main():
    labels = truth()
    rows = []
    for resolution, (genome, per_chrom) in FOLDERS.items():
        for chrom, folder in [("all", genome)] + [(c, per_chrom / c) for c in CHROMS]:
            if not (folder / "fit.npz").exists():
                continue
            y = np.load(folder / "fit.npz")["labels"]
            rows.append({"resolution": resolution, "chromosome": chrom,
                         "ari": adjusted_rand_score(labels, y), "nmi": normalized_mutual_info_score(labels, y),
                         "path": str(folder / "fit.npz")})
    with open(HERE / "ari_summary.tsv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    for r in ("250000", "500000"):
        own = np.load(HERE / r / "genome/fit.npz")["labels"]
        print(f"{r} genome-wide with its own-bin graph (not used in the figure): "
              f"ARI {adjusted_rand_score(labels, own):.3f}")
    for resolution in FOLDERS:
        sub = [r for r in rows if r["resolution"] == resolution]
        genome = [r["ari"] for r in sub if r["chromosome"] == "all"]
        chroms = [r["ari"] for r in sub if r["chromosome"] != "all"]
        if sub:
            print(f"{resolution}: genome-wide {genome[0] if genome else float('nan'):.3f}; "
                  f"{len(chroms)} chromosomes, median {np.median(chroms) if chroms else float('nan'):.3f}, "
                  f"best {max(chroms) if chroms else float('nan'):.3f}")


if __name__ == "__main__":
    main()
