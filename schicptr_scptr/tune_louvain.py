#!/usr/bin/env python3
"""Sweep the Louvain resolution on a cached scHiCPTR embedding.

scHiCPTR's Section 3.4 reports an accuracy of 0.91 on these cells at 250 kb from a Louvain
clustering of its embedding. Their accuracy is computed by ``eva.cluster_acc``, which refuses a
clustering whose cluster count differs from the number of true classes, so the run behind that
number returned exactly five clusters. At the shipped default ``louvain_res = 0.5`` the
clustering here returns three, and the resolution is the parameter that sets that count.

This script re-clusters a cached embedding over a grid of resolutions and reports the cluster
count and the scores at each, so scHiCPTR can be credited with the best clustering its own
pipeline reaches on this data rather than only with the one its default returns.

Usage:
    python tune_louvain.py --resolution 250kb --chrom all
    python tune_louvain.py --resolution 250kb --chrom all --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import run_schicptr_scptr as R

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", default="250kb")
    parser.add_argument("--chrom", default="all")
    parser.add_argument("--louvain-res", type=float, nargs="+",
                        default=[0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--source", default="louvain",
                        help="which cached embedding to read (louvain or kmeans; identical)")
    args = parser.parse_args()

    embedding_path = (HERE / "results" / args.resolution / args.chrom / args.source
                      / "embedding.npy")
    if not embedding_path.exists():
        raise SystemExit(f"No cached embedding at {embedding_path}; run run_schicptr_scptr.py first")
    embedding = np.load(embedding_path)

    cells = R.discover_cells()
    labels = cells["cell_type"].to_numpy()
    connectivities = R.schicptr_graph(embedding)

    rows = []
    print(f"{args.chrom} {args.resolution}, {len(cells)} cells")
    print(f"{'louvain_res':>12} {'seed':>5} {'k':>4} {'ARI':>7} {'NMI':>7} {'ACC':>7}")
    for louvain_res in args.louvain_res:
        for seed in args.seeds:
            predicted = R.louvain_communities(connectivities, louvain_res, seed)
            metrics = R.score(labels, predicted, embedding)
            rows.append({"louvain_res": louvain_res, "seed": seed, **metrics})
            print(f"{louvain_res:>12} {seed:>5} {metrics['n_clusters']:>4} "
                  f"{metrics['ARI']:>7.4f} {metrics['NMI']:>7.4f} {metrics['ACC']:>7.4f}")

    out = embedding_path.parent.parent / "louvain_resolution_sweep.tsv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda r: r["ARI"])
    at_five = [r for r in rows if r["n_clusters"] == 5]
    print(f"\nbest ARI {best['ARI']:.4f} at louvain_res {best['louvain_res']} "
          f"(k = {best['n_clusters']}, ACC {best['ACC']:.4f})")
    if at_five:
        five = max(at_five, key=lambda r: r["ARI"])
        print(f"best of the five-cluster runs: ARI {five['ARI']:.4f}, ACC {five['ACC']:.4f} "
              f"at louvain_res {five['louvain_res']}")
    else:
        print("no resolution in the grid returned exactly five clusters")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
