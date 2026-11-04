#!/usr/bin/env python3
"""ORACLE diagnostic, not a method: start the EM from the true stage partition.

Reads stage labels on purpose. For each genome-wide hierarchy set and each method it reports
  - the objective of the stage partition itself (centroids of the true stages, no EM step)
  - where the EM converges when started from it (ARI, objective)
next to the best objective the stage-blind starts found (runs/.../result.npz). If the EM stays
near the stages and its objective is lower than every stage-blind start's, the TAD objective
does prefer the stage partition and the stage-blind fit failed only to find it.

    python oracle_diagnostic.py
"""

import csv

import numpy as np
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

from cluster import kmeans, load, median_em
from common import CB, HERE, K, RESOLUTIONS
from run_nussinov_median import Encoder
from score import truth


def main():
    stages = truth()
    oracle = np.unique(stages, return_inverse=True)[1]
    assert oracle.max() + 1 == K
    rows = []
    for resolution in RESOLUTIONS:
        for hierarchy in ("raw", "imputed"):
            cells, chroms, _ = load(resolution, "all", hierarchy)
            enc = Encoder(cells, chroms)
            x = sparse.hstack([enc.cells["T"][0], np.sqrt(CB) * enc.cells["S"][0],
                               np.sqrt(CB) * enc.cells["E"][0]]).tocsr()
            fitted = np.load(HERE / f"runs/{resolution}/all/{hierarchy}/result.npz")
            for method in ("median", "kmeans"):
                if method == "median":
                    run = lambda init, it: median_em(cells, chroms, enc, init, map, 0, max_iter=it)
                else:
                    run = lambda init, it: kmeans(x, init, max_iter=it)
                at_truth = stage_objective(method, cells, chroms, enc, x, oracle)
                labels, objective, _, _ = run(oracle, 300)
                blind = fitted[f"{method}_objective"]
                row = {"resolution": resolution, "hierarchy": hierarchy, "method": method,
                       "objective_at_stages": at_truth, "oracle_em_objective": objective,
                       "oracle_em_ari": adjusted_rand_score(stages, labels),
                       "best_stage_blind_objective": float(blind.min()),
                       "best_tad_only_objective": float(blind[1:].min())}
                rows.append(row)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    with open(HERE / "oracle_diagnostic.tsv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def stage_objective(method, cells, chroms, enc, x, partition):
    """Sum over cells of d(cell, centroid of its own stage), with no reassignment."""
    if method == "kmeans":
        total = 0.0
        for c in range(K):
            members = x[partition == c]
            mu = np.asarray(members.mean(axis=0)).ravel()
            sq = np.asarray(members.multiply(members).sum(axis=1)).ravel()
            total += float(np.sum(np.maximum(sq + mu @ mu - 2 * (members @ mu), 0)))
        return total
    from cluster import median_job
    medians = [{chrom: median_job([cells[i][chrom] for i in np.flatnonzero(partition == c)])
                for chrom in chroms} for c in range(K)]
    d = enc.distances(medians, CB)
    return float(d[np.arange(len(cells)), partition].sum())


if __name__ == "__main__":
    main()
