#!/usr/bin/env python3
"""Median EM vs k-means under the paper's TAD distance, from identical starts; no stage labels.

Distance (the paper's, tolerance 0, summed over chromosomes):
    d(H, H') = |T delta T'| + Cb (|S delta S'| + |E delta E'|),   Cb = 1
T, S, E are binary indicator vectors (domains, start boundaries, end boundaries), so d is the
squared Euclidean distance between the concatenated vectors x = [T | S | E]. Hence:

  median   the paper's EM: centroid = exact median hierarchy (fast_median, = nussinov.py cost),
           a laminar set of domains; assign each cell to the nearest median under d
  kmeans   Lloyd's k-means on x: centroid = the mean of the cluster's x (a fractional
           "hierarchy"), assign to the nearest mean under the same d. No median is computed.

Both run from the same initial partitions: the paper's spectral start (contact embedding, see
common.py) and RANDOM_STARTS k-means++ starts drawn under d (UM/nussinov_median kmeanspp, as the
paper's fits). Every start's labels and objective are saved; the selection rules are applied in
evaluate.py, which is the only script that reads stage labels.

    python cluster.py --resolution 1000000 --hierarchy raw --scopes all
    python cluster.py --resolution 500000 --hierarchy imputed --scopes chromosomes
"""

import argparse
import json
from multiprocessing import Pool

import numpy as np
from scipy import sparse
from sklearn.cluster import SpectralClustering

from common import CB, CHROMS, K, RANDOM_STARTS, embedding, imputed_folder, raw_folder, run_folder
from run_nussinov_median import Encoder, kmeanspp, load_cells
from fast_median import median


def median_job(member_sets):
    return median(member_sets, 1.0, CB)[0]


def median_em(cells, chroms, enc, init, mapper, seed, max_iter=100):
    """The EM of UM/multires/fit.py (same tie rule, cache and empty-cluster reseed)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(init).copy()
    cache, objective = {}, None
    for iteration in range(1, max_iter + 1):
        medians = [dict() for _ in range(K)]
        jobs, slots = [], []
        for c in range(K):
            members = tuple(np.flatnonzero(labels == c))
            if not members:
                medians[c] = {ch: set(s) for ch, s in cells[int(rng.integers(len(cells)))].items()}
                continue
            for chrom in chroms:
                if (chrom, members) in cache:
                    medians[c][chrom] = cache[(chrom, members)]
                else:
                    jobs.append([cells[i][chrom] for i in members])
                    slots.append((c, chrom, members))
        for (c, chrom, members), result in zip(slots, mapper(median_job, jobs)):
            cache[(chrom, members)] = medians[c][chrom] = result
        d = enc.distances(medians, CB)
        new = d.argmin(axis=1)
        tied = np.isclose(d[np.arange(len(cells)), labels], d.min(axis=1))
        new[tied] = labels[tied]
        objective = float(d[np.arange(len(cells)), new].sum())
        if np.array_equal(new, labels):
            return labels, objective, iteration, True
        labels = new
    return labels, objective, max_iter, False


def kmeans(x, init, max_iter=300):
    """Lloyd's k-means on the binary [T|S|E] matrix; same tie rule as median_em."""
    n = x.shape[0]
    sq = np.asarray(x.multiply(x).sum(axis=1)).ravel()
    labels = np.asarray(init).copy()
    objective = None
    for iteration in range(1, max_iter + 1):
        onehot = sparse.csr_matrix((np.ones(n), (labels, np.arange(n))), shape=(K, n))
        sizes = np.bincount(labels, minlength=K)
        mu = np.asarray((onehot @ x).todense()) / np.maximum(sizes, 1)[:, None]
        d = np.maximum(sq[:, None] + (mu ** 2).sum(axis=1)[None, :] - 2 * np.asarray(x @ mu.T), 0)
        for c in np.flatnonzero(sizes == 0):
            # empty cluster: reseed at the cell farthest from its own centroid
            far = int(np.argmax(d[np.arange(n), labels]))
            mu[c] = x[far].toarray().ravel()
            d[:, c] = np.maximum(sq + (mu[c] ** 2).sum() - 2 * np.asarray(x @ mu[c]).ravel(), 0)
        new = d.argmin(axis=1)
        tied = np.isclose(d[np.arange(n), labels], d.min(axis=1))
        new[tied] = labels[tied]
        objective = float(d[np.arange(n), new].sum())
        if np.array_equal(new, labels):
            return labels, objective, iteration, True
        labels = new
    return labels, objective, max_iter, False


def load(resolution, scope, hierarchy):
    folder = imputed_folder(resolution, scope) if hierarchy == "imputed" else raw_folder(resolution)
    cells, chroms = load_cells(folder)
    if scope != "all" and chroms != [scope]:
        # raw hierarchies are per cell, so a chromosome's are the genome-wide ones restricted
        cells, chroms = [{scope: c[scope]} for c in cells], [scope]
    return cells, chroms, folder


def fit(resolution, scope, hierarchy, mapper):
    out = run_folder(resolution, scope, hierarchy)
    if (out / "run.json").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    cells, chroms, folder = load(resolution, scope, hierarchy)
    enc = Encoder(cells, chroms)
    x = sparse.hstack([enc.cells["T"][0], np.sqrt(CB) * enc.cells["S"][0],
                       np.sqrt(CB) * enc.cells["E"][0]]).tocsr()
    spectral = SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=10,
                                  assign_labels="cluster_qr",
                                  random_state=0).fit_predict(embedding(resolution, scope))
    names = ["spectral"] + [f"kmeans++ seed {s}" for s in range(1, RANDOM_STARTS + 1)]
    results = {k: [] for k in ("init", "median_labels", "median_objective", "median_iterations",
                               "median_converged", "kmeans_labels", "kmeans_objective",
                               "kmeans_iterations", "kmeans_converged")}
    for seed, name in enumerate(names):
        init = spectral if seed == 0 else kmeanspp(enc, cells, K, seed, CB)
        results["init"].append(init)
        for method, solved in (("median", median_em(cells, chroms, enc, init, mapper, seed)),
                               ("kmeans", kmeans(x, init))):
            for key, value in zip(("labels", "objective", "iterations", "converged"), solved):
                results[f"{method}_{key}"].append(value)
        print(f"  {resolution} {scope} {hierarchy} {name}: median objective "
              f"{results['median_objective'][-1]:.0f}, kmeans objective {results['kmeans_objective'][-1]:.1f}",
              flush=True)
    np.savez_compressed(out / "result.npz", starts=np.array(names),
                        **{k: np.array(v) for k, v in results.items()})
    (out / "run.json").write_text(json.dumps({
        "resolution": resolution, "scope": scope, "hierarchy": hierarchy,
        "hierarchy_source": str(folder), "k": K, "Ct": 1.0, "Cb": CB, "tolerance": 0,
        "n_domains_per_cell": float(enc.cells["T"][1].mean()),
        "n_distinct_features": int(x.shape[1]),
        "starts": names, "stage_labels_used_in_fit": False,
        "median_not_converged": [n for n, c in zip(names, results["median_converged"]) if not c],
        "kmeans_not_converged": [n for n, c in zip(names, results["kmeans_converged"]) if not c],
    }, indent=2))
    return out


def fit_chromosome(job):
    return fit(*job, map)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--hierarchy", choices=["raw", "imputed"], required=True)
    parser.add_argument("--scopes", choices=["all", "chromosomes"], default="all")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    with Pool(args.workers) as pool:
        if args.scopes == "all":
            fit(args.resolution, "all", args.hierarchy, pool.map)
        else:
            jobs = [(args.resolution, c, args.hierarchy) for c in CHROMS]
            for out in pool.imap_unordered(fit_chromosome, jobs):
                print(f"done {out}", flush=True)


if __name__ == "__main__":
    main()
