#!/usr/bin/env python3
"""Chromosome-only fits with the paper's distance and the Nussinov median; no stage labels.

Same hierarchies as ../per_chromosome/chromosome_only/{chrom}/trees.npz (contacts, graph and
imputation of that chromosome alone) and the same spectral start, rebuilt from that
chromosome's counts exactly as ../per_chromosome.py does. Spectral start + random kmeans++
starts, lowest objective kept. One process per chromosome; chromosomes run in parallel.

    /Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/python per_chromosome_nussinov.py
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.cluster import SpectralClustering

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
sys.path.insert(0, str(EXPERIMENT))

from run_nussinov_median import Encoder, kmeanspp, load_cells, median_em  # noqa: E402
from signals import normalize, pca  # noqa: E402
from extract import CHROMS  # noqa: E402

K, CB = 5, 1.0


def fit(job):
    chrom, random_starts = job
    out = HERE / "per_chromosome" / chrom
    if (out / "run.json").exists():
        return chrom
    out.mkdir(parents=True, exist_ok=True)
    cells, chroms = load_cells(EXPERIMENT / "per_chromosome/chromosome_only" / chrom)
    enc = Encoder(cells, chroms)
    counts = np.load(EXPERIMENT / f"data/{chrom}_counts.npz")["counts"]
    a, b = np.triu_indices(counts.shape[1], 1)
    x = pca(np.sqrt(normalize(counts[:, a, b])), 30)[:, :10]
    spectral = SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=10,
                                  assign_labels="cluster_qr", random_state=0).fit_predict(x)
    solutions = []
    for seed in range(random_starts + 1):
        init = spectral if seed == 0 else kmeanspp(enc, cells, K, seed, CB)
        labels, medians, history = median_em(cells, chroms, enc, K, init, map, seed, CB, log=f"{chrom} ")
        solutions.append((history[-1], labels, history))
    objective, labels, history = min(solutions, key=lambda s: s[0])
    np.savez_compressed(out / "fit.npz", labels=labels, spectral_start=spectral,
                        restart_labels=np.array([s[1] for s in solutions]))
    (out / "run.json").write_text(json.dumps({
        "chromosome": chrom, "k": K, "Ct": 1.0, "Cb": CB, "tolerance": 0,
        "median": "nussinov.infer_tads_with_dp", "stage_labels_used_in_fit": False,
        "objective": objective, "history": history,
        "restart_objectives": [s[0] for s in solutions],
        "starts": ["spectral"] + [f"kmeans++ seed {s}" for s in range(1, random_starts + 1)],
    }, indent=2))
    return chrom


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--random-starts", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    # largest chromosomes first, so the slowest fits do not start last
    order = sorted(CHROMS, key=lambda c: -np.load(EXPERIMENT / f"data/{c}_counts.npz")["counts"].shape[1])
    with Pool(args.workers) as pool:
        for chrom in pool.imap_unordered(fit, [(c, args.random_starts) for c in order]):
            print(f"done {chrom}", flush=True)


if __name__ == "__main__":
    main()
