#!/usr/bin/env python3
"""Median EM on the hierarchies of build.py: paper's distance, exact median; no stage labels.

Same EM as ../nussinov_median/run_nussinov_median.py (distance |T delta T'| + |S delta S'| +
|E delta E'|, tolerance 0; spectral start + random kmeans++ starts; lowest objective kept)
with fast_median.median, which reaches the same optimal cost as nussinov.py
(check_fast_median.py) and is fast enough for 250 kb.

    python fit.py --resolution 500000                      # genome-wide and the 20 chromosome-only fits
    python fit.py --resolution 500000 --only genome_graph1mb  # one genome-wide variant
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
sys.path.insert(0, str(EXPERIMENT / "nussinov_median"))
sys.path.insert(0, str(EXPERIMENT))

from run_nussinov_median import Encoder, kmeanspp, load_cells  # noqa: E402
from fast_median import median  # noqa: E402
from extract import CHROMS  # noqa: E402

K, CB, RANDOM_STARTS = 5, 1.0, 4


def median_job(member_sets):
    return median(member_sets, 1.0, CB)[0]


def median_em(cells, chroms, enc, init, mapper, seed, max_iter=50, log=""):
    rng = np.random.default_rng(seed)
    labels = np.asarray(init).copy()
    cache, history = {}, []
    for iteration in range(max_iter):
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
        history.append(float(d[np.arange(len(cells)), new].sum()))
        print(f"  {log}seed {seed} iter {iteration + 1}: objective {history[-1]:.0f}, "
              f"sizes {np.bincount(new, minlength=K).tolist()}", flush=True)
        if np.array_equal(new, labels):
            return labels, medians, history
        labels = new
    raise RuntimeError("Median EM did not converge")


def fit(folder, mapper, log=""):
    """All starts on one set of hierarchies; writes fit.npz and run.json into folder."""
    if (folder / "run.json").exists():
        return
    cells, chroms = load_cells(folder)
    enc = Encoder(cells, chroms)
    x = np.load(folder / "embedding.npy")
    spectral = SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=10,
                                  assign_labels="cluster_qr", random_state=0).fit_predict(x)
    solutions = []
    for seed in range(RANDOM_STARTS + 1):
        init = spectral if seed == 0 else kmeanspp(enc, cells, K, seed, CB)
        labels, medians, history = median_em(cells, chroms, enc, init, mapper, seed, log=log)
        solutions.append((history[-1], labels, history))
    objective, labels, history = min(solutions, key=lambda s: s[0])
    np.savez_compressed(folder / "fit.npz", labels=labels, spectral_start=spectral,
                        restart_labels=np.array([s[1] for s in solutions]))
    (folder / "run.json").write_text(json.dumps({
        "k": K, "Ct": 1.0, "Cb": CB, "tolerance": 0, "median": "fast_median (exact, = nussinov.py cost)",
        "stage_labels_used_in_fit": False, "objective": objective, "history": history,
        "restart_objectives": [s[0] for s in solutions],
        "starts": ["spectral"] + [f"kmeans++ seed {s}" for s in range(1, RANDOM_STARTS + 1)],
    }, indent=2))


def fit_chromosome(folder):
    fit(folder, map, log=f"{folder.name} ")
    return folder.name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", default=None, help="fit just this genome-wide folder under {R}/")
    args = parser.parse_args()
    root = HERE / str(args.resolution)
    if args.only:
        with Pool(args.workers) as pool:
            fit(root / args.only, pool.map, log=f"{args.only} ")
        return
    with Pool(args.workers) as pool:
        print("genome-wide", flush=True)
        fit(root / "genome", pool.map, log="genome ")
        folders = [root / "chromosome_only" / c for c in CHROMS]
        for chrom in pool.imap_unordered(fit_chromosome, folders):
            print(f"done {chrom}", flush=True)


if __name__ == "__main__":
    main()
