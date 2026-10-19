#!/usr/bin/env python3
"""Median EM with the paper's TAD distance and the Nussinov DP median; no stage metadata is read.

Same imputed hierarchies as ../final (trees.npz: they do not depend on the clustering) and
the same spectral start, but the two pieces the paper specifies replace the shortcuts:

  distance  d(H, H') = |T delta T'| + Cb (|S delta S'| + |E delta E'|), tolerance 0, summed
            over chromosomes (starts and ends are keyed by chromosome)
  median    nussinov.infer_tads_with_dp per chromosome and cluster (exact for this distance);
            ../final used the majority rule, which is exact only when Cb = 0

Restarts: the spectral start of ../final plus random kmeans++ starts under d; the lowest
objective is kept, as in ../final. Results are written after every start.

    /Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/python run_nussinov_median.py
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.cluster import SpectralClustering

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
REPO_ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nussinov import infer_tads_with_dp  # noqa: E402


def load_cells(folder=EXPERIMENT / "final"):
    trees = sparse.load_npz(folder / "trees.npz").tocsr()
    intervals = [tuple(t) for t in json.loads((folder / "intervals.json").read_text())]
    chroms = sorted({c for c, _, _ in intervals})
    cells = []
    for i in range(trees.shape[0]):
        by_chrom = {c: set() for c in chroms}
        for j in trees[i].indices:
            c, a, b = intervals[j]
            by_chrom[c].add((a, b))
        cells.append(by_chrom)
    return cells, chroms


class Encoder:
    """Binary cell-by-feature matrices for intervals, starts and ends; medians may hold
    features no cell has, which only add to the median's size."""

    def __init__(self, cells, chroms):
        self.chroms = chroms
        keys = {"T": set(), "S": set(), "E": set()}
        for cell in cells:
            for c in chroms:
                for a, b in cell[c]:
                    keys["T"].add((c, a, b))
                    keys["S"].add((c, a))
                    keys["E"].add((c, b))
        self.index = {k: {key: i for i, key in enumerate(sorted(v))} for k, v in keys.items()}
        self.cells = {k: self.encode_many(cells, k) for k in keys}

    def features(self, hierarchy, kind):
        out = set()
        for c in self.chroms:
            for a, b in hierarchy[c]:
                out.add({"T": (c, a, b), "S": (c, a), "E": (c, b)}[kind])
        return out

    def encode_many(self, hierarchies, kind):
        rows, cols, sizes = [], [], []
        for i, h in enumerate(hierarchies):
            feats = self.features(h, kind)
            sizes.append(len(feats))
            for f in feats:
                if f in self.index[kind]:
                    rows.append(i)
                    cols.append(self.index[kind][f])
        m = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                              shape=(len(hierarchies), len(self.index[kind])))
        return m, np.array(sizes, float)

    def distances(self, medians, cb=1.0):
        total = 0.0
        for kind, weight in (("T", 1.0), ("S", cb), ("E", cb)):
            x, xs = self.cells[kind]
            m, ms = self.encode_many(medians, kind)
            total = total + weight * (xs[:, None] + ms[None, :] - 2 * (x @ m.T).toarray())
        return np.maximum(total, 0)


def nussinov_median(job):
    """Exact median of one chromosome of one cluster under d (Ct = 1, Cb)."""
    member_sets, cb = job
    pos = set()
    for s in member_sets:
        for a, b in s:
            pos.update((a, b))
    if len(pos) < 2:
        return set()
    leaves = {i + 1: {f"[{a},{b}]" for a, b in s} for i, s in enumerate(member_sets)}
    edges = [(0, i + 1) for i in range(len(member_sets))]
    median, _ = infer_tads_with_dp(edges, leaves, pos, pos, Ct=1.0, Cb=cb)
    return set(median)


def compute_medians(cells, chroms, labels, k, mapper, cache, rng, cb):
    jobs, slots = [], []
    medians = [dict() for _ in range(k)]
    for c_idx in range(k):
        members = tuple(np.flatnonzero(labels == c_idx))
        if not members:
            # empty cluster: reseed with a random cell, as ../hierarchy.median_em does
            medians[c_idx] = {c: set(s) for c, s in cells[int(rng.integers(len(cells)))].items()}
            continue
        for chrom in chroms:
            key = (chrom, members)
            if key in cache:
                medians[c_idx][chrom] = cache[key]
            else:
                jobs.append(([cells[i][chrom] for i in members], cb))
                slots.append((c_idx, chrom, key))
    for (c_idx, chrom, key), result in zip(slots, mapper(nussinov_median, jobs)):
        cache[key] = result
        medians[c_idx][chrom] = result
    return medians


def median_em(cells, chroms, enc, k, initial, mapper, seed, cb, max_iter=50, log=""):
    rng = np.random.default_rng(seed)
    labels = np.asarray(initial).copy()
    cache, history = {}, []
    for iteration in range(max_iter):
        medians = compute_medians(cells, chroms, labels, k, mapper, cache, rng, cb)
        d = enc.distances(medians, cb)
        new = d.argmin(axis=1)
        tied = np.isclose(d[np.arange(len(cells)), labels], d.min(axis=1))
        new[tied] = labels[tied]
        objective = float(d[np.arange(len(cells)), new].sum())
        history.append(objective)
        print(f"  {log}seed {seed} iter {iteration + 1}: objective {objective:.0f}, sizes "
              f"{np.bincount(new, minlength=k).tolist()}", flush=True)
        if np.array_equal(new, labels):
            return labels, medians, history
        labels = new
    raise RuntimeError("Median EM did not converge")


def kmeanspp(enc, cells, k, seed, cb):
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(cells)))]
    for _ in range(k - 1):
        d = enc.distances([cells[i] for i in chosen], cb).min(axis=1)
        d[chosen] = 0
        chosen.append(int(rng.choice(len(cells), p=d / d.sum())))
    return enc.distances([cells[i] for i in chosen], cb).argmin(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--cb", type=float, default=1.0)
    parser.add_argument("--neighbors", type=int, default=10, help="spectral start, as in ../final")
    parser.add_argument("--random-starts", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default="run")
    args = parser.parse_args()
    out = HERE / args.output
    out.mkdir(parents=True, exist_ok=True)

    cells, chroms = load_cells()
    enc = Encoder(cells, chroms)
    embedding = np.load(EXPERIMENT / "final/fit.npz")["embedding"]
    spectral = SpectralClustering(args.k, affinity="nearest_neighbors", n_neighbors=args.neighbors,
                                  assign_labels="cluster_qr", random_state=0).fit_predict(embedding)
    starts = [("spectral", spectral)] + [(f"kmeans++ seed {s}", None) for s in range(1, args.random_starts + 1)]

    solutions = []
    with Pool(args.workers) as pool:
        for seed, (name, init) in enumerate(starts):
            if init is None:
                init = kmeanspp(enc, cells, args.k, seed, args.cb)
            print(f"start {name}", flush=True)
            labels, medians, history = median_em(cells, chroms, enc, args.k, init, pool.map, seed, args.cb)
            solutions.append({"start": name, "objective": history[-1], "history": history,
                              "labels": labels, "medians": medians})
            best = min(solutions, key=lambda s: s["objective"])
            (out / "run.json").write_text(json.dumps({
                "k": args.k, "Ct": 1.0, "Cb": args.cb, "tolerance": 0,
                "median": "nussinov.infer_tads_with_dp per chromosome and cluster",
                "hierarchies": "../final/trees.npz (unchanged)",
                "stage_labels_used_in_fit": False,
                "selected_start": best["start"], "objective": best["objective"],
                "cluster_sizes": np.bincount(best["labels"], minlength=args.k).tolist(),
                "starts": [{"start": s["start"], "objective": s["objective"], "history": s["history"],
                            "cluster_sizes": np.bincount(s["labels"], minlength=args.k).tolist()}
                           for s in solutions],
            }, indent=2))
            np.savez_compressed(out / "fit.npz", labels=best["labels"], spectral_start=spectral,
                                restart_labels=np.array([s["labels"] for s in solutions]))
            with (out / "median_hierarchies.tsv").open("w") as handle:
                handle.write("cluster\tchromosome\tstart_bin\tend_bin\n")
                for c_idx, median in enumerate(best["medians"]):
                    for chrom in chroms:
                        for a, b in sorted(median.get(chrom, ())):
                            handle.write(f"{c_idx}\t{chrom}\t{a}\t{b}\n")


if __name__ == "__main__":
    main()
