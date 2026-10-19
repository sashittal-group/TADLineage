"""Fit per-chromosome medians with shared or chromosome-only imputation graphs.

No evaluation labels are read. Hyperparameters match the frozen k=5 experiment.
"""
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.cluster import SpectralClustering
from hierarchy import diffusion_weights, infer_hierarchies, median_em
from signals import pca, normalize
from extract import CHROMS

HERE = Path(__file__).resolve().parent

def fit_chromosome(trees, embedding, out, mode, chrom, intervals):
    out.mkdir(parents=True, exist_ok=True)
    initial = SpectralClustering(
        5, affinity='nearest_neighbors', n_neighbors=10,
        assign_labels='cluster_qr', random_state=0,
    ).fit_predict(embedding)
    solutions = []
    for seed in range(20):
        y, medians, history = median_em(
            trees, k=5, initial=initial if seed == 0 else None, seed=seed,
        )
        solutions.append((history[-1], y, medians, history))
    objective, y, medians, history = min(solutions, key=lambda s: s[0])
    assert np.all(np.diff(history) <= 1e-8)
    for c in range(5):
        if np.any(y == c):
            expected = np.asarray(trees[y == c].sum(axis=0)).ravel() > np.sum(y == c) / 2
            np.testing.assert_array_equal(medians[c], expected)
    np.savez_compressed(out / 'fit.npz', labels=y, medians=medians,
                        restart_labels=np.array([s[1] for s in solutions]))
    sparse.save_npz(out / 'trees.npz', trees)
    (out / 'intervals.json').write_text(json.dumps(intervals))
    row = dict(chromosome=chrom, mode=mode, n_cells=trees.shape[0], k=5,
               resolution_bp=1000000, neighbors=10, diffusion_steps=5,
               self_weight=.05, sigma=.7, restarts=20, objective=objective,
               history=history, restart_objectives=[s[0] for s in solutions],
               cluster_sizes=np.bincount(y, minlength=5).tolist(),
               stage_labels_used_in_fit=False,
               graph_chromosomes=CHROMS if mode == 'shared_graph' else [chrom],
               initialization='spectral graph + 19 random RF kmeans++ starts')
    (out / 'run.json').write_text(json.dumps(row, indent=2))
    print(f'{mode} {chrom}: objective={objective}, sizes={row["cluster_sizes"]}', flush=True)

def main():
    out = HERE / 'per_chromosome'
    full = sparse.load_npz(HERE / 'final/trees.npz')
    intervals = json.loads((HERE / 'final/intervals.json').read_text())
    embedding = np.load(HERE / 'final/fit.npz')['embedding']
    for chrom in CHROMS:
        shared = out / 'shared_graph' / chrom
        if not (shared / 'run.json').exists():
            columns = [i for i, t in enumerate(intervals) if t[0] == chrom]
            fit_chromosome(full[:, columns], embedding, shared, 'shared_graph', chrom,
                           [intervals[i] for i in columns])
        local = out / 'chromosome_only' / chrom
        if not (local / 'run.json').exists():
            counts = np.load(HERE / f'data/{chrom}_counts.npz')['counts']
            a, b = np.triu_indices(counts.shape[1], 1)
            x = pca(np.sqrt(normalize(counts[:, a, b])), 30)[:, :10]
            weights = diffusion_weights(x, 10, .05, 5)
            trees, local_intervals = infer_hierarchies(weights, sigma=.7, chroms=[chrom])
            fit_chromosome(trees, x, local, 'chromosome_only', chrom, local_intervals)
            np.save(local / 'weights.npy', weights)

if __name__ == '__main__':
    main()
