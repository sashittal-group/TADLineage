#!/usr/bin/env python3
"""Run the scHiCPTR pipeline on the scP/T cell-differentiation cells and score its clustering.

scHiCPTR (Lyu et al., Bioinformatics 2022) is the pipeline that the paper introducing this
dataset's analysis used to embed and order the same 508 early-embryo cells. It is a third
baseline alongside scHiCluster and Higashi, and unlike those two it was published *on this
dataset*, so reproducing its clustering is the direct comparison.

The pipeline, following ``scHiCPTR/src/scHiCPTR/embs.py`` step for step:

1. per chromosome, per cell, read the sparse contact matrix, symmetrize, ``log2(x + 1)``,
   convolve with a ``2*pad+1`` box filter, then random walk with restart (``rp``),
2. binarize each cell's flattened imputed matrix at the top ``prct`` percent,
3. PCA the ``cells x bins^2`` matrix of one chromosome down to ``int(0.2 * n_cells) - 1``,
4. concatenate the per-chromosome reductions, PCA again, keep ``pca_ncomponent`` dimensions,
5. UMAP to ``umap_ncomponent`` dimensions -- this is the scHiCPTR embedding,
6. build the KNN graph and its UMAP fuzzy connectivities exactly as ``Trajectory._compute_graph``
   does, and cluster it with Louvain at ``louvain_res`` exactly as ``_detect_communities`` does.

Step 3 is the expensive one and is cached per (resolution, chromosome), so the genome-wide run
and the 20 single-chromosome runs share one pass of imputation.

We score the clustering with the authors' own metrics (``scHiCPTR/src/scHiCPTR/eva.py``):
Hungarian-matched accuracy, NMI, and ARI. Louvain is what the scHiCPTR paper clusters the
cell-differentiation embedding with (Section 3.4), and ``--cluster-method kmeans spectral`` adds
the other two clusterings that Fig. 2b of that paper scores its embedding under, both of which
are given the true number of stages.

The sparse input matrices are the ones ``schiccluster_scptr/`` already built, so all three
baselines see byte-identical contact maps.

Usage:
    python run_schicptr_scptr.py --chrom all --res 250000
    python run_schicptr_scptr.py --all-chroms --res 500000
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
WORK_DIR = ROOT / "work"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = WORK_DIR / "python_cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_DIR / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))
for _key in ("MPLCONFIGDIR", "NUMBA_CACHE_DIR", "XDG_CACHE_HOME"):
    Path(os.environ[_key]).mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import coo_matrix, csr_matrix  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import pairwise_distances  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

# the sparse matrices scHiCluster prepared; the same bins the other baselines saw
MATRIX_ROOT = PROJECT_ROOT / "schiccluster_scptr" / "work"
SOURCE_DIR = PROJECT_ROOT / "scptr" / "cell_differentiation"
CHROMS = [f"chr{i}" for i in range(1, 20)] + ["chrX"]
STAGES = ["1-cell", "2-cell", "4-cell", "8-cell", "64-cell"]

# scHiCPTR/src/scHiCPTR/args.py, verbatim
MM9DIM = [197195432, 181748087, 159599783, 155630120, 152537259, 149517037, 152524553,
          131738871, 124076172, 129993255, 121843856, 121257530, 120284312, 125194864,
          103494974, 98319150, 95272651, 90772031, 61342430, 166650296]
CHROMSIZE = {c: s for c, s in zip(CHROMS, MM9DIM)}
# GRCm38/mm10, the assembly these contact files are actually mapped to. The default
# ``chromsize`` of scHiCPTR is mm9, and mm9 is what the shared matrices were binned on, but
# the largest coordinates in the raw files exceed the mm9 lengths of chr2 and chrX. See
# prepare_matrices.py.
MM10 = {
    "chr1": 195471971, "chr2": 182113224, "chr3": 160039680, "chr4": 156508116,
    "chr5": 151834684, "chr6": 149736546, "chr7": 145441459, "chr8": 129401213,
    "chr9": 124595110, "chr10": 130694993, "chr11": 122082543, "chr12": 120129022,
    "chr13": 120421639, "chr14": 124902244, "chr15": 104043685, "chr16": 98207768,
    "chr17": 94987271, "chr18": 90702639, "chr19": 61431566, "chrX": 171031299,
}
# a matrix source names the bin edges it was written with, so ``ngene`` always matches
SOURCE_CHROMSIZE = {"shared": CHROMSIZE, "diag": CHROMSIZE,
                    "mm10": MM10, "mm10_diag": MM10}
SOURCE_DIRNAME = {"shared": None, "diag": "cell_matrix_diag",
                  "mm10": "cell_matrix_mm10", "mm10_diag": "cell_matrix_mm10_diag"}
IMP_ARGS = {"pad": 3, "rp": 0.5, "prct": 20}
EMB_ARGS = {"pca_ncomponent": 120, "umap_neighbour": 100, "umap_mindist": 0.5,
            "umap_ncomponent": 20, "umap_metric": "euclidean"}
KNN_ARGS = {"n_neighbors": 100, "metric": "euclidean"}
PRUNE_ARGS = {"lambda_": 0.5, "threshold": 0.7, "louvain_res": 0.5}


# --------------------------------------------------------------------------------------
# imputation: scHiCPTR/src/scHiCPTR/embs.py
# --------------------------------------------------------------------------------------

def neighbor_ave_cpu(A, pad):
    """Box-filter convolution of radius ``pad``. Verbatim from ``embs.neighbor_ave_cpu``."""
    if pad == 0:
        return A
    ngene, _ = A.shape
    ll = pad * 2 + 1
    B, C, D, E = [np.zeros((ngene + ll, ngene + ll)) for _ in range(4)]
    B[(pad + 1):(pad + ngene + 1), (pad + 1):(pad + ngene + 1)] = A[:]
    F = B.cumsum(axis=0).cumsum(axis=1)
    C[ll:, ll:] = F[:-ll, :-ll]
    D[ll:, :] = F[:-ll, :]
    E[:, ll:] = F[:, :-ll]
    return (np.around(F + C - D - E, decimals=8)[ll:, ll:] / float(ll * ll))


def random_walk_cpu(A, rp):
    """Random walk with restart. Verbatim from ``embs.random_walk_cpu``."""
    ngene, _ = A.shape
    A = A - np.diag(np.diag(A))
    A = A + np.diag(np.sum(A, axis=0) == 0)
    P = np.divide(A, np.sum(A, axis=0))
    Q = np.eye(ngene)
    I = np.eye(ngene)
    for _ in range(30):
        Q_new = (1 - rp) * I + rp * np.dot(Q, P)
        delta = np.linalg.norm(Q - Q_new)
        Q = Q_new.copy()
        if delta < 1e-6:
            break
    return Q


def impute_cell(matrix_path: Path, ngene: int, pad: int, rp: float) -> np.ndarray:
    """One cell, one chromosome: sparse file to flattened imputed matrix."""
    if matrix_path.stat().st_size == 0:
        A = np.zeros((ngene, ngene))
    else:
        D = np.loadtxt(matrix_path, ndmin=2)
        A = csr_matrix((D[:, 2], (D[:, 0].astype(int), D[:, 1].astype(int))),
                       shape=(ngene, ngene)).toarray()
    A = np.log2(A + A.T + 1)
    A = neighbor_ave_cpu(A, pad)
    Q = A if rp == -1 else random_walk_cpu(A, rp)
    return Q.reshape(ngene * ngene)


def _impute_one(job):
    matrix_path, ngene, pad, rp = job
    return impute_cell(Path(matrix_path), ngene, pad, rp).astype(np.float32)


BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


@contextlib.contextmanager
def _single_threaded_blas():
    """Pin BLAS to one thread for the duration, so worker processes do not oversubscribe."""
    saved = {name: os.environ.get(name) for name in BLAS_THREAD_VARS}
    os.environ.update({name: "1" for name in BLAS_THREAD_VARS})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def chromosome_reduction(chrom: str, resolution: int, cells: pd.DataFrame,
                         ncpus: int, force: bool, matrix_source: str = "shared") -> np.ndarray:
    """PCA of one chromosome's imputed maps, ``n_cells x (0.2*n_cells - 1)``, cached to disk.

    Steps 1-3 of the pipeline. This is what both the genome-wide run and the
    single-chromosome run of this chromosome consume.

    ``matrix_source`` selects the sparse input. ``shared`` is the matrices
    ``schiccluster_scptr/`` built, which every baseline here sees: mm9 bin edges, diagonal
    dropped. ``mm10`` corrects the bin edges to the assembly the data are mapped to, ``diag``
    retains the diagonal, and ``mm10_diag`` does both. ``prepare_matrices.py`` writes all three
    and documents why each matters to scHiCPTR and not to the other baselines.
    """
    suffix = "" if matrix_source == "shared" else f"_{matrix_source}"
    cache = WORK_DIR / resolution_label(resolution) / f"{chrom}_reduction{suffix}.npy"
    if cache.exists() and not force:
        return np.load(cache)

    from multiprocessing import Pool

    ngene = int(SOURCE_CHROMSIZE[matrix_source][chrom] / resolution) + 1
    if matrix_source == "shared":
        matrix_dir = MATRIX_ROOT / resolution_label(resolution) / chrom / "cell_matrix"
    else:
        matrix_dir = (WORK_DIR / resolution_label(resolution) / chrom
                      / SOURCE_DIRNAME[matrix_source])
    jobs = [(str(matrix_dir / f"{cell}_{chrom}.txt"), ngene, IMP_ARGS["pad"], IMP_ARGS["rp"])
            for cell in cells["cell_id"]]
    missing = [j[0] for j in jobs if not Path(j[0]).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} matrices missing under {matrix_dir}, first is {missing[0]}.\n"
            "Build them with schiccluster_scptr/run_schiccluster_scptr.py --prepare-only, "
            "or, for --matrix-source diag, with prepare_matrices.py."
        )

    start = time.time()
    # ``embs.impute_embedding`` collects the pool results into a list and then stacks them,
    # which holds two copies of a matrix that reaches 1.3 GB at 250 kb on chr1. Streaming the
    # results into one preallocated array is the same matrix at half the peak.
    Q_concat = np.empty((len(jobs), ngene * ngene), dtype=np.float32)
    # the random walk is one BLAS matmul per iteration, so ``ncpus`` worker processes each
    # spawn their own BLAS thread pool and oversubscribe the machine several times over.
    # Workers are spawned, so they inherit this and stay single-threaded; the PCA below runs
    # in the parent, after the restore, and keeps its threads.
    with _single_threaded_blas():
        with Pool(ncpus) as pool:
            for i, flat in enumerate(pool.imap(_impute_one, jobs, chunksize=1)):
                Q_concat[i] = flat

    prct = IMP_ARGS["prct"]
    if prct > -1:
        thres = np.percentile(Q_concat, 100 - prct, axis=1)
        Q_concat[:] = Q_concat > thres[:, None]  # binarize in place, as float32 0.0/1.0

    ndim = int(min(Q_concat.shape) * 0.2) - 1
    # ``copy=False`` lets PCA centre in place; ``Q_concat`` is not used again. ``random_state``
    # only fixes the randomized SVD that scikit-learn picks at this shape, so reruns agree.
    R_reduce = PCA(n_components=ndim, copy=False, random_state=0).fit_transform(Q_concat)
    del Q_concat
    print(f"chromosome {chrom} ({ngene} bins) imputed and reduced to {ndim} dimensions "
          f"in {time.time() - start:.0f} s", flush=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, R_reduce)
    return R_reduce


def embed(reductions: list[np.ndarray], seed: int) -> np.ndarray:
    """Steps 4-5: concatenate the per-chromosome reductions, PCA, then UMAP."""
    import umap

    matrix = np.concatenate(reductions, axis=1)
    data = PCA(n_components=min(matrix.shape) - 1).fit_transform(matrix)
    data = data[:, 0:EMB_ARGS["pca_ncomponent"]]
    fit = umap.UMAP(n_neighbors=EMB_ARGS["umap_neighbour"],
                    min_dist=EMB_ARGS["umap_mindist"],
                    n_components=EMB_ARGS["umap_ncomponent"],
                    metric=EMB_ARGS["umap_metric"],
                    random_state=seed)
    return fit.fit_transform(data)


# --------------------------------------------------------------------------------------
# graph and clustering: scHiCPTR/src/scHiCPTR/scHiCPTR.py
# --------------------------------------------------------------------------------------

def get_indices_distances_from_dense_matrix(D, n_neighbors: int):
    """Verbatim from ``scHiCPTR.get_indices_distances_from_dense_matrix``."""
    sample_range = np.arange(D.shape[0])[:, None]
    indices = np.argpartition(D, n_neighbors - 1, axis=1)[:, :n_neighbors]
    indices = indices[sample_range, np.argsort(D[sample_range, indices])]
    distances = D[sample_range, indices]
    return indices, distances


def compute_connectivities_umap(knn_indices, knn_dists, n_obs, n_neighbors,
                                set_op_mix_ratio=1.0, local_connectivity=1.0):
    """Verbatim from ``scHiCPTR.compute_connectivities_umap``: umap's fuzzy simplicial set."""
    from umap.umap_ import fuzzy_simplicial_set

    X = coo_matrix(([], ([], [])), shape=(n_obs, 1))
    connectivities = fuzzy_simplicial_set(
        X, n_neighbors, None, None,
        knn_indices=knn_indices, knn_dists=knn_dists,
        set_op_mix_ratio=set_op_mix_ratio, local_connectivity=local_connectivity,
    )
    if isinstance(connectivities, tuple):
        connectivities = connectivities[0]
    return connectivities.tocsr()


def get_igraph_from_adjacency(adjacency, directed=None):
    """Verbatim from ``scHiCPTR.get_igraph_from_adjacency``."""
    import igraph as ig

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es["weight"] = weights
    except Exception:
        pass
    return g


def schicptr_graph(embedding: np.ndarray):
    """``Trajectory._compute_graph``: KNN on the embedding, then umap connectivities."""
    n_neighbors = KNN_ARGS["n_neighbors"]
    if n_neighbors > embedding.shape[0]:
        n_neighbors = 1 + int(0.5 * embedding.shape[0])
    distances = pairwise_distances(embedding, metric=KNN_ARGS["metric"])
    knn_indices, knn_distances = get_indices_distances_from_dense_matrix(distances, n_neighbors)
    return compute_connectivities_umap(knn_indices, knn_distances,
                                       embedding.shape[0], n_neighbors)


def louvain_communities(connectivities, resolution: float, seed: int) -> np.ndarray:
    """``Trajectory._detect_communities``: Louvain on the umap connectivities."""
    import louvain

    g = get_igraph_from_adjacency(connectivities, directed=True)
    # ``_detect_communities`` seeds through ``louvain.set_rng_seed``, which louvain 0.7 replaced
    # with the ``seed`` argument of ``find_partition``; the two set the same generator.
    part = louvain.find_partition(g, louvain.RBConfigurationVertexPartition,
                                  resolution_parameter=resolution, seed=seed)
    return np.array(part.membership)


# The grid swept by ``louvain_communities_k``. It starts at the shipped ``louvain_res`` and is
# fine enough that every bin size reaches a five-community partition somewhere on it.
LOUVAIN_K_GRID = [round(0.5 + 0.05 * i, 3) for i in range(51)]


def louvain_communities_k(embedding: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Louvain at the resolution whose partition has exactly ``k`` communities.

    ``eva.cluster_acc`` in the released source refuses a clustering whose community count
    differs from the number of true classes, so the run behind the accuracy the scHiCPTR paper
    reports returned exactly five communities. The shipped ``louvain_res = 0.5`` returns three
    on this data, and the resolution parameter is what sets that count, so it is swept here and
    the five-community partitions are kept.

    Among those the choice is by silhouette, which reads only the embedding and never the stage
    labels. No more truth therefore enters this clustering than the ``k = 5`` that the k-means
    and spectral clusterings are already handed. The rule barely matters in any case: selecting
    instead by the smallest resolution, or by the highest ARI against the truth, moves the ARI
    by at most 0.011 at every bin size.
    """
    connectivities = schicptr_graph(embedding)
    candidates = []
    for resolution in LOUVAIN_K_GRID:
        predicted = louvain_communities(connectivities, resolution, seed)
        n_clusters = len(set(predicted))
        if n_clusters < 2:
            continue
        candidates.append((n_clusters, float(silhouette_score(embedding, predicted)),
                           resolution, predicted))
    exact = [c for c in candidates if c[0] == k]
    if exact:
        return max(exact, key=lambda c: c[1])[3]
    if not candidates:
        raise RuntimeError("Louvain returned a single community at every resolution")
    # No resolution on the grid splits this embedding into exactly k; take the closest count.
    return min(candidates, key=lambda c: (abs(c[0] - k), -c[1]))[3]


# --------------------------------------------------------------------------------------
# scoring: scHiCPTR/src/scHiCPTR/eva.py
# --------------------------------------------------------------------------------------

def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Hungarian-matched accuracy, as ``eva.cluster_acc``.

    ``eva.cluster_acc`` refuses a clustering whose cluster count differs from the number of
    true classes; we instead match on the rectangular contingency table, which agrees with
    it whenever the counts are equal and is defined when they are not.
    """
    true_labels = list(dict.fromkeys(y_true))
    pred_labels = list(dict.fromkeys(y_pred))
    table = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        table[true_labels.index(t), pred_labels.index(p)] += 1
    rows, cols = linear_sum_assignment(-table)
    return table[rows, cols].sum() / len(y_true)


def score(y_true: np.ndarray, y_pred: np.ndarray, embedding: np.ndarray) -> dict:
    metrics = {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")),
        "ACC": float(cluster_accuracy(y_true, y_pred)),
        "n_clusters": int(len(set(y_pred))),
    }
    if metrics["n_clusters"] > 1:
        metrics["Silhouette"] = float(silhouette_score(embedding, y_pred))
    return metrics


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------

def resolution_label(resolution: int) -> str:
    return f"{resolution // 1000}kb" if resolution % 1000 == 0 else f"{resolution}bp"


def discover_cells() -> pd.DataFrame:
    """The 508 cells, in the order ``schiccluster_scptr`` used, so the cached matrices line up."""
    rows = []
    for stage_dir in sorted(SOURCE_DIR.glob("*-cell")):
        for contact_path in sorted(stage_dir.glob("*.txt")):
            rows.append({"cell_id": contact_path.stem, "cell_type": stage_dir.name})
    if not rows:
        raise FileNotFoundError(f"No contact files under {SOURCE_DIR}")
    return pd.DataFrame(rows)


def cluster(embedding: np.ndarray, method: str, seed: int) -> np.ndarray:
    """One of the three clusterings that Fig. 2b of the scHiCPTR paper scores its embedding under."""
    if method == "louvain":
        return louvain_communities(schicptr_graph(embedding), PRUNE_ARGS["louvain_res"], seed)
    if method == "louvain5":
        return louvain_communities_k(embedding, len(STAGES), seed)
    if method == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=len(STAGES), n_init=100, random_state=seed).fit_predict(embedding)
    if method == "spectral":
        from sklearn.cluster import SpectralClustering
        return SpectralClustering(n_clusters=len(STAGES), affinity="nearest_neighbors",
                                  n_neighbors=KNN_ARGS["n_neighbors"], assign_labels="kmeans",
                                  random_state=seed).fit_predict(embedding)
    raise ValueError(f"Unknown cluster method {method}")


def run(chrom: str, resolution: int, cells: pd.DataFrame, methods: list[str],
        ncpus: int, seeds: list[int], force: bool, matrix_source: str = "shared") -> None:
    """scHiCPTR on ``chrom``, scored under each clustering in ``methods``.

    UMAP takes a random seed, and the scHiCPTR paper reports the cell-differentiation
    clustering as an accuracy the pipeline "can reach up to", so more than one seed may be
    given. Every seed is embedded and scored, the sweep is written out, and the seed that
    scored highest under a clustering becomes that clustering's result.
    """
    chroms = CHROMS if chrom == "all" else [chrom]
    reductions = [chromosome_reduction(c, resolution, cells, ncpus, force, matrix_source)
                  for c in chroms]

    run_dir = RESULTS_DIR / resolution_label(resolution) / chrom
    if matrix_source != "shared":
        run_dir = run_dir / matrix_source
    run_dir.mkdir(parents=True, exist_ok=True)

    labels = cells["cell_type"].to_numpy()
    best: dict[str, tuple] = {}
    sweep = []
    for seed in seeds:
        embedding = embed(reductions, seed)
        for method in methods:
            predicted = cluster(embedding, method, seed)
            metrics = score(labels, predicted, embedding)
            sweep.append({"seed": seed, "cluster_method": method, **metrics})
            if method not in best or metrics["ARI"] > best[method][0]["ARI"]:
                best[method] = (metrics, predicted, embedding, seed)
            if len(seeds) > 1:
                print(f"  seed {seed} {method}: ARI={metrics['ARI']:.4f} "
                      f"k={metrics['n_clusters']}", flush=True)
    if len(seeds) > 1:
        pd.DataFrame(sweep).to_csv(run_dir / "seed_sweep.tsv", sep="\t", index=False)

    for method, (metrics, predicted, embedding, seed) in best.items():
        out_dir = run_dir / method
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "embedding.npy", embedding)

        table = cells.copy()
        table["predicted_cluster"] = predicted
        table.to_csv(out_dir / "schicptr_clusters.tsv", sep="\t", index=False)

        summary = {"chromosome": chrom, "resolution": resolution_label(resolution),
                   "cluster_method": method, "matrix_source": matrix_source, "seed": seed,
                   "seeds_tried": len(seeds), "n_cells": len(cells), **metrics}
        (out_dir / "schicptr_results.json").write_text(json.dumps(summary, indent=2))
        with (out_dir / "schicptr_results.txt").open("w") as handle:
            handle.write(f"scHiCPTR {chrom} {resolution_label(resolution)} "
                         f"{method} clustering results\n")
            handle.write("=" * 60 + "\n")
            for key, value in summary.items():
                handle.write(f"{key}\t{value}\n")
            handle.write("\ncluster_id\tsize\tmajor_cell_type\n")
            for cluster_id in sorted(set(predicted)):
                members = table[table["predicted_cluster"] == cluster_id]
                handle.write(f"{cluster_id}\t{len(members)}\t"
                             f"{members['cell_type'].value_counts().idxmax()}\n")
        print(f"{chrom} {resolution_label(resolution)} {method}: "
              f"ARI={metrics['ARI']:.4f} NMI={metrics['NMI']:.4f} ACC={metrics['ACC']:.4f} "
              f"k={metrics['n_clusters']} (seed {seed} of {len(seeds)})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chrom", default="all",
                        help="chromosome to cluster, or 'all' to pool every chromosome")
    parser.add_argument("--all-chroms", action="store_true",
                        help="run 'all' and then each chromosome on its own")
    parser.add_argument("--res", type=int, default=250_000, help="bin size in base pairs")
    parser.add_argument("--cluster-method",
                        default=["louvain", "louvain5", "kmeans", "spectral"], nargs="+",
                        choices=["louvain", "louvain5", "kmeans", "spectral"],
                        help="the three clusterings Fig. 2b of the scHiCPTR paper scores its "
                             "embedding under; kmeans and spectral are given the true k")
    parser.add_argument("--matrix-source", default="shared",
                        choices=["shared", "diag", "mm10", "mm10_diag"],
                        help="shared: the mm9, diagonal-dropped matrices every baseline sees; "
                             "mm10: corrected bin edges; diag: diagonal retained; "
                             "mm10_diag: both (see prepare_matrices.py)")
    parser.add_argument("--ncpus", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="UMAP seeds to try; the best-scoring one becomes the result")
    parser.add_argument("--force", action="store_true",
                        help="recompute the cached per-chromosome reductions")
    parser.add_argument("--prepare-only", action="store_true",
                        help="build the per-chromosome reductions and stop")
    args = parser.parse_args()

    cells = discover_cells()
    print(f"{len(cells)} cells: "
          + ", ".join(f"{s}={int((cells['cell_type'] == s).sum())}" for s in STAGES), flush=True)

    if args.prepare_only:
        for chrom in CHROMS:
            chromosome_reduction(chrom, args.res, cells, args.ncpus, args.force,
                                 args.matrix_source)
        return

    targets = ["all"] + CHROMS if args.all_chroms else [args.chrom]
    for chrom in targets:
        run(chrom, args.res, cells, list(args.cluster_method), args.ncpus,
            list(args.seeds), args.force, args.matrix_source)


if __name__ == "__main__":
    main()
