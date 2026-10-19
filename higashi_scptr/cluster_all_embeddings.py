#!/usr/bin/env python3
"""Cluster genome-wide Higashi embeddings from cached per-chromosome runs."""

import argparse
import os
import pickle
from collections import Counter

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from run_paths import run_paths


CHROMS = [f"chr{i}" for i in range(1, 20)] + ["chrX"]
CELL_TYPE_ORDER = ["1-cell", "2-cell", "4-cell", "8-cell", "64-cell"]
RANDOM_STATE = 0


def configure_cache(out_dir):
    cache_dir = os.path.join(out_dir, "python_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(cache_dir, "matplotlib"))
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(cache_dir, "numba"))
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(cache_dir, "xdg"))
    for key in ("MPLCONFIGDIR", "NUMBA_CACHE_DIR", "XDG_CACHE_HOME"):
        os.makedirs(os.environ[key], exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=500000, help="Bin resolution in base pairs")
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=None,
        help="PCA dimensions before clustering. Defaults to 2 for <=250kb, 20 for 500kb, and 10 for >=1Mb.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=("pca", "pca-coverage"),
        default="pca-coverage",
        help="Use PCA features alone or PCA plus standardized log total coverage.",
    )
    parser.add_argument(
        "--cluster-method",
        choices=("spectral", "kmeans", "ward"),
        default="spectral",
        help="Clustering method for genome-wide Higashi features.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=None,
        help="Nearest neighbors for spectral clustering. Defaults to 6 for <=250kb and 25 otherwise.",
    )
    parser.add_argument("--kmeans-n-init", type=int, default=100, help="KMeans random restarts")
    return parser.parse_args()


def load_label_info(resolution, chrom):
    label_path = run_paths(chrom, resolution)["data_dir"]
    label_path = os.path.join(label_path, "label_info.pickle")
    with open(label_path, "rb") as handle:
        return pickle.load(handle)


def load_cell_embedding(resolution, chrom, expected_rows):
    embed_path = os.path.join(
        run_paths(chrom, resolution)["temp_dir"],
        "embed",
        "scptr_embed_0_origin.npy",
    )
    if not os.path.exists(embed_path):
        raise FileNotFoundError(embed_path)
    embedding = np.load(embed_path)
    if embedding.shape[0] != expected_rows:
        raise ValueError(
            f"{embed_path} has {embedding.shape[0]} rows, expected {expected_rows}. "
            "Use scptr_embed_0_origin.npy for cell embeddings; later indices are genomic-bin embeddings."
        )
    return embedding


def load_all_chrom_data(resolution):
    reference_labels = load_label_info(resolution, CHROMS[0])
    cell_names = reference_labels["cell_name_higashi"]
    cell_types = np.asarray(reference_labels["cell type"])
    embeddings = []
    coverage = []

    for chrom in CHROMS:
        labels = load_label_info(resolution, chrom)
        if labels["cell_name_higashi"] != cell_names:
            raise ValueError(f"Cell order for {chrom} does not match {CHROMS[0]}")
        embeddings.append(load_cell_embedding(resolution, chrom, len(cell_types)))
        coverage.append(np.asarray(labels["coverage"], dtype=float))

    return {
        "cell_names": np.asarray(cell_names),
        "cell_types": cell_types,
        "embedding": np.concatenate(embeddings, axis=1),
        "coverage": np.vstack(coverage).sum(axis=0),
    }


def default_pca_dim(resolution):
    if resolution <= 250000:
        return 2
    return 20 if resolution <= 500000 else 10


def default_n_neighbors(resolution):
    return 6 if resolution <= 250000 else 25


def make_features(raw_embedding, total_coverage, pca_dim, feature_mode):
    dim = min(pca_dim, raw_embedding.shape[0] - 1, raw_embedding.shape[1])
    pca_embedding = PCA(n_components=dim, random_state=RANDOM_STATE).fit_transform(raw_embedding)
    if feature_mode == "pca":
        features = StandardScaler().fit_transform(pca_embedding)
    elif feature_mode == "pca-coverage":
        depth = np.log1p(total_coverage)[:, None]
        features = StandardScaler().fit_transform(np.c_[pca_embedding, depth])
    else:
        raise ValueError(f"Unknown feature mode: {feature_mode}")
    return pca_embedding, features, dim


def cluster_features(features, n_clusters, method, n_neighbors, kmeans_n_init):
    if method == "spectral":
        return SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=min(n_neighbors, features.shape[0] - 1),
            random_state=RANDOM_STATE,
        ).fit_predict(features)
    if method == "kmeans":
        return KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=kmeans_n_init).fit_predict(features)
    if method == "ward":
        return AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(features)
    raise ValueError(f"Unknown cluster method: {method}")


def majority_sort_key(item):
    cell_type, count = item
    order_index = CELL_TYPE_ORDER.index(cell_type) if cell_type in CELL_TYPE_ORDER else len(CELL_TYPE_ORDER)
    return count, -order_index


def write_results(out_path, cell_names, cell_types, total_coverage, predicted, scores, config):
    with open(out_path, "w") as handle:
        for key, value in config.items():
            handle.write(f"{key}\t{value}\n")
        for name in ("ARI", "NMI", "Silhouette"):
            handle.write(f"{name}\t{scores[name]:.6f}\n")
        handle.write("\ncluster_id\tsize\tmajor_cell_type\n")
        for cluster_id in sorted(np.unique(predicted)):
            mask = predicted == cluster_id
            counts = Counter(cell_types[mask])
            majority, _ = max(counts.items(), key=majority_sort_key)
            handle.write(f"{cluster_id}\t{int(mask.sum())}\t{majority}\n")
        handle.write("\ncell_id\tpredicted_cluster\tcell_type\ttotal_coverage\n")
        for cell_name, pred, truth, cov in zip(cell_names, predicted, cell_types, total_coverage):
            handle.write(f"{cell_name}\t{pred}\t{truth}\t{int(cov)}\n")


def write_figure(out_path, pca_embedding, cell_types, predicted):
    import matplotlib.pyplot as plt

    encoder = LabelEncoder().fit(cell_types)
    label_ids = encoder.transform(cell_types)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=220, constrained_layout=True)

    # Plot one series per cell type so the panel carries a legend, keeping the
    # colors that a single scatter with c=label_ids and cmap="tab10" produces.
    cmap = plt.get_cmap("tab10")
    n_types = len(encoder.classes_)
    legend_order = [name for name in CELL_TYPE_ORDER if name in set(encoder.classes_)]
    legend_order += [name for name in encoder.classes_ if name not in set(legend_order)]
    for cell_type in legend_order:
        code = int(encoder.transform([cell_type])[0])
        mask = label_ids == code
        color = cmap(code / (n_types - 1)) if n_types > 1 else cmap(0.0)
        axes[0].scatter(
            pca_embedding[mask, 0],
            pca_embedding[mask, 1],
            color=color,
            s=14,
            linewidths=0,
            label=cell_type,
        )
    axes[0].set_title("Cell Type")
    axes[0].legend(
        markerscale=2.5,
        prop={"size": 9},
        loc="upper right",
        labelspacing=1.1,
        handletextpad=0.9,
        borderpad=0.7,
        framealpha=0.9,
    )

    axes[1].scatter(pca_embedding[:, 0], pca_embedding[:, 1], c=predicted, s=14, cmap="tab10", linewidths=0)
    axes[1].set_title("Higashi clustering")
    for ax in axes:
        ax.set_xlabel("Genome-wide PCA 1")
        ax.set_ylabel("Genome-wide PCA 2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    pca_dim = args.pca_dim or default_pca_dim(args.resolution)
    n_neighbors = args.n_neighbors or default_n_neighbors(args.resolution)
    paths = run_paths("all", args.resolution)
    os.makedirs(paths["results_dir"], exist_ok=True)
    os.makedirs(paths["temp_dir"], exist_ok=True)
    configure_cache(paths["temp_dir"])

    data = load_all_chrom_data(args.resolution)
    pca_embedding, features, used_dim = make_features(
        data["embedding"],
        data["coverage"],
        pca_dim,
        args.feature_mode,
    )

    y_true = LabelEncoder().fit_transform(data["cell_types"])
    predicted = cluster_features(
        features,
        len(np.unique(y_true)),
        args.cluster_method,
        n_neighbors,
        args.kmeans_n_init,
    )
    scores = {
        "ARI": adjusted_rand_score(y_true, predicted),
        "NMI": normalized_mutual_info_score(y_true, predicted),
        "Silhouette": silhouette_score(features, predicted) if len(np.unique(predicted)) > 1 else float("nan"),
    }

    np.save(os.path.join(paths["temp_dir"], f"all_chrom_pca{used_dim}.npy"), pca_embedding)
    np.save(os.path.join(paths["temp_dir"], f"all_chrom_features_pca{used_dim}_{args.feature_mode}.npy"), features)

    config = {
        "resolution": args.resolution,
        "chromosomes": ",".join(CHROMS),
        "source_embedding": "scptr_embed_0_origin.npy",
        "pca_dim": used_dim,
        "feature_mode": args.feature_mode,
        "cluster_method": args.cluster_method,
        "n_neighbors": n_neighbors,
    }
    results_path = os.path.join(paths["results_dir"], "cluster_results.txt")
    figure_path = os.path.join(paths["results_dir"], "cluster_figure.png")
    write_results(
        results_path,
        data["cell_names"],
        data["cell_types"],
        data["coverage"],
        predicted,
        scores,
        config,
    )
    write_figure(figure_path, pca_embedding, data["cell_types"], predicted)

    print(f"Wrote {results_path}")
    print(f"Wrote {figure_path}")
    print(f"ARI={scores['ARI']:.4f} NMI={scores['NMI']:.4f} silhouette={scores['Silhouette']:.4f}")


if __name__ == "__main__":
    main()
