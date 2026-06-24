#!/usr/bin/env python3
"""Plot non-compartment Higashi embedding clusters."""

import os
import pickle
from collections import Counter
from matplotlib import pyplot as plt

import numpy as np
from sklearn.decomposition import PCA

from run_paths import parse_run_args, run_paths


CELL_TYPE_ORDER = ["1-cell", "2-cell", "4-cell", "8-cell", "64-cell"]
FIGURE_DPI = 220


def configure_cache(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(temp_dir, "matplotlib_cache"))
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(temp_dir, "numba_cache"))
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(temp_dir, "python_cache", "xdg"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)


def load_label_info(data_dir):
    labels_path = os.path.join(data_dir, "label_info.pickle")
    with open(labels_path, "rb") as handle:
        return pickle.load(handle)


def load_embeddings(data_dir, embed_dir):
    label_count = len(load_label_info(data_dir)["cell type"])
    candidates = []

    for name in sorted(os.listdir(embed_dir)):
        if not name.endswith("_origin.npy"):
            continue
        path = os.path.join(embed_dir, name)
        arr = np.load(path)
        if arr.shape[0] == label_count:
            candidates.append(arr)

    if not candidates:
        raise FileNotFoundError("Could not find a cell embedding matrix matching the label count")
    return candidates[0]


def encode_cell_types(cell_types):
    cell_types = np.asarray(cell_types)
    known = [name for name in CELL_TYPE_ORDER if name in set(cell_types)]
    extra = sorted(set(cell_types) - set(known))
    classes = known + extra
    class_to_idx = {name: index for index, name in enumerate(classes)}
    return np.asarray([class_to_idx[name] for name in cell_types]), classes


def load_predictions(results_path):
    predictions = []
    truths = []
    with open(results_path, "r") as handle:
        for line in handle:
            if is_metric_or_header(line):
                continue
            pred, truth = line.strip().split("\t")
            predictions.append(int(pred))
            truths.append(truth)
    return np.asarray(predictions), np.asarray(truths)


def is_metric_or_header(line):
    return line.startswith(("ARI", "NMI", "Silhouette", "predicted_cluster"))


def majority_cluster_labels(y_pred, y_true_names):
    labels = {}
    for cluster in sorted(np.unique(y_pred)):
        cluster_truth = y_true_names[y_pred == cluster]
        counts = Counter(cluster_truth)
        majority, _ = max(counts.items(), key=majority_sort_key)
        labels[cluster] = f"{cluster}_{majority}"
    return labels


def majority_sort_key(item):
    cell_type, count = item
    order_index = CELL_TYPE_ORDER.index(cell_type) if cell_type in CELL_TYPE_ORDER else len(CELL_TYPE_ORDER)
    return count, -order_index


def legend_handles(labels, cmap):
    return [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=cmap(index),
            markersize=8,
        )
        for index, label in labels
    ]


def plot_cluster_view(embeddings, y_true, true_names, y_pred, y_true_names, title, out_path):
    # Matplotlib needs the run-specific cache paths before pyplot is imported.


    # PCA gives a stable two-dimensional view without depending on UMAP runtime/cache state.
    xy = PCA(n_components=2, random_state=0).fit_transform(embeddings)
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=FIGURE_DPI)
    cmap_true = plt.get_cmap("tab10", len(unique_true))
    cmap_pred = plt.get_cmap("tab10", len(unique_pred))

    axes[0].scatter(xy[:, 0], xy[:, 1], c=y_true, s=14, cmap=cmap_true, alpha=0.9, linewidths=0)
    axes[0].set_title("True cell type")
    axes[0].set_xlabel("PCA 1")
    axes[0].set_ylabel("PCA 2")

    axes[1].scatter(xy[:, 0], xy[:, 1], c=y_pred, s=14, cmap=cmap_pred, alpha=0.9, linewidths=0)
    axes[1].set_title("KMeans clusters on Higashi embeddings")
    axes[1].set_xlabel("PCA 1")
    axes[1].set_ylabel("PCA 2")

    axes[0].legend(
        handles=legend_handles([(index, true_names[index]) for index in unique_true], cmap_true),
        loc="best",
        frameon=False,
    )

    cluster_labels = majority_cluster_labels(y_pred, y_true_names)
    axes[1].legend(
        handles=legend_handles([(index, cluster_labels[index]) for index in unique_pred], cmap_pred),
        title="Cluster_majority",
        loc="best",
        frameon=False,
    )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_run_args("Plot Higashi clustering results for one chromosome and resolution.")
    paths = run_paths(args.chromosome, args.resolution)
    temp_dir = paths["temp_dir"]
    data_dir = paths["data_dir"]
    embed_dir = os.path.join(temp_dir, "embed")
    results_dir = paths["results_dir"]
    results_path = os.path.join(results_dir, "cluster_results.txt")
    out_path = os.path.join(results_dir, "cluster_figure.png")

    configure_cache(temp_dir)
    os.makedirs(results_dir, exist_ok=True)

    labels = load_label_info(data_dir)
    y_true_names = np.asarray(labels["cell type"])
    y_true, true_names = encode_cell_types(y_true_names)
    y_pred, result_truth_names = load_predictions(results_path)
    embeddings = load_embeddings(data_dir, embed_dir)

    if len(y_pred) != len(y_true_names):
        raise ValueError(f"Prediction count ({len(y_pred)}) does not match labels ({len(y_true_names)})")

    if len(result_truth_names) == len(y_true_names) and np.any(result_truth_names != y_true_names):
        y_true_names = result_truth_names

    title = f"Higashi scptr {paths['chromosome']} {paths['resolution_label']}bp cluster view"
    plot_cluster_view(embeddings, y_true, true_names, y_pred, y_true_names, title, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
