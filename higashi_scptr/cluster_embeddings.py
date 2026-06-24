#!/usr/bin/env python3
"""Cluster Higashi cell embeddings for one chromosome/resolution run."""

import os
import pickle

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import LabelEncoder

from run_paths import parse_run_args, run_paths


CLUSTER_COUNT = 5
RANDOM_STATE = 0


def load_label_info(data_dir):
    labels_path = os.path.join(data_dir, "label_info.pickle")
    with open(labels_path, "rb") as handle:
        return pickle.load(handle)


def candidate_embedding_paths(embed_dir):
    return [
        os.path.join(embed_dir, name)
        for name in sorted(os.listdir(embed_dir))
        if name.endswith("_origin.npy")
    ]


def load_embeddings(data_dir, embed_dir):
    labels = load_label_info(data_dir)
    label_count = len(labels["cell type"])
    arrays = [np.load(path) for path in candidate_embedding_paths(embed_dir)]

    if not arrays:
        raise FileNotFoundError(f"No embedding .npy files found in {embed_dir}")

    matching = [arr for arr in arrays if arr.shape[0] == label_count]
    if matching:
        return matching[0]
    if len(arrays) == 1:
        return arrays[0]
    raise ValueError(f"No embedding array matches label count {label_count}: {[arr.shape for arr in arrays]}")


def cluster_embeddings(embeddings, true_labels):
    y_true_int = LabelEncoder().fit_transform(true_labels)
    clusterer = KMeans(n_clusters=CLUSTER_COUNT, random_state=RANDOM_STATE, n_init=10)
    y_pred = clusterer.fit_predict(embeddings)

    scores = {
        "ARI": adjusted_rand_score(y_true_int, y_pred),
        "NMI": normalized_mutual_info_score(y_true_int, y_pred),
        "Silhouette": silhouette_score(embeddings, y_pred)
        if len(np.unique(y_pred)) > 1
        else float("nan"),
    }
    return y_pred, scores


def validate_shapes(embeddings, labels):
    if embeddings.shape[0] != len(labels):
        raise ValueError(f"Embedding rows ({embeddings.shape[0]}) do not match labels ({len(labels)})")


def write_cluster_results(out_path, y_pred, y_true, scores):
    with open(out_path, "w") as handle:
        for name in ("ARI", "NMI", "Silhouette"):
            handle.write(f"{name}\t{scores[name]:.6f}\n")
        handle.write("predicted_cluster\tcell_type\n")
        for pred, truth in zip(y_pred, y_true):
            handle.write(f"{pred}\t{truth}\n")


def main():
    args = parse_run_args("Cluster Higashi cell embeddings for one chromosome and resolution.")
    paths = run_paths(args.chromosome, args.resolution)
    data_dir = paths["data_dir"]
    embed_dir = os.path.join(paths["temp_dir"], "embed")
    results_dir = paths["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    labels = load_label_info(data_dir)
    y_true = np.asarray(labels["cell type"])
    embeddings = load_embeddings(data_dir, embed_dir)
    validate_shapes(embeddings, y_true)

    # This is the non-compartment clustering path: KMeans runs directly on Higashi embeddings.
    y_pred, scores = cluster_embeddings(embeddings, y_true)

    out_path = os.path.join(results_dir, "cluster_results.txt")
    write_cluster_results(out_path, y_pred, y_true, scores)

    print(f"Wrote {out_path}")
    print(
        f"ARI={scores['ARI']:.4f} "
        f"NMI={scores['NMI']:.4f} "
        f"silhouette={scores['Silhouette']:.4f}"
    )


if __name__ == "__main__":
    main()
