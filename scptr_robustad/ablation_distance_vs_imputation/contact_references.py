#!/usr/bin/env python3
"""Reference clusterings that never build a TAD hierarchy; no stage labels are read.

All genome-wide on the 1 Mb contacts behind UM/final (the cell graph every genome-wide fit in
the paper uses, whatever the bin size of its hierarchies):

  embedding_spectral        the paper's spectral start itself (sqrt-normalized contacts, PCA 10,
                            10-neighbour spectral clustering)
  embedding_kmeans          k-means on the same 10-PC embedding
  imputed_contacts_spectral contacts pooled with the paper's W (= the maps the imputed
  imputed_contacts_kmeans   hierarchies are built from), then sqrt, PCA 10, and either clustering

    python contact_references.py
"""

import json

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering

from common import HERE, K, RANDOM_STARTS, UM
from signals import normalize, pca


def spectral(x):
    return SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=10,
                              assign_labels="cluster_qr", random_state=0).fit_predict(x)


def kmeans(x):
    return KMeans(K, n_init=RANDOM_STARTS + 1, random_state=0).fit_predict(x)


def main():
    out = HERE / "runs/contact_references"
    out.mkdir(parents=True, exist_ok=True)
    final = np.load(UM / "final/fit.npz")
    x, w = final["embedding"], final["weights"]
    contacts = normalize(np.load(UM / "data/features.npz")["contacts"])
    imputed = pca(np.sqrt(w @ contacts), 30)[:, :10]
    labels = {"embedding_spectral": spectral(x), "embedding_kmeans": kmeans(x),
              "imputed_contacts_spectral": spectral(imputed), "imputed_contacts_kmeans": kmeans(imputed)}
    np.savez_compressed(out / "labels.npz", **labels)
    (out / "run.json").write_text(json.dumps({"runs": list(labels), "k": K,
                                              "stage_labels_used_in_fit": False}, indent=2))
    print("wrote", ", ".join(labels))


if __name__ == "__main__":
    main()
