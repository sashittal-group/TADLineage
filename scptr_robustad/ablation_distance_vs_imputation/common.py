"""Paths and inputs shared by the ablation scripts; nothing here reads stage labels.

Everything upstream is reused read-only from ../results_unsupervised_median (UM below):

  imputed hierarchies   the exact trees behind the paper's Fig. 3
                          genome-wide  1 Mb: UM/final            250/500 kb: UM/multires/{R}/genome_graph1mb
                          one chrom    1 Mb: UM/per_chromosome/chromosome_only/{chrom}
                                       250/500 kb: UM/multires/{R}/chromosome_only/{chrom}
  raw hierarchies       built here by build_raw.py: same tree builder, smoothing and span limits,
                        but every cell's map is its own (W = I), no pooling across cells
  spectral start        the paper's: 10-neighbour spectral clustering of the contact embedding
                        (genome-wide: UM/final/fit.npz, the 1 Mb embedding used at every bin size;
                        one chromosome: that chromosome's embedding)
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
UM = HERE.parent / "results_unsupervised_median"
REPO_ROOT = HERE.parents[1]
for path in (UM, UM / "nussinov_median", UM / "multires"):
    sys.path.insert(0, str(path))

from extract import CHROMS  # noqa: E402

RESOLUTIONS = [250_000, 500_000, 1_000_000]
LABEL = {250_000: "250 kb", 500_000: "500 kb", 1_000_000: "1 Mb"}
K, CB = 5, 1.0
RANDOM_STARTS = 20          # k-means++ starts under the TAD distance, seeds 1..20
PAPER_RANDOM_STARTS = 4     # the paper's fits used spectral + seeds 1..4


def imputed_folder(resolution, scope):
    if scope == "all":
        return UM / "final" if resolution == 1_000_000 else UM / f"multires/{resolution}/genome_graph1mb"
    if resolution == 1_000_000:
        return UM / "per_chromosome/chromosome_only" / scope
    return UM / f"multires/{resolution}/chromosome_only/{scope}"


def raw_folder(resolution):
    return HERE / "hierarchies" / str(resolution) / "raw"


def embedding(resolution, scope):
    """The contact embedding behind the paper's spectral start for this fit."""
    if scope == "all":
        return np.load(UM / "final/fit.npz")["embedding"]
    if resolution != 1_000_000:
        return np.load(UM / f"multires/{resolution}/chromosome_only/{scope}/embedding.npy")
    # as UM/nussinov_median/per_chromosome_nussinov.py rebuilds it
    from signals import normalize, pca
    counts = np.load(UM / f"data/{scope}_counts.npz")["counts"]
    a, b = np.triu_indices(counts.shape[1], 1)
    return pca(np.sqrt(normalize(counts[:, a, b])), 30)[:, :10]


def run_folder(resolution, scope, hierarchy):
    return HERE / "runs" / str(resolution) / scope / hierarchy
