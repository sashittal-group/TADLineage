#!/usr/bin/env python3
"""The frozen 1 Mb pipeline of ../run_pipeline.py at another bin size; no stage labels.

For bin size R (250 kb, 500 kb; 1 Mb is used only to check this script against ../final):

  1. bin the raw 250 kb contacts of schicptr_scptr/work/250kb at R (as ../extract.py does at 1 Mb)
  2. embedding: sqrt of total-normalized off-diagonal cis contacts of all chromosomes, PCA 30,
     first 10 components. PCA is computed from the centred cell-by-cell Gram matrix, which gives
     the same scores as ../signals.pca without holding the dense 250 kb matrix
  3. W = 0.05 I + 0.95 P^5 on the 10-neighbour graph (../hierarchy.diffusion_weights)
  4. pooled maps, Gaussian smoothing sigma 0.7 bins, contiguous hierarchies spanning 2-40 bins
     (../hierarchy.adjacent_tree, unchanged; at 250 kb that is 0.5-10 Mb)

Writes {R}/genome/ (all chromosomes, one graph) and {R}/chromosome_only/{chrom}/ (that
chromosome's contacts, graph and hierarchies alone), as ../per_chromosome.py does at 1 Mb.

--graph-from-1mb writes {R}/genome_graph1mb/ instead: cell similarity (the neighbour graph W and
the embedding behind the spectral start) is taken from the same contacts binned at 1 Mb, i.e.
../final/fit.npz, and only the imputed maps and hierarchies are at R. Single-cell contacts at
250/500 kb are too sparse for a reliable 10-PC cell embedding; binning them at 1 Mb pools them.

    python build.py --resolution 500000
    python build.py --resolution 500000 --graph-from-1mb
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
REPO_ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(EXPERIMENT))

from extract import CHROMS, SIZES  # noqa: E402
from hierarchy import adjacent_tree, diffusion_weights  # noqa: E402

SOURCE = REPO_ROOT / "schicptr_scptr/work/250kb"
SOURCE_BP = 250_000


def cell_names():
    """Cell file stems in the hash order of ../extract.py, checked against ../data/cells.json."""
    files = (SOURCE / "chr1/cell_matrix_mm10").glob("*.txt")
    names = sorted([p.name.removesuffix("_chr1.txt") for p in files],
                   key=lambda s: hashlib.sha256(s.split("_")[0].encode()).hexdigest())
    ids = json.loads((EXPERIMENT / "data/cells.json").read_text())
    assert [s.split("_")[0] for s in names] == ids, "cell order differs from ../data/cells.json"
    return names


def load_counts(names, chrom, resolution, cache):
    """Upper-triangular counts of one chromosome at R, as a cell x (nb*nb) sparse matrix."""
    path = cache / f"{chrom}.npz"
    nb = (SIZES[CHROMS.index(chrom)] + resolution - 1) // resolution
    if path.exists():
        return sparse.load_npz(path).tocsr(), nb
    factor = resolution // SOURCE_BP
    rows, cols, vals = [], [], []
    for i, name in enumerate(names):
        arr = np.loadtxt(SOURCE / chrom / "cell_matrix_mm10" / f"{name}_{chrom}.txt", ndmin=2)
        if not len(arr):
            continue
        a = arr[:, 0].astype(np.int64) // factor
        b = arr[:, 1].astype(np.int64) // factor
        rows.append(np.full(len(a), i))
        cols.append(a * nb + b)
        vals.append(arr[:, 2])
    m = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(names), nb * nb))
    m.sum_duplicates()
    sparse.save_npz(path, m)
    return m, nb


def off_diagonal(counts, nb):
    """Keep entries a < b (the frozen pipeline drops the diagonal of the binned map)."""
    coo = counts.tocoo()
    keep = (coo.col // nb) < (coo.col % nb)
    return sparse.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=counts.shape)


def embedding(blocks, n_components=30, keep=10):
    """PCA scores of sqrt(total-normalized contacts) via the centred Gram matrix."""
    total = sum(np.asarray(b.sum(axis=1)).ravel() for b in blocks)
    scale = sparse.diags(1.0 / np.maximum(total, 1))
    gram = np.zeros((blocks[0].shape[0],) * 2)
    for b in blocks:
        x = (scale @ b).sqrt()
        gram += (x @ x.T).toarray()
    n = len(gram)
    centre = np.eye(n) - 1.0 / n
    vals, vecs = np.linalg.eigh(centre @ gram @ centre)
    order = np.argsort(vals)[::-1][:n_components]
    return (vecs[:, order] * np.sqrt(np.maximum(vals[order], 0)))[:, :keep]


def hierarchies(weights, chrom_counts, chunk=64):
    """Contiguous hierarchies of every cell from its W-pooled, smoothed map (../hierarchy)."""
    n = weights.shape[0]
    cells = [[] for _ in range(n)]
    for chrom, (counts, nb) in chrom_counts.items():
        total = np.asarray(counts.sum(axis=1)).ravel()
        flat = sparse.diags(1.0 / np.maximum(total, 1)) @ counts
        flat_t = flat.T.tocsr()
        for start in range(0, n, chunk):
            pooled = (flat_t @ weights[start:start + chunk].T).T      # chunk x nb*nb, dense
            for offset, row in enumerate(np.asarray(pooled)):
                m = row.reshape(nb, nb)
                m = m + m.T
                np.fill_diagonal(m, 0)
                m = gaussian_filter(m, sigma=0.7)
                cells[start + offset].extend((chrom, int(a), int(b)) for a, b in adjacent_tree(m))
    features = sorted({t for cell in cells for t in cell})
    lookup = {t: j for j, t in enumerate(features)}
    rows = [i for i, cell in enumerate(cells) for _ in cell]
    cols = [lookup[t] for cell in cells for t in cell]
    trees = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(features)))
    return trees, features


def save(out, trees, intervals, x, weights):
    out.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out / "trees.npz", trees)
    (out / "intervals.json").write_text(json.dumps(intervals))
    np.save(out / "embedding.npy", x)
    np.save(out / "weights.npy", weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", type=int, required=True, help="bin size in bp")
    parser.add_argument("--embedding-only", action="store_true", help="stop after the embedding (for checks)")
    parser.add_argument("--graph-from-1mb", action="store_true",
                        help="neighbour graph and embedding from the 1 Mb pipeline, hierarchies at R")
    args = parser.parse_args()
    root = HERE / str(args.resolution)
    cache = root / "counts"
    cache.mkdir(parents=True, exist_ok=True)
    names = cell_names()

    chrom_counts = {}
    for chrom in CHROMS:
        chrom_counts[chrom] = load_counts(names, chrom, args.resolution, cache)
        print(f"binned {chrom}", flush=True)
    if args.graph_from_1mb:
        out = root / "genome_graph1mb"
        if not (out / "trees.npz").exists():
            one_mb = np.load(EXPERIMENT / "final/fit.npz")
            trees, intervals = hierarchies(one_mb["weights"], chrom_counts)
            save(out, trees, intervals, one_mb["embedding"], one_mb["weights"])
            print(f"genome, 1 Mb graph: {trees.sum() / trees.shape[0]:.0f} domains per cell", flush=True)
        return
    x = embedding([off_diagonal(c, nb) for c, nb in chrom_counts.values()])
    np.save(root / "embedding.npy", x)
    if args.embedding_only:
        return

    if not (root / "genome/trees.npz").exists():
        weights = diffusion_weights(x, 10, 0.05, 5)
        trees, intervals = hierarchies(weights, chrom_counts)
        save(root / "genome", trees, intervals, x, weights)
        print(f"genome: {trees.shape[1]} distinct domains, {trees.sum() / trees.shape[0]:.0f} per cell", flush=True)

    for chrom in CHROMS:
        out = root / "chromosome_only" / chrom
        if (out / "trees.npz").exists():
            continue
        counts, nb = chrom_counts[chrom]
        xc = embedding([off_diagonal(counts, nb)])
        weights = diffusion_weights(xc, 10, 0.05, 5)
        trees, intervals = hierarchies(weights, {chrom: (counts, nb)})
        save(out, trees, intervals, xc, weights)
        print(f"chromosome-only {chrom}: {trees.sum() / trees.shape[0]:.0f} domains per cell", flush=True)


if __name__ == "__main__":
    main()
