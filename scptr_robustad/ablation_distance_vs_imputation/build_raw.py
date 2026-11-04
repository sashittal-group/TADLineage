#!/usr/bin/env python3
"""Per-cell hierarchies WITHOUT imputation; no stage labels are read.

The paper's hierarchies come from W-pooled maps, W = 0.05 I + 0.95 P^5 on the 10-neighbour
contact graph, so 95% of every cell's map is borrowed from its graph neighbours. Here W = I:
each cell's own total-normalized map, then the same Gaussian smoothing (sigma 0.7 bins) and the
same contiguous tree builder with the same 2-40 bin span limits (UM/multires/build.hierarchies,
unchanged). Only the pooling is removed.

Counts: 250/500 kb from the binned cache in UM/multires/{R}/counts (read only); 1 Mb from
UM/data/{chrom}_counts.npz, the counts behind UM/final.

--check rebuilds the paper's imputed 1 Mb hierarchies the same way (with UM/final's W) and
compares them with UM/final/trees.npz, to show that only W differs between the two.

    python build_raw.py --resolution 1000000 --check
"""

import argparse
import json

import numpy as np
from scipy import sparse

from common import CHROMS, UM, raw_folder
from build import cell_names, hierarchies, load_counts, save


def counts_1mb(chrom):
    dense = np.load(UM / f"data/{chrom}_counts.npz")["counts"]
    n, nb, _ = dense.shape
    return sparse.csr_matrix(dense.reshape(n, nb * nb).astype(float)), nb


def chrom_counts(resolution):
    if resolution == 1_000_000:
        return {chrom: counts_1mb(chrom) for chrom in CHROMS}
    names = cell_names()
    cache = UM / f"multires/{resolution}/counts"
    assert all((cache / f"{c}.npz").exists() for c in CHROMS), f"missing count cache {cache}"
    return {chrom: load_counts(names, chrom, resolution, cache) for chrom in CHROMS}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    counts = chrom_counts(args.resolution)
    n = next(iter(counts.values()))[0].shape[0]

    if args.check:
        assert args.resolution == 1_000_000, "--check compares with UM/final, which is 1 Mb"
        trees, intervals = hierarchies(np.load(UM / "final/fit.npz")["weights"], counts)
        ref = sparse.load_npz(UM / "final/trees.npz").tocsr()
        ref_intervals = [tuple(t) for t in json.loads((UM / "final/intervals.json").read_text())]
        same = (intervals == ref_intervals and trees.shape == ref.shape and (trees != ref).nnz == 0)
        print(f"rebuilt imputed 1 Mb hierarchies identical to UM/final: {same}")
        if not same:
            raise SystemExit(1)

    out = raw_folder(args.resolution)
    trees, intervals = hierarchies(np.eye(n), counts)
    save(out, trees, intervals, np.zeros((n, 0)), np.eye(n))
    (out / "weights.npy").unlink()      # W = I; nothing worth storing
    (out / "embedding.npy").unlink()    # the spectral start comes from common.embedding
    print(f"raw {args.resolution}: {trees.shape[1]} distinct domains, "
          f"{trees.sum() / n:.0f} per cell")


if __name__ == "__main__":
    main()
