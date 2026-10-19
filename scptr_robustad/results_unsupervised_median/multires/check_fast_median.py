#!/usr/bin/env python3
"""fast_median must reach the same optimal cost as nussinov.infer_tads_with_dp.

1. random small laminar leaf sets (many shapes, Cb in {0, 0.5, 1, 2})
2. the real 1 Mb clusters of ../nussinov_median on chr19 and chr1
Also checks the returned median is laminar and that its recomputed cost equals the DP value.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
sys.path.insert(0, str(EXPERIMENT.parents[1]))
from nussinov import infer_tads_with_dp  # noqa: E402
from fast_median import median  # noqa: E402


def cost(M, cells, cb):
    total = 0.0
    for T in cells:
        total += len(M ^ T)
        total += cb * len({a for a, _ in M} ^ {a for a, _ in T})
        total += cb * len({b for _, b in M} ^ {b for _, b in T})
    return total


def laminar(M):
    return not any(a < c < b < d for a, b in M for c, d in M)


def random_laminar(rng, n, size):
    out = set()
    for _ in range(size * 4):
        a, b = sorted(rng.sample(range(1, n + 1), 2))
        if laminar(out | {(a, b)}):
            out.add((a, b))
        if len(out) >= size:
            break
    return out


def compare(cells, cb):
    pos = {p for T in cells for iv in T for p in iv}
    leaves = {i + 1: {f"[{a},{b}]" for a, b in T} for i, T in enumerate(cells)}
    ref, ref_cost = infer_tads_with_dp([(0, i + 1) for i in range(len(cells))], leaves, pos, pos, Ct=1.0, Cb=cb)
    M, fast_cost = median(cells, 1.0, cb)
    assert laminar(M), "fast median is not laminar"
    assert abs(cost(M, cells, cb) - fast_cost) < 1e-6, (cost(M, cells, cb), fast_cost)
    assert abs(cost(set(ref), cells, cb) - ref_cost) < 1e-6
    assert abs(fast_cost - ref_cost) < 1e-6, (fast_cost, ref_cost)
    return M == set(ref)


def main():
    rng = random.Random(0)
    same = 0
    for trial in range(3000):
        n = rng.randint(3, 14)
        cells = [random_laminar(rng, n, rng.randint(0, 6)) for _ in range(rng.randint(1, 7))]
        if not any(cells):
            continue
        same += compare(cells, rng.choice([0.0, 0.5, 1.0, 2.0]))
    print(f"random: 3000 instances, all optimal costs equal; identical medians in {same}")

    trees = sparse.load_npz(EXPERIMENT / "final/trees.npz").tocsr()
    intervals = json.loads((EXPERIMENT / "final/intervals.json").read_text())
    labels = np.load(EXPERIMENT / "nussinov_median/run/fit.npz")["labels"]
    for chrom in ("chr19", "chr1"):
        for c in range(5):
            cells = [{(intervals[j][1], intervals[j][2]) for j in trees[i].indices if intervals[j][0] == chrom}
                     for i in np.flatnonzero(labels == c)]
            print(f"real {chrom} cluster {c}: equal cost, identical median = {compare(cells, 1.0)}", flush=True)


if __name__ == "__main__":
    main()
