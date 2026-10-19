#!/usr/bin/env python3
"""Are RobusTAD's scores informative enough to be worth weighting by?

The weighted method rests on two assumptions. Neither is free, and if either
fails the weighting cannot help no matter how it is tuned, so both are measured
here directly from the cached calls -- before reading anything into an F1 curve.

  A. **TAD/boundary scores separate right from wrong calls.** If a leaf's true
     positives score no higher than its false positives, ``leaf_importance``
     pulls good and bad domains onto the parent equally and is just noise.

  B. **A leaf retains boundary evidence at the domains it MISSED.** This is the
     assumption behind the per-leaf imputation price: RobusTAD scores every bin,
     including where it assembled nothing, so a leaf that half-sees a domain
     should show more boundary support at its own missed truth than at intervals
     that are not true for it at all. If those two are indistinguishable, the
     discount is charging leaves at random.

Both are reported as an AUC (the probability that a randomly chosen positive
outscores a randomly chosen negative): 0.5 is no information, 1.0 is perfect.

Usage (gurobi/plotting env -- no RobusTAD needed, this reads the cache):
    python diagnose_scores.py --calls-dir calls
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
SIM_ROOT = EXP_DIR.parent
REPO_ROOT = SIM_ROOT.parent
for p in (str(REPO_ROOT), str(SIM_ROOT), str(SIM_ROOT / "data_quality_sweep"), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_tree_impute_rt as rti
from trees import get_tree
from tad_utilities import match_tads_within_tolerance


def auc(pos: list[float], neg: list[float]) -> tuple[float, int, int]:
    """Mann-Whitney AUC with ties at half credit, via mid-ranks (O(n log n)).

    The pools run to 1e5 entries at the noisy end, so the naive all-pairs form is
    not an option.
    """
    if not pos or not neg:
        return float("nan"), len(pos), len(neg)
    values = sorted((v, i) for i, v in enumerate(pos + neg))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1][0] == values[i][0]:
            j += 1
        mid = (i + j) / 2.0 + 1.0          # 1-based mid-rank for the tie block
        for k in range(i, j + 1):
            ranks[values[k][1]] = mid
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    rank_sum = sum(ranks[:n_pos])
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg), n_pos, n_neg


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calls-dir", type=Path, default=Path("calls"))
    p.add_argument("--n-nodes", type=int, default=16)
    p.add_argument("--tree-seed", type=int, default=20240101)
    p.add_argument("--tolerance", type=int, default=2)
    p.add_argument("--n-bins", type=int, default=500)
    p.add_argument("--n-top-tads", type=int, default=10)
    p.add_argument("--nesting-depth", type=int, default=2)
    p.add_argument("--min-size", type=int, default=8)
    p.add_argument("--min-score-node-tads", type=int, default=6)
    p.add_argument("--events-per-edge", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--amplitude", type=float, default=rti.DEFAULT_AMPLITUDE)
    args = p.parse_args()
    if args.seed is None:
        import config
        args.seed = config.DEFAULT_PARAMS.seed
    params = rti.build_params(args)

    paths = sorted(args.calls_dir.rglob("region_*.json"))
    if not paths:
        raise SystemExit(f"no cached calls under {args.calls_dir}")

    # depth -> pools
    tad_tp, tad_fp = defaultdict(list), defaultdict(list)
    miss_sup, false_sup = defaultdict(list), defaultdict(list)
    called_frac = defaultdict(list)

    for path in paths:
        rec = json.loads(path.read_text())
        tree = get_tree(rec["tree"], args.n_nodes, args.tree_seed)
        hiers = rti.build_hierarchies(tree, rec["region"], params)
        truth = {l: rti.true_intervals(hiers[l]) for l in tree.leaves}

        for pt in rec["points"]:
            depth = pt["depth_label"]
            calls, scores = {}, {}
            for leaf, rows in pt["leaf_calls"].items():
                calls[leaf] = {(int(r[0]), int(r[1])) for r in rows}
                scores[leaf] = {(int(r[0]), int(r[1])): float(r[2]) for r in rows}
            boundary = {
                leaf: rti.BoundaryScores.from_record(b, args.tolerance)
                for leaf, b in pt.get("leaf_boundary", {}).items()
            }
            # Candidate pool for (B): every interval SOME leaf called, which is
            # exactly the universe imputation may draw from.
            pool = sorted(set().union(*(calls[l] for l in tree.leaves))) if calls else []

            for leaf in tree.leaves:
                if leaf not in calls:
                    continue
                # match_tads_within_tolerance returns (inferred, truth) pairs.
                matched, _, _ = match_tads_within_tolerance(calls[leaf], truth[leaf],
                                                            args.tolerance)
                hit = {tuple(inf) for inf, _true in matched}
                for iv, sc in scores[leaf].items():
                    (tad_tp if iv in hit else tad_fp)[depth].append(sc)

                bs = boundary.get(leaf)
                if bs is None:
                    continue
                # (B) a leaf's own support at intervals it did NOT call, split by
                # whether they are true for that leaf.
                uncalled = [iv for iv in pool if iv not in calls[leaf]]
                if not uncalled:
                    continue
                true_un, false_un = [], []
                for iv in uncalled:
                    m, _, _ = match_tads_within_tolerance({iv}, truth[leaf], args.tolerance)
                    (true_un if m else false_un).append(bs.interval_support(*iv))
                miss_sup[depth].extend(true_un)
                false_sup[depth].extend(false_un)
                called_frac[depth].append(len(calls[leaf]))

    order = [d for d, _ in rti.DEPTH_LEVELS if d in tad_tp or d in miss_sup]
    print(f"pooled over {len(paths)} region files\n")
    print("A. TAD score separates the leaf's true from false calls")
    print(f"  {'depth':>7}  {'AUC':>6}  {'mean TP':>8}  {'mean FP':>8}  {'nTP':>5}  {'nFP':>5}")
    for d in order:
        a, npos, nneg = auc(tad_tp[d], tad_fp[d])
        print(f"  {d:>7}  {a:>6.3f}  {mean(tad_tp[d]):>8.3f}  {mean(tad_fp[d]):>8.3f}"
              f"  {npos:>5}  {nneg:>5}")

    print("\nB. A leaf's own boundary support at intervals it did NOT call,")
    print("   true-for-this-leaf vs not (this is what the price discount reads)")
    print(f"  {'depth':>7}  {'AUC':>6}  {'mean true':>10}  {'mean false':>10}  {'nTrue':>6}  {'nFalse':>6}")
    for d in order:
        a, npos, nneg = auc(miss_sup[d], false_sup[d])
        print(f"  {d:>7}  {a:>6.3f}  {mean(miss_sup[d]):>10.3f}  {mean(false_sup[d]):>10.3f}"
              f"  {npos:>6}  {nneg:>6}")

    print("\nCalls per leaf")
    for d in order:
        print(f"  {d:>7}  {mean(called_frac[d]):.1f}")


if __name__ == "__main__":
    main()
