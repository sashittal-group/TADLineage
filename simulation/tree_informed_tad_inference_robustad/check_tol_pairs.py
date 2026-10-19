#!/usr/bin/env python3
"""Two further consequences of tolerance matching, both measured not assumed.

1. Does the TRUTH contain pairs within tolerance of each other? If it does, the
   metric cannot fully distinguish them: one predicted interval can only claim
   one of them (matching is greedy one-to-one), so recall is structurally capped
   and a "near-duplicate" prediction is not always wasted.

2. Do RobusTAD's own RAW calls contain such pairs? Its assembly emits nested
   levels, so (a,b) and (a,b+1) can both be called -- and only one of them can
   ever match. That would cap the baseline's precision independently of the tree.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from collections import defaultdict
from pathlib import Path

EXP = Path("/Users/PSLabMonisha/Desktop/tads/simulation/tree_informed_tad_inference_robustad")
for p in (EXP, EXP.parent, EXP.parent.parent, EXP.parent / "data_quality_sweep"):
    sys.path.insert(0, str(p))

import config
import run_tree_impute_rt as rti
from trees import get_tree

TOL = 2
params = dataclasses.replace(
    config.DEFAULT_PARAMS, n_bins=500, n_top_tads=10, nesting_depth=2,
    min_size=8, min_score_node_tads=6, events_per_edge=1, tolerance=TOL,
)


def near_pairs(ivs, tol=TOL):
    ivs = sorted(ivs)
    n = 0
    for i in range(len(ivs)):
        for j in range(i + 1, len(ivs)):
            if abs(ivs[i][0] - ivs[j][0]) <= tol and abs(ivs[i][1] - ivs[j][1]) <= tol:
                n += 1
    return n


paths = sorted((EXP / "calls").rglob("region_*.json"))[:15]

truth_pairs = truth_total = 0
call_pairs = defaultdict(int)
call_total = defaultdict(int)

for path in paths:
    rec = json.loads(path.read_text())
    tree = get_tree(rec["tree"], 16, 20240101)
    hiers = rti.build_hierarchies(tree, rec["region"], params)
    for leaf in tree.leaves:
        t = rti.true_intervals(hiers[leaf])
        truth_pairs += near_pairs(t)
        truth_total += len(t)
    for pt in rec["points"]:
        d = pt["depth_label"]
        for leaf, rows in pt["leaf_calls"].items():
            ivs = {(int(r[0]), int(r[1])) for r in rows}
            call_pairs[d] += near_pairs(ivs)
            call_total[d] += len(ivs)

print(f"{len(paths)} regions, tolerance={TOL}\n")
print(f"1. TRUTH: {truth_pairs} within-tolerance pairs over {truth_total} true TADs "
      f"({100.0 * truth_pairs / max(truth_total, 1):.2f}% )")
print("   -> a prediction can claim only one of a colliding pair (greedy 1-to-1)\n")
print("2. RAW RobusTAD calls, within-tolerance pairs per leaf set:")
print(f"   {'depth':>7} {'pairs':>7} {'calls':>7} {'% of calls':>11}")
for d, _ in rti.DEPTH_LEVELS:
    if not call_total[d]:
        continue
    print(f"   {d:>7} {call_pairs[d]:>7} {call_total[d]:>7} "
          f"{100.0 * call_pairs[d] / call_total[d]:>10.2f}%")
