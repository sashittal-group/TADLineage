#!/usr/bin/env python3
"""Does imputation ever add an interval the leaf ALREADY has, within tolerance?

The solver compares intervals by exact equality; the scoring metric matches
within +/- tolerance and one-to-one. So an imputed (1,11) on a leaf that already
called (1,10) earns no recall (the truth TAD is already claimed) and costs
precision outright. Snapping should prevent it -- both ends land on one consensus
-- but mode-seeking does not guarantee it: if 12 is absorbed by centre 13 first,
10 can be left as its own centre, and 10 vs 12 are still "the same" to the metric.

Counts how often it actually happens, per method and depth.
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

paths = sorted((EXP / "calls").rglob("region_*.json"))[:15]
dupes = defaultdict(int)
added = defaultdict(int)
regions_hit = defaultdict(set)

for path in paths:
    rec = json.loads(path.read_text())
    tree = get_tree(rec["tree"], 16, 20240101)
    for pt in rec["points"]:
        d = pt["depth_label"]
        calls, scores = {}, {}
        for leaf, rows in pt["leaf_calls"].items():
            calls[leaf] = {(int(r[0]), int(r[1])) for r in rows}
            scores[leaf] = {(int(r[0]), int(r[1])): float(r[2]) for r in rows}
        bnd = {leaf: rti.BoundaryScores.from_record(b, TOL)
               for leaf, b in pt.get("leaf_boundary", {}).items()}
        for prefix in ("snap", "wsnap"):
            use_imp, use_pri, wsnap = rti.METHOD_SWITCHES[prefix]
            pred, _ = rti.method_impute(
                tree, calls, scores, bnd, 0.5, 1.0, 1.0, "convex", TOL,
                use_importance=use_imp, use_price_discount=use_pri,
                weighted_snap=wsnap, importance_weight=0.1,
                price_discount=0.6, price_floor=0.25)
            if pred is None:
                continue
            for leaf in tree.leaves:
                new = pred[leaf] - calls[leaf]
                added[(prefix, d)] += len(new)
                for iv in new:
                    if any(abs(iv[0] - c[0]) <= TOL and abs(iv[1] - c[1]) <= TOL
                           for c in calls[leaf]):
                        dupes[(prefix, d)] += 1
                        regions_hit[(prefix, d)].add((rec["tree"], rec["region"]))

print(f"{len(paths)} regions, impute_cost=0.5\n")
print(f"{'method':>6} {'depth':>7} {'imputed':>8} {'near-dup':>9} {'%':>6}  regions affected")
for prefix in ("snap", "wsnap"):
    for d, _ in rti.DEPTH_LEVELS:
        k = (prefix, d)
        if not added[k]:
            continue
        pct = 100.0 * dupes[k] / added[k]
        print(f"{prefix:>6} {d:>7} {added[k]:>8} {dupes[k]:>9} {pct:>5.1f}%  "
              f"{len(regions_hit[k])}/{len(paths)}")
