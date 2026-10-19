#!/usr/bin/env python3
"""
infer_full_tree.py

Infer the hematopoietic tree over all ten cell types with the large parsimony ILP,
and score it by quartet consistency with the known lineage.

Every cell type is given: the ten observed TAD hierarchies

    LT, ST, MPP, CLP, CMP, MEP, GMP, MKP, MK, G

are the leaves of the ILP, and no tree is given. The ILP chooses a binary tree over
them (8 unobserved internal vertices) together with the hierarchies of those
vertices. The known lineage is

    LT -> ST -> MPP -> {CLP, CMP};  CMP -> {MEP -> MKP -> MK, GMP -> G}

in which most cell types are ancestors of others. As an unrooted tree on the ten
labelled cell types, each of its edges splits them into two groups.

Quartet consistency: for every four cell types {a, b, c, d}, the reference resolves
them as ab|cd if one of its edges separates {a, b} from {c, d}. 36 of the 210
quartets are not resolved by any edge (e.g. {LT, MPP, CLP, CMP}: MPP sits at the
branch point and the other three hang off it separately); they are left out. The
inferred tree is binary, so it resolves every quartet. The score is the fraction of
the 174 resolved reference quartets that the inferred tree resolves the same way.
A random binary tree scores 1/3 in expectation.

The root sits on the edge above LT (``--free-root`` lifts this). The objective
ignores the root, so this excludes no unrooted topology.

Solver (--method). The single large ILP proves optimality slowly with ten leaves
(chr19 at 1 Mb: 9 minutes for 31 candidate TADs). The default, bnb, is an exact
branch and bound over topologies (topology_search.py): a Fitch lower bound prunes
partial trees and each surviving complete tree is solved exactly by small
parsimony. It returns EVERY optimal tree; the reported consistency is the mean over
them (min and max are recorded too). --method ilp returns one optimal tree, the
one Gurobi happens to find.

Usage (from an analysis folder, e.g. hemato_cells/large_parsimony/ontad_unweighted_full):
    python ../infer_full_tree.py --data-dir ../.. --chrom 19 --resolution 1mb \
        --tolerance 1 --output results/jsons/full_chr19_1mb_1.json
"""

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from topology_search import TopologySearch  # noqa: E402
from infer_quartet_tree import harmonize_private, tad_path, write_json  # noqa: E402
from filtering import cluster_positions_median  # noqa: E402
from large_parsimony_ilp import LargeParsimonyILP, candidate_universe, nontrivial_splits  # noqa: E402

NODES = ["LT", "ST", "MPP", "CLP", "CMP", "MEP", "GMP", "MKP", "MK", "G"]
REFERENCE_EDGES = [
    ("LT", "ST"), ("ST", "MPP"), ("MPP", "CMP"), ("MPP", "CLP"),
    ("CMP", "MEP"), ("CMP", "GMP"), ("MEP", "MKP"), ("MKP", "MK"), ("GMP", "G"),
]
ROOT_NAME = "LT"
MAX_TREES_IN_JSON = 100


def reference_splits(nodes, edges):
    """For each edge of the (labelled, non-binary) reference tree, the set of cell
    types on one side of it."""
    adj = {n: set() for n in nodes}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    splits = []
    for a, b in edges:
        side, stack = {b}, [b]
        while stack:
            w = stack.pop()
            for nxt in adj[w]:
                if nxt != a and nxt not in side:
                    side.add(nxt)
                    stack.append(nxt)
        splits.append(frozenset(side))
    return splits


def quartet_topology(splits, quartet, universe):
    """The pair containing quartet[0] under the split that resolves the quartet,
    or None if no split does. Both sides of each split are checked."""
    q = set(quartet)
    for side in splits:
        for s in (side, universe - side):
            inside = q & s
            if len(inside) == 2 and quartet[0] in inside:
                return frozenset(inside)
    return None


def ontad_score_weights(raw_paths, harmonized, tolerance):
    """Weighted TAD distance d_w from OnTAD's confidence (column 5 of tads.tad).

    Harmonization moves boundaries to cluster medians and drops the score column, so
    each harmonized TAD of a cell is traced back to that cell's OnTAD call(s) with
    the same boundary clusters (or the original interval, for pairs harmonization
    restored); several calls -> the highest score. A TAD's weight is the mean of its
    scores over the cells that call it, scaled so the mean weight over the observed
    TADs is 1. A boundary's weight is the largest weight of the TADs starting (or
    ending) there. Unobserved candidate TADs keep weight 1 (Ct)."""
    raw = {}
    for name, path in raw_paths.items():
        rows = []
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2:
                    rows.append((int(parts[0]), int(parts[1]), float(parts[4]) if len(parts) >= 5 else None))
        raw[name] = rows
    start_map = cluster_positions_median([a for rows in raw.values() for a, _, _ in rows], tolerance)
    end_map = cluster_positions_median([b for rows in raw.values() for _, b, _ in rows], tolerance)

    scores, missing = {}, 0
    for name, rows in raw.items():
        lookup = {}
        for a, b, score in rows:
            if score is None:
                continue
            for key in ((start_map[a], end_map[b]), (a, b)):
                lookup[key] = max(lookup.get(key, float("-inf")), score)
        for t in harmonized[name]:
            if t in lookup:
                scores.setdefault(t, []).append(lookup[t])
            else:
                missing += 1
    raw_w = {t: sum(v) / len(v) for t, v in scores.items()}
    scale = sum(raw_w.values()) / len(raw_w)
    tad_w = {t: w / scale for t, w in raw_w.items()}
    start_w, end_w = {}, {}
    for (a, b), w in tad_w.items():
        start_w[a] = max(start_w.get(a, 0.0), w)
        end_w[b] = max(end_w.get(b, 0.0), w)
    return tad_w, start_w, end_w, {"n_weighted_tads": len(tad_w), "n_unmatched_calls": missing,
                                   "score_scale": scale, "weight_min": min(tad_w.values()),
                                   "weight_max": max(tad_w.values())}


SCORE_FLOOR = 0.01      # RobusTAD scores can be <= 0; weights must stay positive


def robustad_score_weights(raw_paths, harmonized, tolerance, chrom, res, robustad_root):
    """Weighted TAD distance d_w from RobusTAD's scores: each TAD's score
    (tads_all_with_bins.tsv) and each boundary's left / right score
    (boundary_with_bins.tsv: a start's leftScore at its bin, an end's rightScore at
    the next bin, where RobusTAD puts the right boundary of a TAD ending in bin b).

    As in ontad_score_weights, each harmonized TAD or boundary of a cell is traced
    back to that cell's calls through the boundary clusters (several -> the highest
    score); its weight is the mean over the cells that have it. Scores are floored
    at SCORE_FLOOR, then TAD weights and boundary weights are each scaled to mean 1.
    Unobserved candidate TADs keep weight 1 (Ct)."""
    import csv

    raw = {}
    for name, path in raw_paths.items():
        with open(path) as fh:
            raw[name] = [tuple(int(x) for x in line.split()[:2]) for line in fh if len(line.split()) >= 2]
    start_map = cluster_positions_median([a for rows in raw.values() for a, _ in rows], tolerance)
    end_map = cluster_positions_median([b for rows in raw.values() for _, b in rows], tolerance)

    tad_scores, start_scores, end_scores, missing = {}, {}, {}, 0
    for name in raw_paths:
        base = os.path.join(robustad_root, name, f"chr{chrom}", res, f"{name}_chr{chrom}_{res}")
        called = {}
        with open(base + "_tads_all_with_bins.tsv") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                key = (int(float(row["start_bin"])), int(float(row["end_bin"])))
                called[key] = max(called.get(key, float("-inf")), float(row["score"]))
        left, right = {}, {}
        with open(base + "_boundary_with_bins.tsv") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                pos = int(float(row["bin250kb"]))       # the bin at this resolution
                left[pos] = float(row["leftScore"])
                right[pos] = float(row["rightScore"])

        tad_lookup, start_lookup, end_lookup = {}, {}, {}
        for (a, b) in raw[name]:
            score = called.get((a, b))
            if score is not None:
                for key in ((start_map[a], end_map[b]), (a, b)):
                    tad_lookup[key] = max(tad_lookup.get(key, float("-inf")), score)
            for key in (start_map[a], a):
                if a in left:
                    start_lookup[key] = max(start_lookup.get(key, float("-inf")), left[a])
            # the right boundary of a TAD ending in bin b sits at bin b + 1 in the
            # boundary file (end_bin = (end - 1) // resolution)
            for key in (end_map[b], b):
                if b + 1 in right:
                    end_lookup[key] = max(end_lookup.get(key, float("-inf")), right[b + 1])
        for t in harmonized[name]:
            if t in tad_lookup:
                tad_scores.setdefault(t, []).append(max(tad_lookup[t], SCORE_FLOOR))
            else:
                missing += 1
        for a in {a for a, _ in harmonized[name]}:
            if a in start_lookup:
                start_scores.setdefault(a, []).append(max(start_lookup[a], SCORE_FLOOR))
        for b in {b for _, b in harmonized[name]}:
            if b in end_lookup:
                end_scores.setdefault(b, []).append(max(end_lookup[b], SCORE_FLOOR))

    mean = lambda v: sum(v) / len(v)  # noqa: E731
    tad_raw = {t: mean(v) for t, v in tad_scores.items()}
    t_scale = mean(list(tad_raw.values()))
    tad_w = {t: w / t_scale for t, w in tad_raw.items()}
    start_raw = {p: mean(v) for p, v in start_scores.items()}
    end_raw = {p: mean(v) for p, v in end_scores.items()}
    b_scale = mean(list(start_raw.values()) + list(end_raw.values()))
    start_w = {p: w / b_scale for p, w in start_raw.items()}
    end_w = {p: w / b_scale for p, w in end_raw.items()}
    n_starts = len({a for tads in harmonized.values() for a, _ in tads})
    n_ends = len({b for tads in harmonized.values() for _, b in tads})
    return tad_w, start_w, end_w, {
        "n_weighted_tads": len(tad_w), "n_unmatched_calls": missing,
        "n_weighted_starts": f"{len(start_w)}/{n_starts}", "n_weighted_ends": f"{len(end_w)}/{n_ends}",
        "tad_weight_range": [min(tad_w.values()), max(tad_w.values())],
        "boundary_weight_range": [min(list(start_w.values()) + list(end_w.values())),
                                  max(list(start_w.values()) + list(end_w.values()))],
    }


def newick(arcs, names, root):
    children = {}
    for u, v in arcs:
        children.setdefault(u, []).append(v)

    def rec(v):
        if v not in children:
            return names[v]
        return "(" + ",".join(rec(c) for c in sorted(children[v], key=lambda c: rec(c))) + ")"

    return rec(root) + ";"


def newick_of_edges(search, edges, nodes):
    k = len(nodes)
    vname = {v: (nodes[v - 1] if v <= k else f"I{v}") for v in range(1, 2 * k)}
    return newick(search.rooted_arcs(edges), vname, 2 * k - 1)


def quartet_scorer(nodes, reference_edges):
    """Function: split masks of a binary tree (bit i = nodes[i]) -> fraction of the
    resolved reference quartets it resolves the reference's way. For each possible
    split, the reference quartets it resolves that way are precomputed; a binary tree
    resolves every quartet exactly once, so its consistent quartets are the union
    over its splits."""
    k = len(nodes)
    reference = reference_splits(nodes, reference_edges)
    universe = frozenset(nodes)
    ref_quartets = []
    for quartet in itertools.combinations(nodes, 4):
        ref = quartet_topology(reference, quartet, universe)
        if ref is not None:
            a, b = sorted(nodes.index(n) for n in ref)
            c, d = sorted(nodes.index(n) for n in set(quartet) - ref)
            ref_quartets.append((a, b, c, d))
    n_resolved = len(ref_quartets)
    consistent_by_split = [0] * (1 << k)
    for mask in range(1 << k):
        for i, (a, b, c, d) in enumerate(ref_quartets):
            sa, sb, sc, sd = (mask >> a & 1, mask >> b & 1, mask >> c & 1, mask >> d & 1)
            if sa == sb and sc == sd and sa != sc:
                consistent_by_split[mask] |= 1 << i

    def score(masks):
        agree = 0
        for m in masks:
            agree |= consistent_by_split[m]
        return agree.bit_count() / n_resolved

    score.n_resolved = n_resolved
    return score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="../..", help="folder holding <NODE>/chr<C>/<RES>/tads.tad")
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--resolution", required=True)
    ap.add_argument("--tolerance", type=int, default=1, help="boundary tolerance in bins")
    ap.add_argument("--Ct", type=float, default=1.0)
    ap.add_argument("--Cb", type=float, default=1.0)
    ap.add_argument("--mode", default="convex", choices=["convex", "lex_boundary_first", "lex_tad_first"])
    ap.add_argument("--threads", type=int, default=0, help="Gurobi threads (0 = Gurobi default)")
    ap.add_argument("--time-limit", type=float, default=0, help="seconds per Gurobi call (0 = none)")
    ap.add_argument("--lp-method", type=int, default=-1,
                    help="Gurobi Method for LP relaxations: -1 auto, 1 dual simplex, 2 barrier, 3 concurrent")
    ap.add_argument("--solver-log", action="store_true", help="print Gurobi's progress log")
    ap.add_argument("--free-root", action="store_true",
                    help="let the ILP place the root anywhere (same optimum, slower)")
    ap.add_argument("--method", default="bnb", choices=["bnb", "ilp"],
                    help="bnb: branch and bound over topologies, all optimal trees (default); "
                         "ilp: the single large parsimony ILP, one optimal tree")
    ap.add_argument("--weighting", default="none", choices=["none", "ontad_score", "robustad_score"],
                    help="none: TAD distance with Ct, Cb; ontad_score / robustad_score: weighted "
                         "TAD distance d_w from the caller's scores (see ontad_score_weights, "
                         "robustad_score_weights)")
    ap.add_argument("--robustad-root", default=str(REPO_ROOT / "robustad_predictions"),
                    help="robustad_predictions folder (robustad_score weighting)")
    ap.add_argument("--output", required=True, help="summary JSON path")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    chrom = str(args.chrom).removeprefix("chr")
    res, tol = args.resolution, args.tolerance

    # leaf vertex ids 1..k in the ILP
    names = {i + 1: n for i, n in enumerate(NODES)}
    ids = {n: v for v, n in names.items()}
    k = len(NODES)

    raw_paths = {n: tad_path(args.data_dir, n, chrom, res) for n in NODES}
    for p in raw_paths.values():
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    print(f"chr{chrom} {res} tol={tol}: {k} leaves {', '.join(NODES)}")
    harmonized = harmonize_private(raw_paths, tol)
    leaf_tads = [harmonized[n] for n in NODES]

    summary = {
        "chrom": chrom,
        "resolution": res,
        "tolerance": tol,
        "leaves": NODES,
        "reference_edges": REFERENCE_EDGES,
        "mode": args.mode,
        "Ct": args.Ct,
        "Cb": args.Cb,
        "root_on_lt_edge": not args.free_root,
        "n_tads_after_harmonization": {n: len(harmonized[n]) for n in NODES},
    }

    weights = {}
    summary["weighting"] = args.weighting
    if args.weighting != "none":
        if args.mode != "convex":
            raise SystemExit("--weighting is implemented for --mode convex only")
        if args.weighting == "ontad_score":
            tad_w, start_w, end_w, info = ontad_score_weights(raw_paths, harmonized, tol)
        else:
            tad_w, start_w, end_w, info = robustad_score_weights(
                raw_paths, harmonized, tol, chrom, res, args.robustad_root)
        weights = {"tad_weights": tad_w, "start_weights": start_w, "end_weights": end_w}
        summary["weights"] = info
        print(f"{args.weighting} weights: {info}")

    t_start = time.time()
    reference = reference_splits(NODES, REFERENCE_EDGES)
    universe = frozenset(NODES)

    if args.method == "ilp":
        ilp = LargeParsimonyILP(
            leaf_tads, Ct=args.Ct, Cb=args.Cb, mode=args.mode,
            threads=args.threads or None, time_limit=args.time_limit or None,
            root_leaf=None if args.free_root else ids[ROOT_NAME],
            lp_method=None if args.lp_method == -1 else args.lp_method,
            solver_log=args.solver_log, verbose=args.verbose, **weights,
        )
        summary.update({"n_candidate_tads": len(ilp.candidates), "build_seconds": ilp.build_seconds})
        result = ilp.solve()
        summary["status"] = result["status"]
        if result["status"] != "optimal":
            summary["seconds_total"] = time.time() - t_start
            write_json(summary, args.output)
            print(f"Large parsimony solve did not finish: {result['status']}")
            return
        vname = lambda v: names.get(v, f"I{v}")  # noqa: E731
        split_masks = [[sum(1 << (v - 1) for v in side) for side in nontrivial_splits(result["arcs"], k)]]
        newick_of = [lambda: newick(result["arcs"], {v: vname(v) for v in range(1, 2 * k)}, 2 * k - 1)]
        summary.update({"cost": result["cost"], "solve_seconds": result["seconds"]})
    else:
        if args.mode != "convex":
            raise SystemExit("--method bnb is implemented for --mode convex only")
        # leaf 0 of the search is LT (NODES[0]); the tree is rooted on its pendant edge
        search = TopologySearch(leaf_tads, Ct=args.Ct, Cb=args.Cb,
                                threads=args.threads or None, time_limit=args.time_limit or None,
                                verbose=args.verbose, **weights)
        summary["n_candidate_tads"] = len(candidate_universe(search.leaf_tads)[2])
        result = search.run()
        summary["status"] = result["status"]
        vname = lambda v: names.get(v, f"I{v}")  # noqa: E731
        paths = result["optimal_paths"]
        split_masks = (search.split_masks(search.edges_of(p)) for p in paths)
        newick_of = [
            (lambda p=p: newick(search.rooted_arcs(search.edges_of(p)),
                                {v: vname(v) for v in range(1, 2 * k)}, 2 * k - 1))
            for p in paths
        ]
        summary.update({key: result[key] for key in (
            "cost", "ub_initial", "n_bnb_nodes", "n_complete_survivors", "settled_by",
            "step_seconds", "enumeration_seconds")})
        summary["solve_seconds"] = result["seconds"]

    # Quartet consistency of every optimal tree
    score = quartet_scorer(NODES, REFERENCE_EDGES)
    n_resolved = score.n_resolved
    per_tree = [score(masks) for masks in split_masks]
    order = sorted(range(len(per_tree)), key=lambda i: -per_tree[i])
    top = [(newick_of[i](), per_tree[i]) for i in order[:MAX_TREES_IN_JSON]]
    summary.update({
        "n_optimal_trees": len(per_tree),
        "n_quartets_resolved_in_reference": n_resolved,
        "quartet_consistency": sum(per_tree) / len(per_tree),
        "quartet_consistency_min": min(per_tree),
        "quartet_consistency_max": max(per_tree),
        "optimal_trees": [{"newick": nw, "quartet_consistency": qc} for nw, qc in top],
        "seconds_total": time.time() - t_start,
    })
    print(f"Optimal cost {summary['cost']:g}; {len(per_tree)} optimal tree(s)")
    for nw, qc in top[:5]:
        print(f"  {nw}  consistency {qc:.3f}")
    print(f"Quartet consistency over the optimal trees: mean {summary['quartet_consistency']:.3f}, "
          f"min {min(per_tree):.3f}, max {max(per_tree):.3f}")
    write_json(summary, args.output)

if __name__ == "__main__":
    main()
