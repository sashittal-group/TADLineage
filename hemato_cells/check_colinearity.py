#!/usr/bin/env python3

import os
import argparse
import itertools
import sys
import networkx as nx
import csv

# allow importing utilities from parent directory
sys.path.append(os.path.abspath(".."))

from tad_utilities import (
    read_tad_intervals,
    parse_interval,
    boundaries_from_tads,
    relaxed_tad_rf,
    relaxed_boundary_rf
)

NODES = ["LT","ST","MPP","CMP","CLP","MEP","GMP","MKP","G","MK"]

def build_reference_tree():
    """
    Build the fixed reference tree as a NetworkX graph.
    """
    ref_edges = [
        ("LT","ST"),
        ("ST","MPP"),
        ("MPP","CMP"),
        ("MPP","CLP"),
        ("CMP","MEP"),
        ("CMP","GMP"),
        ("MEP","MKP"),
        ("GMP","G"),
        ("MKP","MK")
    ]
    G = nx.Graph()
    G.add_edges_from(ref_edges)
    return G

# ---------------- Original TAD-distance utilities ----------------
def read_node_tads(chrom, res):
    node_tads = {}
    for node in NODES:
        path = f"{node}/chr{chrom}/{res}/tads.tad"
        if not os.path.exists(path):
            print(f"Warning: missing {path}")
            continue
        intervals = read_tad_intervals(path)
        node_tads[node] = {parse_interval(s) for s in intervals}
    return node_tads

def compute_tad_distance(Tu, Tv, tolerance=0, Ct=1.0, Cb=1.0):
    rf_tad = relaxed_tad_rf(Tu, Tv, tol=tolerance)
    Su, Eu = boundaries_from_tads(Tu)
    Sv, Ev = boundaries_from_tads(Tv)
    rf_boundary = relaxed_boundary_rf(Su, Sv, Eu, Ev, tol=tolerance)
    return Ct * rf_tad + Cb * rf_boundary

def compute_distance_matrix(node_tads, tolerance):
    """
    Compute pairwise distances for all present nodes in node_tads.
    Returns a dict keyed by (u,v) -> float (symmetric).
    """
    dist = {}
    keys = list(node_tads.keys())
    for u, v in itertools.combinations(keys, 2):
        d = compute_tad_distance(node_tads[u], node_tads[v], tolerance=tolerance)
        dist[(u, v)] = float(d)
        dist[(v, u)] = float(d)
    # self distances
    for k in keys:
        dist[(k, k)] = 0.0
    return dist

def all_unique_three_node_paths(tree):
    """
    Return list of unique 3-node paths (A,B,C) where A and C are
    distinct neighbors of B. Each unordered pair {A,C} for center B
    appears only once, with ordering (A,B,C) where A < C (lexicographic).
    """
    paths = []
    for center in tree.nodes():
        neighbors = sorted(list(tree.neighbors(center)))
        # choose unordered pairs of neighbors
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                a = neighbors[i]
                c = neighbors[j]
                paths.append((a, center, c))
    return paths

def write_csv(records, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "chrom", "resolution", "tolerance",
            "node1", "node2", "node3",
            "dist_1_2", "dist_2_3", "dist_1_3",
            "sum_1_2_2_3", "diff"
        ])
        writer.writeheader()
        for r in records:
            writer.writerow(r)

def main():
    parser = argparse.ArgumentParser(description="Compute 3-node path distance comparisons on reference tree.")
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--resolution", required=True)
    parser.add_argument("--tolerance", type=int, default=0)
    args = parser.parse_args()

    chrom = args.chrom
    res = args.resolution
    tolerance = args.tolerance

    # read TADs for nodes available
    node_tads = read_node_tads(chrom, res)
    if not node_tads:
        print("No node TADs found for given chrom/resolution. Exiting.")
        return

    # compute pairwise distances among available nodes
    dist = compute_distance_matrix(node_tads, tolerance)

    # reference tree and all unique 3-node paths
    ref_tree = build_reference_tree()
    three_paths = all_unique_three_node_paths(ref_tree)

    records = []
    for a, b, c in three_paths:
        # skip if any node missing
        if a not in node_tads or b not in node_tads or c not in node_tads:
            print(f"Skipping path {a}-{b}-{c}: missing node TAD file")
            continue

        # distances: attempt to fetch from dist dict; if missing, compute on the fly
        def get_pair_dist(x, y):
            if (x, y) in dist:
                return dist[(x, y)]
            # fallback compute if both present
            if x in node_tads and y in node_tads:
                return float(compute_tad_distance(node_tads[x], node_tads[y], tolerance=tolerance))
            return float("nan")

        d_ab = get_pair_dist(a, b)
        d_bc = get_pair_dist(b, c)
        d_ac = get_pair_dist(a, c)
        sum_ab_bc = d_ab + d_bc if not (isinstance(d_ab, float) and isinstance(d_bc, float) and (d_ab != d_ab or d_bc != d_bc)) else float("nan")
        diff = d_ac - sum_ab_bc if d_ac == d_ac and sum_ab_bc == sum_ab_bc else float("nan")

        records.append({
            "chrom": chrom,
            "resolution": res,
            "tolerance": tolerance,
            "node1": a,
            "node2": b,
            "node3": c,
            "dist_1_2": d_ab,
            "dist_2_3": d_bc,
            "dist_1_3": d_ac,
            "sum_1_2_2_3": sum_ab_bc,
            "diff": diff
        })

    out_csv = f"3nodes/chr{chrom}/{res}/three_node_paths_tol_{tolerance}.csv"
    write_csv(records, out_csv)
    print(f"Saved three-node path comparisons to: {out_csv}")
    # also print summary
    print(f"Computed {len(records)} three-node paths (skipped missing nodes).")

if __name__ == "__main__":
    main()