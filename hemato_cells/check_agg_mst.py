#!/usr/bin/env python3

import os
import argparse
import itertools
import sys
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import zss

# allow importing utilities
sys.path.append(os.path.abspath(".."))

from tad_utilities import (
    read_tad_intervals,
    parse_interval,
    boundaries_from_tads,
    relaxed_tad_rf,
    relaxed_boundary_rf
)

# ---------------- Constants ----------------

NODES = ["LT","ST","MPP","CMP","CLP","MEP","GMP","MKP","G","MK"]
CHROMS = [str(i) for i in range(1,20)]

PREFERRED_EDGES = {
    tuple(sorted(("LT","ST"))),
    tuple(sorted(("ST","MPP"))),
    tuple(sorted(("MPP","CMP"))),
    tuple(sorted(("MPP","CLP"))),
    tuple(sorted(("CMP","MEP"))),
    tuple(sorted(("CMP","GMP"))),
    tuple(sorted(("MEP","MKP"))),
    tuple(sorted(("GMP","G"))),
    tuple(sorted(("MKP","MK"))),
}

# ---------------- Tree Edit Distance ----------------

class TreeNode:
    def __init__(self, label):
        self.label = label
        self.children = []

def nx_to_zss(root_label, G):
    node = TreeNode(root_label)

    if not hasattr(nx_to_zss, "visited"):
        nx_to_zss.visited = set()

    nx_to_zss.visited.add(root_label)

    # Sort neighbors for consistent tree structure (zss is sensitive to child order)
    for child in sorted(G.neighbors(root_label)):
        if child in nx_to_zss.visited:
            continue
        node.children.append(nx_to_zss(child, G))

    return node


def tree_edit_distance(G1, G2, root="LT"):

    nx_to_zss.visited = set()
    T1 = nx_to_zss(root, G1)

    nx_to_zss.visited = set()
    T2 = nx_to_zss(root, G2)

    return zss.simple_distance(
        T1,
        T2,
        get_children=lambda n: n.children,
        get_label=lambda n: n.label
    )

def build_reference_tree():

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


# ---------------- TAD utilities ----------------

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


def compute_denominator_for_lex(node_tads):

    max_starts = 0
    max_ends = 0

    for tads in node_tads.values():

        starts, ends = boundaries_from_tads(tads)

        max_starts = max(max_starts, len(starts))
        max_ends = max(max_ends, len(ends))

    denom = 2 * max_starts * max_ends + 1

    if denom < 1:
        denom = 1

    return denom

def compute_global_denominator(res):

    max_starts = 0
    max_ends = 0

    for chrom in CHROMS:

        node_tads = read_node_tads(chrom, res)

        for tads in node_tads.values():

            starts, ends = boundaries_from_tads(tads)

            max_starts = max(max_starts, len(starts))
            max_ends = max(max_ends, len(ends))

    denom = 2 * max_starts * max_ends + 1

    if denom < 1:
        denom = 1

    print(
        f"Global denominator computed: "
        f"max_starts={max_starts}, max_ends={max_ends}, denom={denom}"
    )

    return denom

def compute_tad_distance(Tu, Tv, tolerance=0, Ct=1.0, Cb=1.0, denom=None):

    rf_tad = relaxed_tad_rf(Tu, Tv, tol=tolerance)

    Su, Eu = boundaries_from_tads(Tu)
    Sv, Ev = boundaries_from_tads(Tv)

    rf_boundary = relaxed_boundary_rf(Su, Sv, Eu, Ev, tol=tolerance)

    if denom is None:
        return Ct * rf_tad + Cb * rf_boundary
    else:
        return Ct * (rf_tad / float(denom)) + Cb * rf_boundary


# ---------------- Genome-wide distances ----------------

def compute_genomewide_distances(res, tolerance, Ct=1.0, Cb=1.0, mode="convex", preferred_edge_factor=1.0):

    edge_sum = defaultdict(float)
    edge_count = defaultdict(int)

    denom = None

    if mode == "LexBoundaryFirst":
        denom = compute_global_denominator(res)

    for chrom in CHROMS:

        print(f"Processing chr{chrom}")

        node_tads = read_node_tads(chrom, res)

        if not node_tads:
            continue

        for u, v in itertools.combinations(node_tads.keys(), 2):

            d = compute_tad_distance(
                node_tads[u],
                node_tads[v],
                tolerance=tolerance,
                Ct=Ct,
                Cb=Cb,
                denom=denom
            )

            ek = tuple(sorted((u, v)))
            # apply preferred-edge scaling before aggregation
            if ek in PREFERRED_EDGES:
                d *= preferred_edge_factor

            edge_sum[ek] += d
            edge_count[ek] += 1

    # compute averaged distances (symmetrize to keep compatibility with callers)
    total_dist = {}
    for ek, s in edge_sum.items():
        cnt = edge_count.get(ek, 0)
        if cnt <= 0:
            continue
        avg = s / float(cnt)
        u, v = ek
        total_dist[(u, v)] = avg
        total_dist[(v, u)] = avg

    print(f"Aggregated {len(edge_sum)} unique edges across chromosomes")

    return total_dist


# ---------------- Build MST ----------------

def build_mst(dist):

    G = nx.Graph()

    nodes = set()

    for u,v in dist:
        nodes.add(u)
        nodes.add(v)

    for n in nodes:
        G.add_node(n)

    for (u,v),d in dist.items():
        if u < v:
            G.add_edge(u,v,weight=d)

    mst = nx.minimum_spanning_tree(G)

    return mst


def plot_mst(mst, output_path):

    pos = nx.spring_layout(mst, seed=1, k=1.5)

    plt.figure(figsize=(10,10))

    nx.draw(
        mst,
        pos,
        with_labels=True,
        node_size=2500,
        node_color="lightblue",
        font_size=10
    )

    edge_labels = {
        (u,v): f"{d['weight']:.1f}"
        for u,v,d in mst.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        mst,
        pos,
        edge_labels=edge_labels,
        label_pos=0.5,
        font_size=9,
        rotate=False
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.title("Genome-wide MST (TAD distance)")

    plt.savefig(output_path, dpi=300)

    plt.close()

    print("Saved:", output_path)


# ---------------- Main ----------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--resolution", required=True)

    parser.add_argument("--tolerance", type=int, default=0)

    parser.add_argument(
        "--mode",
        choices=["convex","LexBoundaryFirst"],
        default="convex"
    )

    parser.add_argument("--Ct", type=float, default=1.0)

    parser.add_argument("--Cb", type=float, default=1.0)

    parser.add_argument(
        "--preferred-edge-factor",
        type=float,
        default=1.0,
        help="Multiply weights of preferred/reference edges by this factor before aggregation",
    )

    args = parser.parse_args()

    res = args.resolution
    tolerance = args.tolerance
    mode = args.mode
    Ct = args.Ct
    Cb = args.Cb
    preferred_edge_factor = args.preferred_edge_factor

    print(
        f"Running genome-wide MST: "
        f"res={res}, tol={tolerance}, mode={mode}, Ct={Ct}, Cb={Cb}, pref_factor={preferred_edge_factor}"
    )

    # compute distances
    dist = compute_genomewide_distances(
        res,
        tolerance,
        Ct=Ct,
        Cb=Cb,
        mode=mode,
        preferred_edge_factor=preferred_edge_factor
    )

    # build MST
    mst = build_mst(dist)

    # save figure
    out_png = f"msts/genomewide/{res}/mstree_tol_{tolerance}_{mode}_ef_{preferred_edge_factor}.png"

    plot_mst(mst, out_png)

    # compute tree edit distance
    ref_tree = build_reference_tree()

    ted = tree_edit_distance(mst, ref_tree, root="LT")

    print("Tree edit distance:", ted)

    out_txt = f"msts/genomewide/{res}/tree_distance_tol_{tolerance}_{mode}_ef_{preferred_edge_factor}.txt"

    os.makedirs(os.path.dirname(out_txt), exist_ok=True)

    with open(out_txt,"w") as f:
        f.write(f"{ted}\n")

    print("Saved tree distance:", out_txt)


if __name__ == "__main__":
    main()