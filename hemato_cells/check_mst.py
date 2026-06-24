#!/usr/bin/env python3

import os
import argparse
import itertools
import sys
import networkx as nx
import matplotlib.pyplot as plt
import zss  # Zhang-Shasha tree edit distance

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

# ---------------- Tree Edit Distance Utilities ----------------
class TreeNode:
    def __init__(self, label):
        self.label = label
        self.children = []

def nx_to_zss(root_label, G):
    """
    Convert a NetworkX tree to zss.TreeNode.
    Assumes G is a tree and root_label is the root.
    """
    node = TreeNode(root_label)
    # use a per-call visited set stored on the function to avoid global state leaking
    if not hasattr(nx_to_zss, "visited") or nx_to_zss.visited is None:
        nx_to_zss.visited = set()
    nx_to_zss.visited.add(root_label)
    for child in G.neighbors(root_label):
        if child in nx_to_zss.visited:
            continue
        node.children.append(nx_to_zss(child, G))
    return node

def tree_edit_distance(G1, G2, root="LT"):
    """
    Compute tree edit distance using zss.
    """
    # clear visited and build T1
    nx_to_zss.visited = set()
    T1 = nx_to_zss(root, G1)

    # clear visited and build T2
    nx_to_zss.visited = set()
    T2 = nx_to_zss(root, G2)

    return zss.simple_distance(
        T1, T2,
        get_children=lambda n: n.children,
        get_label=lambda n: n.label
    )

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

# ---------------- Original MST / TAD Utilities ----------------
def read_node_tads(chrom, res):
    """
    Read TADs for all nodes (NODES) from expected paths:
      <NODE>/chr<chrom>/<res>/tads.tad

    Returns dict node -> set of (start,end) tuples
    """
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
    """
    Compute denominator = 2 * max_starts * max_ends + 1
    max_starts / max_ends are computed across node_tads (per-run).
    Returns integer denominator (>=1).
    """
    max_starts = 0
    max_ends = 0
    for tads in node_tads.values():
        starts, ends = boundaries_from_tads(tads)
        max_starts = max(max_starts, len(starts))
        max_ends = max(max_ends, len(ends))
    denom = 2 * max_starts * max_ends + 1
    # guarantee >= 1
    if denom < 1:
        denom = 1
    return denom, max_starts, max_ends

def compute_tad_distance(Tu, Tv, tolerance=0, Ct=1.0, Cb=1.0, denom=None):
    """
    Distance between two TAD sets.
    If denom is provided (int), use lex_boundary_first scaling:
      Ct * rf_tad / denom + Cb * rf_boundary
    Otherwise (denom is None) use convex:
      Ct * rf_tad + Cb * rf_boundary
    """
    rf_tad = relaxed_tad_rf(Tu, Tv, tol=tolerance)

    Su, Eu = boundaries_from_tads(Tu)
    Sv, Ev = boundaries_from_tads(Tv)

    rf_boundary = relaxed_boundary_rf(Su, Sv, Eu, Ev, tol=tolerance)

    if denom is None:
        return Ct * rf_tad + Cb * rf_boundary
    else:
        # denom should be > 0 (we ensured in compute_denominator_for_lex)
        return Ct * (rf_tad / float(denom)) + Cb * rf_boundary

def compute_distance_matrix(node_tads, tolerance, Ct=1.0, Cb=1.0, mode="convex"):
    """
    Compute pairwise distances between nodes in node_tads.
    If mode == 'LexBoundaryFirst', compute denom using all node_tads
    and scale the tad RF accordingly.
    Returns dict keyed by (u,v) -> float (symmetric).
    """
    denom = None
    if mode == "LexBoundaryFirst":
        denom, max_s, max_e = compute_denominator_for_lex(node_tads)
        print(f"[mode=LexBoundaryFirst] denom={denom}, max_starts={max_s}, max_ends={max_e}")

    dist = {}
    keys = list(node_tads.keys())
    for u, v in itertools.combinations(keys, 2):
        d = compute_tad_distance(
            node_tads[u],
            node_tads[v],
            tolerance=tolerance,
            Ct=Ct,
            Cb=Cb,
            denom=denom
        )
        dist[(u, v)] = float(d)
        dist[(v, u)] = float(d)
    # self distances
    for k in keys:
        dist[(k, k)] = 0.0
    return dist

def build_mst(node_tads, dist):
    """
    Build MST from full graph (nodes in node_tads).
    """
    G = nx.Graph()
    for node in node_tads:
        G.add_node(node)
    for (u, v), d in dist.items():
        # avoid adding duplicate edges since dist is symmetric
        if u < v:
            G.add_edge(u, v, weight=d)
    mst = nx.minimum_spanning_tree(G)
    return mst

def plot_mst(mst, output_path):
    pos = nx.spring_layout(mst, seed=1, k=1.5)
    plt.figure(figsize=(10,10))
    nx.draw(
        mst, pos,
        with_labels=True,
        node_size=2500,
        node_color="lightblue",
        font_size=10
    )
    edge_labels = {(u,v): f"{d['weight']:.1f}" for u,v,d in mst.edges(data=True)}
    nx.draw_networkx_edge_labels(mst, pos, edge_labels=edge_labels, label_pos=0.5, font_size=9, rotate=False)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.title("Minimum Spanning Tree (TAD distance)")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print("Saved:", output_path)

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--resolution", required=True)
    parser.add_argument("--tolerance", type=int, default=0)
    parser.add_argument("--mode", choices=["convex", "LexBoundaryFirst"], default="convex",
                        help="convex or lex_boundary_first (prioritize boundaries)")
    parser.add_argument("--Ct", type=float, default=1.0, help="Ct weight for TAD RF")
    parser.add_argument("--Cb", type=float, default=1.0, help="Cb weight for boundary RF")
    args = parser.parse_args()

    chrom = args.chrom
    res = args.resolution
    tolerance = args.tolerance
    mode = args.mode
    Ct = args.Ct
    Cb = args.Cb

    print(f"Running MST analysis: chrom={chrom}, res={res}, tol={tolerance}, mode={mode}, Ct={Ct}, Cb={Cb}")

    # Read TADs
    node_tads = read_node_tads(chrom, res)
    if not node_tads:
        print("No TAD files found; exiting.")
        return

    # Compute distances
    dist = compute_distance_matrix(node_tads, tolerance, Ct=Ct, Cb=Cb, mode=mode)

    # Build MST
    mst = build_mst(node_tads, dist)

    # Save figure
    out_path = f"msts/chr{chrom}/{res}/mstree_tol_{tolerance}_{mode}.png"
    plot_mst(mst, out_path)

    # ---------------- Compute tree edit distance ----------------
    ref_tree = build_reference_tree()
    ted = tree_edit_distance(mst, ref_tree, root="LT")
    print(f"Tree edit distance from MST to reference tree: {ted}")

    # Save distance in txt
    out_txt = f"msts/chr{chrom}/{res}/tree_distance_tol_{tolerance}_{mode}.txt"
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w") as f:
        f.write(f"{ted}\n")
    print("Saved tree distance:", out_txt)


if __name__ == "__main__":
    main()