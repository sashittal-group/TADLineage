"""Reusable MGR-style tree construction helpers for TAD sets."""

import os

from nussinov import infer_tads_with_dp
from tad_utilities import parse_interval, read_tad_intervals


def load_tad_set(path, resolution=None):
    """Load a .tad file as a set of (start, end) tuples."""
    return {parse_interval(s) for s in read_tad_intervals(path, resolution=resolution)}


def load_leaf_tads_from_paths(leaf_paths, resolution=None):
    """Load a sequence of leaf paths into a numeric node-id mapping."""
    return {
        idx + 1: load_tad_set(path, resolution=resolution)
        for idx, path in enumerate(leaf_paths)
    }


def compute_median_from_children(child_sets, Ct=1.0, Cb=1.0):
    """
    Call the DP median solver on an arbitrary number of children by building
    a temporary star tree.
    """
    edges = [(0, i + 1) for i in range(len(child_sets))]

    leaf_intervals = {}
    all_boundaries = set()

    for i, child_set in enumerate(child_sets):
        leaf_intervals[i + 1] = [f"[{a},{b}]" for (a, b) in child_set]
        for a, b in child_set:
            all_boundaries.add(a)
            all_boundaries.add(b)

    boundaries = sorted(all_boundaries)

    intervals, score = infer_tads_with_dp(
        edges,
        leaf_intervals,
        boundaries,
        boundaries,
        Ct=Ct,
        Cb=Cb,
        trace=False,
    )

    return intervals, score


def tad_distance(a_set, b_set, Ct=1.0, Cb=1.0):
    """Compute an RF-style distance between two TAD sets."""
    a_set = set(a_set)
    b_set = set(b_set)

    a_starts = {a for (a, b) in a_set}
    a_ends = {b for (a, b) in a_set}
    b_starts = {a for (a, b) in b_set}
    b_ends = {b for (a, b) in b_set}

    return (
        Ct * (len(a_set - b_set) + len(b_set - a_set))
        + Cb * (len(a_starts - b_starts) + len(b_starts - a_starts))
        + Cb * (len(a_ends - b_ends) + len(b_ends - a_ends))
    )


def build_tree_mgr(leaf_tads, Ct=1.0, Cb=1.0):
    """
    Build an MGR-style greedy tree using exact median inference.

    Parameters
    ----------
    leaf_tads : dict
        Mapping from node id to a set of (start, end) TAD intervals.
    Ct, Cb : float
        Interval and boundary weights passed to the median solver.
    """
    if len(leaf_tads) < 3:
        raise ValueError("Need at least three leaves to build an MGR tree.")

    nodes = {node_id: set(tads) for node_id, tads in leaf_tads.items()}
    next_node_id = max(nodes.keys()) + 1

    leaves = sorted(nodes.keys())
    seed_leaves = leaves[:3]

    root_tads, _ = compute_median_from_children(
        [nodes[seed_leaves[0]], nodes[seed_leaves[1]], nodes[seed_leaves[2]]],
        Ct,
        Cb,
    )

    root = next_node_id
    next_node_id += 1
    nodes[root] = root_tads

    edges = [
        (root, seed_leaves[0]),
        (root, seed_leaves[1]),
        (root, seed_leaves[2]),
    ]

    remaining = [node_id for node_id in leaves if node_id not in seed_leaves]

    for leaf in remaining:
        best_cost = float("inf")
        best_edge = None
        best_median = None

        for u, v in edges:
            median_tads, _ = compute_median_from_children(
                [nodes[u], nodes[v], nodes[leaf]],
                Ct,
                Cb,
            )

            cost = (
                tad_distance(median_tads, nodes[u], Ct, Cb)
                + tad_distance(median_tads, nodes[v], Ct, Cb)
                + tad_distance(median_tads, nodes[leaf], Ct, Cb)
                - tad_distance(nodes[u], nodes[v], Ct, Cb)
            )

            if cost < best_cost:
                best_cost = cost
                best_edge = (u, v)
                best_median = median_tads

        if best_edge is None:
            raise RuntimeError("Failed to identify an insertion edge for the next leaf.")

        u, v = best_edge
        new_node = next_node_id
        next_node_id += 1
        nodes[new_node] = best_median

        edges.remove((u, v))
        edges.append((u, new_node))
        edges.append((new_node, v))
        edges.append((new_node, leaf))

    return edges, nodes


def format_tree_edges(edges, node_names=None):
    """Render edges with optional node-name mapping."""
    node_names = node_names or {}
    return [(node_names.get(u, u), node_names.get(v, v)) for u, v in edges]


def write_tree_summary(edges, nodes, output_path, node_names=None):
    """Write a compact text summary of the inferred tree."""
    node_names = node_names or {}
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as handle:
        handle.write("Edges\n")
        for u, v in format_tree_edges(edges, node_names=node_names):
            handle.write(f"{u}\t{v}\n")

        handle.write("\nNodes\n")
        for node_id in sorted(nodes):
            label = node_names.get(node_id, node_id)
            handle.write(f"{label}\t{sorted(nodes[node_id])}\n")


if __name__ == "__main__":
    leaf_tads = {
        1: {(1, 5), (6, 10)},
        2: {(1, 5), (6, 10)},
        3: {(1, 4), (5, 10)},
        4: {(1, 3), (4, 10)},
    }

    edges, nodes = build_tree_mgr(leaf_tads, Ct=1.0, Cb=1.0)

    print("\nFinal Tree Edges:")
    for edge in edges:
        print(edge)

    print("\nNode TAD assignments:")
    for node_id, tads in nodes.items():
        print(node_id, sorted(tads))