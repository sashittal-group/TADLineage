"""Infer internal-node TADs with a Fitch-Hartigan style pass.

This script treats each possible TAD interval as a binary character across the
leaf nodes. It then performs:

1. a bottom-up majority pass, and
2. a top-down resolution pass,

to infer which intervals are present at each internal node.

The input format is intentionally simple and matches the style used in the
existing TAD scripts in this workspace:

    edges = [(0, 1), (0, 2), (0, 3)]
    leaf_intervals = {
        1: ['[3,6]'],
        2: ['[1,5]', '[2,5]'],
        3: ['[2,4]', '[2,5]'],
    }

By default, the script runs that example if no arguments are supplied.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from tad_utilities import boundaries_from_tads, parse_interval, relaxed_boundary_rf, relaxed_tad_rf


Interval = Tuple[int, int]
State = int | None


def normalize_leaf_intervals(leaf_intervals: Mapping[int, Iterable[str]]) -> Dict[int, Set[Interval]]:
    """Convert leaf interval strings to sets of (start, end) tuples."""
    normalized: Dict[int, Set[Interval]] = {}
    for node, intervals in leaf_intervals.items():
        normalized[node] = {parse_interval(interval) for interval in intervals}
    return normalized


def build_tree(edges: Sequence[Tuple[int, int]]) -> Tuple[Dict[int, List[int]], Dict[int, int], List[int], int]:
    """Build child/parent maps and identify the root."""
    children: Dict[int, List[int]] = defaultdict(list)
    parent: Dict[int, int] = {}
    nodes: Set[int] = set()

    for u, v in edges:
        children[u].append(v)
        parent[v] = u
        nodes.add(u)
        nodes.add(v)

    roots = sorted(node for node in nodes if node not in parent)
    if not roots:
        raise ValueError("Could not identify a root from the provided edges.")
    if len(roots) > 1:
        raise ValueError(f"Expected a single rooted tree, found multiple roots: {roots}")

    root = roots[0]
    return children, parent, sorted(nodes), root


def postorder_nodes(root: int, children: Mapping[int, Sequence[int]]) -> List[int]:
    order: List[int] = []

    def dfs(node: int) -> None:
        for child in children.get(node, []):
            dfs(child)
        order.append(node)

    dfs(root)
    return order


def preorder_nodes(root: int, children: Mapping[int, Sequence[int]]) -> List[int]:
    order: List[int] = []

    def dfs(node: int) -> None:
        order.append(node)
        for child in children.get(node, []):
            dfs(child)

    dfs(root)
    return order


def infer_internal_tads_fitch_hartigan(
    edges: Sequence[Tuple[int, int]],
    leaf_intervals: Mapping[int, Iterable[str]],
) -> Dict[int, Set[Interval]]:
    """Infer TAD sets for every node using the user's FH-style rules.

    Bottom-up rule:
    - leaves are 1 if they contain the interval, else 0
    - an internal node becomes 1 if more children are 1 than 0
    - it becomes 0 if more children are 0 than 1
    - ties produce an ambiguous state {0,1}

    Top-down rule:
    - if the root is ambiguous, assign 0
    - definite 0/1 states stay fixed
    - ambiguous nodes inherit their parent's state

    Ambiguous child states are treated as contributing to both counts, so they do
    not bias the majority.
    """
    children, parent, nodes, root = build_tree(edges)
    leaf_sets = normalize_leaf_intervals(leaf_intervals)

    candidate_intervals = sorted(set().union(*leaf_sets.values())) if leaf_sets else []
    if not candidate_intervals:
        return {node: set() for node in nodes}

    node_states: Dict[int, Dict[Interval, State]] = {node: {} for node in nodes}

    for node in postorder_nodes(root, children):
        if node in leaf_sets:
            for interval in candidate_intervals:
                node_states[node][interval] = 1 if interval in leaf_sets[node] else 0
            continue

        child_list = children.get(node, [])
        for interval in candidate_intervals:
            ones = 0
            zeros = 0
            for child in child_list:
                state = node_states[child].get(interval, 0)
                if state == 1:
                    ones += 1
                elif state == 0:
                    zeros += 1
                else:
                    ones += 1
                    zeros += 1

            if ones > zeros:
                node_states[node][interval] = 1
            elif zeros > ones:
                node_states[node][interval] = 0
            else:
                node_states[node][interval] = None

    for node in preorder_nodes(root, children):
        for interval in candidate_intervals:
            state = node_states[node].get(interval, 0)
            if node == root and state is None:
                node_states[node][interval] = 0
            elif state is None:
                node_states[node][interval] = node_states[parent[node]][interval]

    inferred: Dict[int, Set[Interval]] = {node: set() for node in nodes}
    for node in nodes:
        for interval, state in node_states[node].items():
            if state == 1:
                inferred[node].add(interval)

    return inferred


def infer_tree_rf_distance(
    edges: Sequence[Tuple[int, int]],
    node_tads: Mapping[int, Set[Interval]],
    tolerance: int = 0,
) -> Dict[str, object]:
    """Compute a tree-style RF score from parent-child TAD differences."""
    total_tad_rf = 0
    total_boundary_rf = 0
    per_edge = []

    for u, v in edges:
        tu = node_tads.get(u, set())
        tv = node_tads.get(v, set())

        tad_rf = relaxed_tad_rf(tu, tv, tol=tolerance)
        su, eu = boundaries_from_tads(tu)
        sv, ev = boundaries_from_tads(tv)
        boundary_rf = relaxed_boundary_rf(su, sv, eu, ev, tol=tolerance)

        total_tad_rf += tad_rf
        total_boundary_rf += boundary_rf
        per_edge.append(
            {
                "edge": [u, v],
                "tad_rf": tad_rf,
                "boundary_rf": boundary_rf,
                "cost": tad_rf + boundary_rf,
            }
        )

    return {
        "total_tad_rf": total_tad_rf,
        "total_boundary_rf": total_boundary_rf,
        "total_rf": total_tad_rf + total_boundary_rf,
        "per_edge": per_edge,
    }


def to_jsonable(node_tads: Mapping[int, Set[Interval]]) -> Dict[str, List[str]]:
    """Format node assignments for display."""
    formatted: Dict[str, List[str]] = {}
    for node, intervals in sorted(node_tads.items()):
        formatted[str(node)] = [f"[{a},{b}]" for a, b in sorted(intervals)]
    return formatted


def load_literal(value: str):
    """Parse a Python literal from a file path."""
    with open(value) as handle:
        text = handle.read().strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def parse_input_or_literal(value: str | None, fallback):
    if value is None:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return load_literal(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer internal-node TADs with a Fitch-Hartigan style majority pass."
    )
    parser.add_argument(
        "--edges",
        help="Python literal or JSON for the tree edges, e.g. '[(0,1),(0,2)]'.",
    )
    parser.add_argument(
        "--leaf-intervals",
        help="Python literal or JSON for leaf interval assignments.",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=0,
        help="Optional tolerance for the RF-style score.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result as JSON instead of a human-readable summary.",
    )
    args = parser.parse_args()

    default_edges = [(0, 1), (0, 2), (1, 3), (1, 4), (1, 5), (2, 6), (2, 7)]
    default_leaf_intervals = {
        3: ["[1,3]","[3,6]"], 
        4: ["[1,3]", "[4,10]"],
        5: ["[1,3]", "[2,3]"],
        6: ["[1,5]", "[4,10]"],
        7: ["[1,5]", "[2,5]"],
    }

    edges = parse_input_or_literal(args.edges, default_edges)
    leaf_intervals = parse_input_or_literal(args.leaf_intervals, default_leaf_intervals)

    edges = [tuple(edge) for edge in edges]
    leaf_intervals = {int(node): list(intervals) for node, intervals in leaf_intervals.items()}

    inferred = infer_internal_tads_fitch_hartigan(edges, leaf_intervals)
    rf_score = infer_tree_rf_distance(edges, inferred, tolerance=args.tolerance)

    if args.json:
        print(
            json.dumps(
                {
                    "edges": edges,
                    "leaf_intervals": leaf_intervals,
                    "inferred_node_intervals": to_jsonable(inferred),
                    "rf_score": rf_score,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    print("Input edges:")
    print(edges)
    print("\nLeaf intervals:")
    for node in sorted(leaf_intervals):
        print(f"  {node}: {leaf_intervals[node]}")

    print("\nInferred node intervals (Fitch-Hartigan style):")
    for node in sorted(inferred):
        print(f"  {node}: {[f'[{a},{b}]' for a, b in sorted(inferred[node])]}")

    print("\nTree RF-style score:")
    print(f"  TAD RF:      {rf_score['total_tad_rf']}")
    print(f"  Boundary RF: {rf_score['total_boundary_rf']}")
    print(f"  Total RF:    {rf_score['total_rf']}")


if __name__ == "__main__":
    main()