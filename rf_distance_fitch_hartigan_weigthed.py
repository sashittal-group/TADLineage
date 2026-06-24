"""Weighted Sankoff-style TAD inference and RF scoring.

This mirrors rf_distance_fitch_hartigan.py but accepts weighted leaf TAD
importance scores. For every candidate interval, a two-state Sankoff DP is
run bottom-up with leaf costs derived from the observed weights and internal
nodes constrained to binary states (0/1).

Input leaf format examples supported:
- list of strings: ["[1,3]:0.7", "[4,6]:0.2"]
- mapping: {"[1,3]": 0.7, "[4,6]": 0.2}

If no args are provided, a small default example is run.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from tad_utilities import boundaries_from_tads, parse_interval, relaxed_boundary_rf, relaxed_tad_rf

Interval = Tuple[int, int]


def normalize_weighted_leaf_intervals(leaf_intervals: Mapping[int, Iterable]) -> Dict[int, Dict[Interval, float]]:
    """Return mapping node -> {interval: weight}.

    Accepts either lists of strings like "[1,3]:0.7" or mappings
    {"[1,3]": 0.7} as leaf interval representations.
    """
    normalized: Dict[int, Dict[Interval, float]] = {}
    for node, intervals in leaf_intervals.items():
        node_map: Dict[Interval, float] = {}
        # If intervals is a mapping (dict-like)
        if isinstance(intervals, dict):
            for interval_str, weight in intervals.items():
                interval = parse_interval(interval_str)
                node_map[interval] = float(weight)
        else:
            # Expect iterable of strings like "[1,3]:0.7"
            for entry in intervals:
                if isinstance(entry, str) and ":" in entry:
                    interval_str, weight_str = entry.split(":", 1)
                    interval = parse_interval(interval_str)
                    node_map[interval] = float(weight_str)
                else:
                    # Fallback: try to parse as interval with implicit weight 1.0
                    interval = parse_interval(str(entry))
                    node_map[interval] = 1.0

        normalized[int(node)] = node_map
    return normalized


def build_tree(edges: Sequence[Tuple[int, int]]) -> Tuple[Dict[int, List[int]], Dict[int, int], List[int], int]:
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


def _traceback_sankoff_states(
    root: int,
    children: Mapping[int, Sequence[int]],
    dp: Mapping[int, Mapping[int, float]],
) -> Dict[int, int]:
    """Recover one optimal binary assignment from Sankoff DP tables."""

    assigned: Dict[int, int] = {}

    def recurse(node: int, parent_state: int | None = None) -> None:
        if parent_state is None:
            state = 0 if dp[node][0] <= dp[node][1] else 1
        else:
            cost0 = dp[node][0] + abs(parent_state - 0)
            cost1 = dp[node][1] + abs(parent_state - 1)
            state = 0 if cost0 <= cost1 else 1

        assigned[node] = state
        for child in children.get(node, []):
            recurse(child, state)

    recurse(root)
    return assigned


def infer_internal_tads_weighted(
    edges: Sequence[Tuple[int, int]],
    leaf_intervals: Mapping[int, Iterable],
) -> Tuple[Dict[int, Dict[Interval, float]], Dict[int, Set[Interval]]]:
    """Infer weighted importance for every candidate interval.

    For each interval, leaf nodes receive costs based on the observed weight:
    state 0 costs |0 - w| and state 1 costs |1 - w|. Internal nodes use the
    standard two-state Sankoff recurrence with transition cost |parent-child|.

    Returns two objects:
    - node_weights: mapping node -> {interval: chosen_state_cost}
    - assigned_intervals: mapping node -> set(intervals) assigned state 1
    """
    children, parent, nodes, root = build_tree(edges)
    leaf_maps = normalize_weighted_leaf_intervals(leaf_intervals)

    candidate_intervals = sorted(set().union(*[set(m.keys()) for m in leaf_maps.values()])) if leaf_maps else []
    if not candidate_intervals:
        return ({node: {} for node in nodes}, {node: set() for node in nodes})

    node_weights: Dict[int, Dict[Interval, float]] = {node: {} for node in nodes}
    assigned: Dict[int, Set[Interval]] = {node: set() for node in nodes}

    ordered_nodes = postorder_nodes(root, children)

    for interval in candidate_intervals:
        dp: Dict[int, Dict[int, float]] = {node: {0: 0.0, 1: 0.0} for node in nodes}

        for node in ordered_nodes:
            child_list = children.get(node, [])
            if not child_list:
                weight = float(leaf_maps.get(node, {}).get(interval, 0.0))
                dp[node][0] = abs(0.0 - weight)
                dp[node][1] = abs(1.0 - weight)
                continue

            for state in (0, 1):
                total = 0.0
                for child in child_list:
                    child_cost = min(
                        dp[child][0] + abs(state - 0),
                        dp[child][1] + abs(state - 1),
                    )
                    total += child_cost
                dp[node][state] = total

        optimal_states = _traceback_sankoff_states(root, children, dp)
        for node in nodes:
            state = optimal_states[node]
            node_weights[node][interval] = dp[node][state]
            if state == 1:
                assigned[node].add(interval)

    return node_weights, assigned


def infer_tree_rf_distance(
    edges: Sequence[Tuple[int, int]],
    node_tads: Mapping[int, Set[Interval]],
    tolerance: int = 0,
) -> Dict[str, object]:
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


def to_jsonable_intervals(node_weights: Mapping[int, Dict[Interval, float]]) -> Dict[str, Dict[str, float]]:
    formatted: Dict[str, Dict[str, float]] = {}
    for node, weights in sorted(node_weights.items()):
        formatted[str(node)] = {f"[{a},{b}]": float(w) for (a, b), w in sorted(weights.items())}
    return formatted


def load_literal(value: str):
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
    parser = argparse.ArgumentParser(description="Infer weighted internal-node TADs and RF score.")
    parser.add_argument("--edges", help="Python literal or JSON for the tree edges, e.g. '[(0,1),(0,2)]'.")
    parser.add_argument("--leaf-intervals", help="Python literal or JSON for leaf interval assignments (weighted).")
    parser.add_argument("--tolerance", type=int, default=0, help="Optional tolerance for the RF-style score.")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON instead of a human-readable summary.")
    args = parser.parse_args()

    default_edges = [(0, 1), (0, 2), (1, 3), (1, 4), (1, 5), (2, 6), (2, 7)]
    default_leaf_intervals = {
        3: ["[1,3]:0.2","[3,6]:0.6"],
        4: ["[1,3]:0.2", "[4,10]:0.3"],
        5: ["[1,3]:0.2", "[2,3]:0.4"],
        6: ["[1,3]:0.2", "[4,10]:0.6"],
        7: ["[1,3]:0.4", "[2,3]:0.6"],
    }

    edges = parse_input_or_literal(args.edges, default_edges)
    leaf_intervals = parse_input_or_literal(args.leaf_intervals, default_leaf_intervals)

    edges = [tuple(edge) for edge in edges]

    node_weights, inferred = infer_internal_tads_weighted(edges, leaf_intervals)
    rf_score = infer_tree_rf_distance(edges, inferred, tolerance=args.tolerance)

    if args.json:
        print(
            json.dumps(
                {
                    "edges": edges,
                    "leaf_intervals": leaf_intervals,
                    "node_interval_weights": to_jsonable_intervals(node_weights),
                    "inferred_node_intervals": {str(k): [f"[{a},{b}]" for a,b in sorted(v)] for k,v in inferred.items()},
                    "rf_score": rf_score,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    print("Input edges:")
    print(edges)
    print("\nLeaf intervals (weighted):")
    for node in sorted(leaf_intervals):
        print(f"  {node}: {leaf_intervals[node]}")

    print("\nNode interval costs (Sankoff DP):")
    for node in sorted(node_weights):
        weights = {f"[{a},{b}]": w for (a, b), w in sorted(node_weights[node].items())}
        print(f"  {node}: {weights}")

    print("\nAssigned intervals (state 1):")
    for node in sorted(inferred):
        print(f"  {node}: {[f'[{a},{b}]' for a,b in sorted(inferred[node])]}")

    internal_nodes = [node for node in sorted(inferred) if node not in leaf_intervals]
    print("\nInternal-node assigned intervals:")
    for node in internal_nodes:
        print(f"  {node}: {[f'[{a},{b}]' for a,b in sorted(inferred[node])]}")

    print("\nTree RF-style score:")
    print(f"  TAD RF:      {rf_score['total_tad_rf']}")
    print(f"  Boundary RF: {rf_score['total_boundary_rf']}")
    print(f"  Total RF:    {rf_score['total_rf']}")


if __name__ == "__main__":
    main()
