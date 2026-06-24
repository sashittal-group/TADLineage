import argparse
import ast
import csv
from collections import defaultdict
from pathlib import Path

import networkx as nx


def edge_key(u, v):
    u = str(u)
    v = str(v)
    return tuple(sorted((u, v), key=lambda x: (len(x), x)))


def load_aggregated_edge_weights(
    csv_path,
    target_resolution="1000kb",
    target_tree_distance=0.0,
):
    edge_weight_sum = defaultdict(float)
    row_count = 0

    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("resolution") != target_resolution:
                continue

            try:
                tree_distance = float(row.get("tree_distance", "nan"))
            except Exception:
                continue
            if tree_distance > target_tree_distance:
                continue

            row_count += 1
            spanning_tree_raw = row.get("spanning_tree", "")
            if not spanning_tree_raw:
                continue

            try:
                spanning_tree = ast.literal_eval(spanning_tree_raw)
            except Exception:
                continue

            if not isinstance(spanning_tree, list):
                continue

            for edge in spanning_tree:
                if not isinstance(edge, dict):
                    continue

                u = edge.get("source")
                v = edge.get("target")
                w = edge.get("weight")
                if u is None or v is None or w is None:
                    continue

                try:
                    weight = float(w)
                except Exception:
                    continue

                ek = edge_key(u, v)
                edge_weight_sum[ek] += weight

    return edge_weight_sum, row_count


def build_graph(edge_weight_sum):
    g = nx.Graph()
    for (u, v), w in edge_weight_sum.items():
        g.add_edge(u, v, weight=w)
    return g


def print_graph_edges(g, title):
    print(title)
    for u, v, data in sorted(g.edges(data=True), key=lambda x: (x[0], x[1])):
        print(f"  {u}-{v}: {data['weight']:.6f}")


def main():

    csv_path = Path("results/scptr_tree_distance/all_tree_distances.csv")
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    edge_weight_sum, used_rows = load_aggregated_edge_weights(
        csv_path,
        target_resolution="1000kb",
        target_tree_distance=0.0,
    )
    if used_rows == 0:
        raise SystemExit(
            f"No rows found"
        )
    if not edge_weight_sum:
        raise SystemExit("No valid spanning_tree edges found for filtered rows")

    aggregated_graph = build_graph(edge_weight_sum)

    print(
        f"Rows used : {used_rows}"
    )
    print(f"Unique aggregated edges: {aggregated_graph.number_of_edges()}")
    print_graph_edges(aggregated_graph, "\nAggregated graph edges (summed weights):")

    if not nx.is_connected(aggregated_graph):
        raise SystemExit("Aggregated graph is disconnected; cannot produce a single MST.")

    mst = nx.minimum_spanning_tree(aggregated_graph, weight="weight")

    print_graph_edges(mst, "\nMinimum spanning tree edges:")
    total_weight = sum(data["weight"] for _, _, data in mst.edges(data=True))
    print(f"\nMST total weight: {total_weight:.6f}")


if __name__ == "__main__":
    main()
