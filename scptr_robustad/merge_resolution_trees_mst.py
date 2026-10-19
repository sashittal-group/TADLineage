#!/usr/bin/env python3

import argparse
import ast
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

PREFERRED_EDGES = {
    ("1", "2"),
    ("2", "4"),
    ("4", "8"),
    ("8", "64"),
}


def edge_key(u, v):
    u = str(u)
    v = str(v)
    return tuple(sorted((u, v), key=lambda x: (len(x), x)))


def load_aggregated_edge_weights(csv_path, target_resolution, preferred_edge_factor):
    edge_weight_sum = defaultdict(float)
    edge_count = defaultdict(int)
    row_count = 0

    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("resolution") != target_resolution:
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
                if ek in PREFERRED_EDGES:
                    weight *= preferred_edge_factor
                edge_weight_sum[ek] += weight
                edge_count[ek] += 1

    edge_weight_avg = {
        ek: edge_weight_sum[ek] / edge_count[ek]
        for ek in edge_weight_sum
        if edge_count[ek] > 0
    }

    return edge_weight_avg, row_count


def build_graph(edge_weight_avg):
    graph = nx.Graph()
    for (u, v), w in edge_weight_avg.items():
        graph.add_edge(u, v, weight=w)
    return graph


def print_graph_edges(graph, title):
    print(title)
    for u, v, data in sorted(graph.edges(data=True), key=lambda x: (x[0], x[1])):
        print(f"  {u}-{v}: {data['weight']:.6f}")


def draw_mst(mst, resolution, output_path):
    ordered_nodes = sorted(mst.nodes(), key=lambda x: (len(str(x)), str(x)))
    pos = nx.spring_layout(mst, seed=42)

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(
        mst,
        pos,
        nodelist=ordered_nodes,
        node_size=2200,
        node_color="#dbeafe",
        edgecolors="#1e3a8a",
        linewidths=1.5,
        ax=ax,
    )
    nx.draw_networkx_edges(mst, pos, width=2.5, edge_color="#475569", ax=ax)
    nx.draw_networkx_labels(mst, pos, font_size=12, font_weight="bold", ax=ax)

    edge_labels = {(u, v): f"{data['weight']:.1f}" for u, v, data in mst.edges(data=True)}
    nx.draw_networkx_edge_labels(mst, pos, edge_labels=edge_labels, font_size=10, ax=ax)

    ax.set_title(f"Merged MST for {resolution}")
    ax.set_axis_off()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Merge spanning trees for one resolution and build an MST from summed edge weights."
    )
    parser.add_argument(
        "--csv",
        default="results_updated/scptr_tree_distance/all_tree_distances.csv",
        help="Aggregated CSV from aggregate_tree_distance_json.py",
    )
    parser.add_argument(
        "--resolution",
        required=True,
        help="Resolution to include, for example 500kb or 1000kb",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG for the merged MST",
    )
    parser.add_argument(
        "--preferred-edge-factor",
        type=float,
        default=0.5,
        help="Multiply weights of preferred edges by this factor before aggregation",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    output_path = Path(args.output) if args.output else Path(
        f"results_updated/scptr_tree_distance/merged_mst_{args.resolution}.png"
    )

    edge_weight_avg, used_rows = load_aggregated_edge_weights(
        csv_path,
        args.resolution,
        preferred_edge_factor=args.preferred_edge_factor,
    )
    if used_rows == 0:
        raise SystemExit(f"No rows found for resolution={args.resolution}")
    if not edge_weight_avg:
        raise SystemExit("No valid spanning_tree edges found for filtered rows")

    aggregated_graph = build_graph(edge_weight_avg)

    print(f"Rows used: {used_rows}")
    print(f"Preferred edge factor: {args.preferred_edge_factor}")
    print(f"Unique aggregated edges: {aggregated_graph.number_of_edges()}")
    print_graph_edges(aggregated_graph, "\nAggregated graph edges (average weights):")

    if not nx.is_connected(aggregated_graph):
        raise SystemExit("Aggregated graph is disconnected; cannot produce a single MST.")

    mst = nx.minimum_spanning_tree(aggregated_graph, weight="weight")

    print_graph_edges(mst, "\nMinimum spanning tree edges:")
    total_weight = sum(data["weight"] for _, _, data in mst.edges(data=True))
    print(f"\nMST total weight: {total_weight:.6f}")

    draw_mst(mst, args.resolution, output_path)
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
