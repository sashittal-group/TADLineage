#!/usr/bin/env python3
"""
Create a tree figure for a hemato edge-event summary across tolerances.

The plot averages total merge/split events per edge over all supplied
summary JSON files for one chromosome-resolution-leaf combination and
draws the counts directly on the tree edges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: str | Path) -> dict:
    with Path(path).open() as fh:
        return json.load(fh)


def aggregate_summaries(summaries: list[dict]) -> dict:
    if not summaries:
        raise ValueError("No summaries supplied")

    edge_values = defaultdict(list)
    for summary in summaries:
        for edge_name, edge_data in summary.get("edges", {}).items():
            edge_values[edge_name].append(int(edge_data.get("total_events", 0)))

    if not edge_values:
        raise ValueError("No edge data found in supplied summaries")

    averaged_edges = {}
    for edge_name, values in edge_values.items():
        averaged_edges[edge_name] = {
            "mean_total_events": float(np.mean(values)),
            "rounded_total_events": int(round(float(np.mean(values)))),
            "n_tolerances": len(values),
        }

    first = summaries[0]
    tolerance_values = [summary.get("tolerance") for summary in summaries if summary.get("tolerance") is not None]

    return {
        "variant": first.get("variant", "unknown"),
        "chrom": first.get("chrom", "?"),
        "resolution": first.get("resolution", "?"),
        "leaf1": first.get("leaf1", "?"),
        "leaf2": first.get("leaf2", "?"),
        "tad_kind": first.get("tad_kind", "all"),
        "tolerances": tolerance_values,
        "edges": averaged_edges,
    }


def _tree_layout(summary: dict) -> tuple[dict[str, tuple[float, float]], list[tuple[str, str, str]], list[tuple[float, float]]]:
    leaf1 = summary.get("leaf1", "leaf1")
    leaf2 = summary.get("leaf2", "leaf2")
    variant = summary.get("variant", "unknown")

    if variant == "inferred":
        nodes = {
            "MPP": (0.0, 3.0),
            "CLP": (-2.0, 1.4),
            "CMP": (2.0, 1.4),
            leaf1: (1.0, 0.0),
            leaf2: (3.0, 0.0),
        }
        edges = [
            ("MPP", "CLP", "MPP-CLP"),
            ("MPP", "CMP", "MPP-CMP"),
            ("CMP", leaf1, f"CMP-{leaf1}"),
            ("CMP", leaf2, f"CMP-{leaf2}"),
        ]
        node_labels = ["MPP", "CLP", "CMP", leaf1, leaf2]
    else:
        nodes = {
            "LT": (0.0, 4.2),
            "MPP": (0.0, 3.0),
            "CLP": (-2.0, 1.4),
            "CMP": (2.0, 1.4),
            leaf1: (1.0, 0.0),
            leaf2: (3.0, 0.0),
        }
        edges = [
            ("LT", "MPP", "LT-MPP"),
            ("MPP", "CLP", "MPP-CLP"),
            ("MPP", "CMP", "MPP-CMP"),
            ("CMP", leaf1, f"CMP-{leaf1}"),
            ("CMP", leaf2, f"CMP-{leaf2}"),
        ]
        node_labels = ["LT", "MPP", "CLP", "CMP", leaf1, leaf2]

    return nodes, edges, [nodes[name] for name in node_labels]


def _draw_tree(ax, summary: dict) -> list[str]:
    nodes, edges, _ = _tree_layout(summary)
    edge_order = []

    for parent, child, edge_name in edges:
        edge_order.append(edge_name)
        x0, y0 = nodes[parent]
        x1, y1 = nodes[child]
        ax.plot([x0, x1], [y0, y1], color="#4C78A8", linewidth=2.5, solid_capstyle="round")

    edges_data = summary.get("edges", {})
    for parent, child, edge_name in edges:
        x0, y0 = nodes[parent]
        x1, y1 = nodes[child]
        edge_data = edges_data.get(edge_name, {})
        value = edge_data.get("rounded_total_events", edge_data.get("total_events", 0))
        mid_x = (x0 + x1) / 2.0
        mid_y = (y0 + y1) / 2.0

        dx = x1 - x0
        dy = y1 - y0
        norm = float(np.hypot(dx, dy)) or 1.0
        offset_x = -dy / norm * 0.12
        offset_y = dx / norm * 0.12

        ax.text(
            mid_x + offset_x,
            mid_y + offset_y,
            f"{int(round(float(value)))}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#888888", "lw": 0.8},
        )

    for label, (x, y) in nodes.items():
        ax.scatter([x], [y], s=180, color="#F2F2F2", edgecolor="black", zorder=3)
        ax.text(x, y + 0.22, label, ha="center", va="bottom", fontsize=11, fontweight="bold")

    xs = [coord[0] for coord in nodes.values()]
    ys = [coord[1] for coord in nodes.values()]
    pad_x = 0.8
    pad_y = 0.7
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return edge_order


def make_plot(summary: dict, output_path: str | Path) -> dict:
    edges = summary.get("edges", {})
    if not edges:
        raise ValueError("Summary JSON does not contain any edges")

    variant = summary.get("variant", "unknown")
    chrom = summary.get("chrom", "?")
    resolution = summary.get("resolution", "?")
    leaf1 = summary.get("leaf1", "?")
    leaf2 = summary.get("leaf2", "?")

    fig, ax = plt.subplots(figsize=(10, 6))
    edge_order = _draw_tree(ax, summary)

    mean_events = [float(edges[name].get("mean_total_events", edges[name].get("total_events", 0))) for name in edge_order]
    rounded_average = int(round(float(np.mean(mean_events))))

    ax.set_title(
        f"{variant}: chr{chrom} {resolution} {leaf1}-{leaf2}\n"
        f"Average merge/split events across tolerances = {rounded_average}",
        fontweight="bold",
    )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "output_file": str(output_path),
        "rounded_average_total_events": rounded_average,
        "edge_totals": dict(zip(edge_order, mean_events)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot merge/split event averages from edge summary JSONs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="one or more edge summary JSON paths")
    parser.add_argument("--output", required=True, help="output PNG path")
    args = parser.parse_args()

    summaries = [load_summary(path) for path in args.inputs]
    summary = aggregate_summaries(summaries)
    result = make_plot(summary, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()