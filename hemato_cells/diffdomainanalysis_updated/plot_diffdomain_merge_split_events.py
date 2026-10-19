#!/usr/bin/env python3
"""
Plot whole-TAD diffDomain agreement for edge merge/split summaries.

Unlike the original plotter (which showed the fraction of *our* merge/split
events that diffDomain confirmed), this version shows the whole-TAD accuracy:
over every source TAD on an edge, how often our label (split/merge/none) matches
diffDomain's label.

Modes:
  edge      Draw one tree figure for a chr/res/leaf pair, picking the tolerance
            with the best whole-TAD agreement.
  aggregate Draw a bar chart of whole-TAD agreement accuracy grouped by
            chromosome or resolution.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: str | Path) -> dict:
    with Path(path).open() as fh:
        return json.load(fh)


def summary_accuracy(summary: dict) -> float | None:
    aggregate = summary.get("aggregate", {})
    total = int(aggregate.get("total_source_tads", 0))
    agree = int(aggregate.get("agree", 0))
    return agree / total * 100.0 if total else None


def comparable_accuracy(summary: dict) -> float:
    accuracy = summary_accuracy(summary)
    return accuracy if accuracy is not None else -1.0


def select_best_tolerance_summary(summaries: list[dict]) -> dict:
    if not summaries:
        raise ValueError("No summaries supplied")

    def sort_key(summary: dict):
        aggregate = summary.get("aggregate", {})
        total_tads = int(aggregate.get("total_source_tads", 0))
        return (comparable_accuracy(summary), total_tads, -int(summary.get("tolerance", 0)))

    best = max(summaries, key=sort_key)
    result = dict(best)
    result["selected_tolerance"] = best.get("tolerance")
    result["candidate_tolerances"] = [summary.get("tolerance") for summary in summaries]
    result["selection_mode"] = "best_accuracy"
    return result


def _tree_layout(summary: dict):
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
    return nodes, edges


def plot_edge(summary: dict, output_path: str | Path) -> dict:
    nodes, edges = _tree_layout(summary)
    edge_data = summary.get("edges", {})

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for parent, child, edge_name in edges:
        x0, y0 = nodes[parent]
        x1, y1 = nodes[child]
        ax.plot([x0, x1], [y0, y1], color="#4C78A8", linewidth=2.5, solid_capstyle="round")

        agreement = edge_data.get(edge_name, {}).get("tad_agreement", {})
        total = int(agreement.get("total_source_tads", 0))
        agree = int(agreement.get("agree", 0))
        accuracy = agree / total * 100.0 if total else None

        mid_x = (x0 + x1) / 2.0
        mid_y = (y0 + y1) / 2.0
        dx = x1 - x0
        dy = y1 - y0
        norm = float(np.hypot(dx, dy)) or 1.0
        offset_x = -dy / norm * 0.18
        offset_y = dx / norm * 0.18

        accuracy_label = f"{accuracy:.0f}%" if accuracy is not None else "N/A"
        label = f"{agree}/{total}\n{accuracy_label}"
        ax.text(
            mid_x + offset_x,
            mid_y + offset_y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#888888", "lw": 0.8},
        )

    for label, (x, y) in nodes.items():
        ax.scatter([x], [y], s=180, color="#F2F2F2", edgecolor="black", zorder=3)
        ax.text(x, y + 0.22, label, ha="center", va="bottom", fontsize=11, fontweight="bold")

    xs = [coord[0] for coord in nodes.values()]
    ys = [coord[1] for coord in nodes.values()]
    ax.set_xlim(min(xs) - 0.9, max(xs) + 0.9)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    totals = []
    agrees = []
    for _, _, edge_name in edges:
        agreement = edge_data.get(edge_name, {}).get("tad_agreement", {})
        totals.append(int(agreement.get("total_source_tads", 0)))
        agrees.append(int(agreement.get("agree", 0)))
    total = sum(totals)
    matched = sum(agrees)
    overall = matched / total * 100.0 if total else None
    overall_label = f"{overall:.1f}%" if overall is not None else "N/A"

    ax.set_title(
        f"{summary.get('variant')}: chr{summary.get('chrom')} {summary.get('resolution')} "
        f"{summary.get('leaf1')}-{summary.get('leaf2')}\n"
        f"best tolerance = {summary.get('selected_tolerance', summary.get('tolerance'))}, "
        f"whole-TAD diffDomain agreement = {overall_label}",
        fontweight="bold",
    )
    fig.text(0.02, 0.02, "label = agreeing TADs / all source TADs on the edge", fontsize=9)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "output_file": str(output_path),
        "selected_tolerance": summary.get("selected_tolerance", summary.get("tolerance")),
        "overall_accuracy_percent": overall,
    }


def plot_aggregate(summaries: list[dict], group_by: str, output_path: str | Path) -> dict:
    if group_by not in {"chrom", "resolution"}:
        raise ValueError("--group-by must be chrom or resolution")

    best_by_run = {}
    for summary in summaries:
        key = (
            summary.get("variant"),
            summary.get("chrom"),
            summary.get("resolution"),
            summary.get("leaf1"),
            summary.get("leaf2"),
        )
        current = best_by_run.get(key)
        if current is None or comparable_accuracy(summary) > comparable_accuracy(current):
            best_by_run[key] = summary

    grouped = defaultdict(lambda: {"tads": 0, "agree": 0, "runs": 0})
    for summary in best_by_run.values():
        key = str(summary.get(group_by, "unknown"))
        agg = summary.get("aggregate", {})
        total = int(agg.get("total_source_tads", 0))
        if total == 0:
            continue
        agree = int(agg.get("agree", 0))
        grouped[key]["tads"] += total
        grouped[key]["agree"] += agree
        grouped[key]["runs"] += 1

    def sort_key(value: str):
        try:
            return (0, int(value))
        except ValueError:
            order = {"100kb": 100_000, "250kb": 250_000, "500kb": 500_000, "1mb": 1_000_000, "2mb": 2_000_000}
            return (1, order.get(value, 10**12), value)

    labels = sorted(grouped, key=sort_key)
    percentages = [
        grouped[label]["agree"] / grouped[label]["tads"] * 100.0
        for label in labels
    ]

    fig_width = max(8, len(labels) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    bars = ax.bar(labels, percentages, color="#4C78A8", edgecolor="#2F4B68", linewidth=0.8) if labels else []
    # Headroom above 100 so the bar labels stay inside the axes box.
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("diffDomain agreement (%)")
    ax.set_xlabel("Chromosome" if group_by == "chrom" else "Resolution")
    ax.set_title(f"Merge/split agreement with diffDomain by {group_by}", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if labels:
        ax.bar_label(bars, labels=[f"{value:.1f}%" for value in percentages], padding=3, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No source TADs", ha="center", va="center", transform=ax.transAxes)
    if group_by == "chrom":
        ax.tick_params(axis="x", labelrotation=0)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "output_file": str(output_path),
        "group_by": group_by,
        "groups": {
            label: {
                "tads": grouped[label]["tads"],
                "agree": grouped[label]["agree"],
                "runs": grouped[label]["runs"],
                "accuracy_percent": percentages[idx],
            }
            for idx, label in enumerate(labels)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot whole-TAD diffDomain merge/split agreement summaries.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    edge = subparsers.add_parser("edge")
    edge.add_argument("--inputs", nargs="+", required=True)
    edge.add_argument("--output", required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--inputs", nargs="+", required=True)
    aggregate.add_argument("--group-by", required=True, choices=["chrom", "resolution"])
    aggregate.add_argument("--output", required=True)

    args = parser.parse_args()
    summaries = [load_summary(path) for path in args.inputs]
    if args.mode == "edge":
        result = plot_edge(select_best_tolerance_summary(summaries), args.output)
    else:
        result = plot_aggregate(summaries, args.group_by, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
