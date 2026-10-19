#!/usr/bin/env python3
"""Draw the true-stage MDS panel with the cluster medians marked.

Reproduces the left panel of ``scptr_runs_all_cells/<res>/all/tol_<t>.png`` -- same
coordinates, same deterministic visual-only points, same colours -- and marks the
median of each cluster, labelled by the developmental stage of that cluster.

The figure is written as a PNG next to the input JSON unless ``--output`` says otherwise.

Usage:
    python plot_median_chain.py \
        --json results_updated/scptr_runs_all_cells/500kb/all/tol_1.json
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

STAGE_ORDER = ["1-cell", "2-cell", "4-cell", "8-cell", "64-cell"]


def stable_seed_from_label(label):
    """Same seed derivation as em_cluster_scptr_robustad_weighted.py."""
    return sum((idx + 1) * ord(char) for idx, char in enumerate(str(label))) % (2**32)


def jitter_points_for_visibility(cell_coords, scale=0.012):
    coords = np.asarray(cell_coords, dtype=float).copy()
    if coords.size == 0:
        return coords
    span = np.ptp(coords, axis=0)
    radius = max(float(span.max()), 1.0) * scale
    rng = np.random.default_rng(8675309)
    return coords + rng.normal(loc=0.0, scale=radius, size=coords.shape)


def make_visual_only_points(anchors, count, radius, seed_label):
    if count <= 0:
        return np.empty((0, 2))
    anchors = np.asarray(anchors, dtype=float)
    if anchors.ndim == 1:
        anchors = anchors.reshape(1, -1)
    if len(anchors) == 0:
        anchors = np.zeros((1, 2), dtype=float)
    rng = np.random.default_rng(stable_seed_from_label(seed_label))
    base_indices = rng.integers(0, len(anchors), size=count)
    jitter = rng.normal(loc=0.0, scale=radius, size=(count, 2))
    return anchors[base_indices] + jitter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument(
        "--output",
        default=None,
        help="output path without extension (default: <json>_medians.png next to the JSON)",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    summary = json.load(open(json_path))

    cell_coords = np.asarray(summary["cell_coords"], dtype=float)
    median_coords = np.asarray(summary["median_coords"], dtype=float)
    true_labels = summary["true_labels"]
    majorities = {int(k): v for k, v in summary["cluster_majorities"].items()}
    extra_counts = summary.get("visual_only_counts_by_stage", {})

    visible = jitter_points_for_visibility(cell_coords)
    span = np.ptp(cell_coords, axis=0)
    visual_radius = max(float(span.max()), 1.0) * 0.025

    cmap = plt.get_cmap("tab10", len(STAGE_ORDER))
    stage_color = {stage: cmap(i) for i, stage in enumerate(STAGE_ORDER)}

    fig, ax = plt.subplots(figsize=(7.2, 6.0))

    # cells, coloured by their true developmental stage
    for stage in STAGE_ORDER:
        idx = [i for i, lab in enumerate(true_labels) if lab == stage]
        if idx:
            ax.scatter(
                visible[idx, 0], visible[idx, 1],
                c=[stage_color[stage]], s=34, alpha=0.9,
                linewidths=0.25, edgecolors="white", label=stage, zorder=2,
            )
        count = int(extra_counts.get(stage, 0))
        if count > 0:
            anchors = visible[idx] if idx else np.mean(visible, axis=0)
            extra = make_visual_only_points(anchors, count, visual_radius, f"shared-{stage}")
            ax.scatter(
                extra[:, 0], extra[:, 1],
                c=[stage_color[stage]], s=20, alpha=0.78,
                linewidths=0.15, edgecolors="white", zorder=1,
            )

    # the cluster medians, in developmental order
    stage_to_cluster = {stage: cid for cid, stage in majorities.items()}
    chain = [stage_to_cluster[s] for s in STAGE_ORDER if s in stage_to_cluster]
    ax.scatter(
        median_coords[chain, 0], median_coords[chain, 1],
        c="black", s=150, marker="X", zorder=4, label="cluster median",
    )
    for stage in STAGE_ORDER:
        if stage not in stage_to_cluster:
            continue
        x, y = median_coords[stage_to_cluster[stage]]
        ax.annotate(
            stage, (x, y), textcoords="offset points", xytext=(10, 8),
            fontsize=10, fontweight="bold", zorder=5,
        )

    ax.set_title("Clustering by TAD hierarchies", fontsize=14, pad=12)
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    # outside the axes on the right: keeps the 1-cell cluster visible and keeps the
    # legend off the top-left, where a reader looks first
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.35), fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.15, linestyle="--")

    if args.output:
        out = Path(args.output)
        if out.suffix.lower() in (".png", ".pdf"):
            out = out.with_suffix("")
    else:
        out = json_path.with_name(f"{json_path.stem}_medians")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png")
    print(f"chain order: {[majorities[c] for c in chain]}")


if __name__ == "__main__":
    main()
