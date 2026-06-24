#!/usr/bin/env python3
"""Visualize EM-style stage medians with fixed chain 1-2-4-8-64."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS

from em_cluster_scptr_tads import (
    STAGE_ORDER,
    STAGE_TO_ID,
    collect_scptr_files,
    compute_joint_distance_matrix,
    compute_median_for_cluster,
    filtered_path_for,
    impute_children_for_stage,
    read_tad_file_as_strings,
    resolve_base_dir,
    sample_records_per_stage,
)
from filtering import harmonize_leaf_boundaries


def fixed_chain_edges(stage_order):
    """Return consecutive stage pairs for fixed developmental chain."""
    return [(stage_order[i], stage_order[i + 1]) for i in range(len(stage_order) - 1)]


def main():
    parser = argparse.ArgumentParser(
        description="Follow em_cluster_scptr_tads.py flow, but connect final medians as 1-2-4-8-64."
    )
    parser.add_argument("--base-dir", default="cell_differentiation", help="Base directory for stage folders")
    parser.add_argument("--chromosome", default="chrX", help="Single chromosome to use for all stages")
    parser.add_argument("--resolution", default="1000kb", help="Resolution folder")
    parser.add_argument("--samples-per-stage", type=int, default=15, help="Files sampled from each stage")
    parser.add_argument("--ct", type=float, default=1.0, help="Interval mismatch weight")
    parser.add_argument("--cb", type=float, default=1.0, help="Boundary mismatch weight")
    parser.add_argument("--tolerance", type=int, default=1, help="Boundary harmonization tolerance")
    parser.add_argument("--random-state", type=int, default=100, help="Random seed")
    parser.add_argument("--output", default="results/scptr_runs/1000kb/aggregate.png", help="Output PNG")
    args = parser.parse_args()

    resolved_base_dir = resolve_base_dir(args.base_dir)
    records = collect_scptr_files(resolved_base_dir, args.chromosome, args.resolution, STAGE_ORDER)
    if not records:
        raise SystemExit(
            "No tads.tad files found for "
            f"chromosome={args.chromosome}, resolution={args.resolution} under {resolved_base_dir}"
        )

    print("Sampling files per stage...", flush=True)
    sampled_records = sample_records_per_stage(records, args.samples_per_stage, random_state=args.random_state)
    sampled_records = sorted(sampled_records, key=lambda record: (record["stage"], record["sample"], record["path"]))

    print(
        "Sampled files per stage: "
        + ", ".join(
            f"{stage}={sum(1 for r in sampled_records if r['stage'] == stage)}"
            for stage in STAGE_ORDER
        ),
        flush=True,
    )

    file_paths = [r["path"] for r in sampled_records]
    print(
        f"Harmonizing leaf boundaries with tolerance={args.tolerance} for {len(file_paths)} files...",
        flush=True,
    )
    harmonize_leaf_boundaries(file_paths, args.tolerance)

    filtered_paths = [str(filtered_path_for(path)) for path in file_paths]
    missing_filtered = [path for path in filtered_paths if not Path(path).exists()]
    if missing_filtered:
        raise FileNotFoundError(
            "Expected filtered TAD files were not created: " + ", ".join(missing_filtered)
        )

    grouped_filtered = {}
    for record, filtered_path in zip(sampled_records, filtered_paths):
        grouped_filtered.setdefault(record["stage"], []).append(filtered_path)

    imputed_paths_by_stage = {}
    for stage in STAGE_ORDER:
        stage_paths = grouped_filtered.get(stage, [])
        if not stage_paths:
            continue
        print(
            f"Running nussinov_with_imputation for stage={stage} on {len(stage_paths)} files...",
            flush=True,
        )
        imputed_paths_by_stage[stage] = impute_children_for_stage(stage_paths, ct=args.ct, cb=args.cb)

    imputed_paths = []
    ordered_true_labels = []
    for stage in STAGE_ORDER:
        for path in imputed_paths_by_stage.get(stage, []):
            imputed_paths.append(path)
            ordered_true_labels.append(stage)

    if not imputed_paths:
        raise SystemExit("No imputed files were produced.")

    cells = [read_tad_file_as_strings(path) for path in imputed_paths]

    # Stage median is computed with the same DP median routine used in em_cluster.
    stage_medians = {}
    for stage in STAGE_ORDER:
        stage_cells = [cell for cell, label in zip(cells, ordered_true_labels) if label == stage]
        if not stage_cells:
            continue
        median_set, _ = compute_median_for_cluster(stage_cells, ct=args.ct, cb=args.cb)
        stage_medians[stage] = set(f"[{a},{b}]" for (a, b) in median_set)

    medoid_order = [stage for stage in STAGE_ORDER if stage in stage_medians]
    medians = [stage_medians[stage] for stage in medoid_order]

    if not medians:
        raise SystemExit("No stage medians were produced.")

    dist = compute_joint_distance_matrix(
        cells,
        medians,
        ct=args.ct,
        cb=args.cb,
        tolerance=args.tolerance,
    )

    positive = dist[dist > 0]
    if len(positive) > 0:
        scale = np.median(positive)
        if scale > 0:
            dist = dist / scale

    print("Applying MDS...", flush=True)
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        metric=True,
        n_init=20,
        max_iter=5000,
        eps=1e-9,
        random_state=args.random_state,
    )
    coords = mds.fit_transform(dist)

    n_cells = len(cells)
    cell_coords = coords[:n_cells]
    median_coords = coords[n_cells:]

    stage_to_indices = {}
    cursor = 0
    for stage in STAGE_ORDER:
        count = sum(1 for label in ordered_true_labels if label == stage)
        if count > 0:
            stage_to_indices[stage] = (cursor, cursor + count)
            cursor += count

    stage_to_coord = {stage: median_coords[idx] for idx, stage in enumerate(medoid_order)}

    print("Creating visualization...", flush=True)
    fig, ax = plt.subplots(figsize=(10, 8))

    cmap_stages = plt.colormaps.get_cmap("tab10") if hasattr(plt, "colormaps") else plt.cm.get_cmap("tab10")
    stage_to_color = {stage: idx for idx, stage in enumerate(STAGE_ORDER)}

    for stage in STAGE_ORDER:
        if stage not in stage_to_indices:
            continue
        start_idx, end_idx = stage_to_indices[stage]
        color_idx = stage_to_color[stage]
        ax.scatter(
            cell_coords[start_idx:end_idx, 0],
            cell_coords[start_idx:end_idx, 1],
            c=[cmap_stages(color_idx)] * (end_idx - start_idx),
            s=44,
            alpha=0.7,
            label=stage,
        )

    for src, dst in fixed_chain_edges(STAGE_ORDER):
        if src in stage_to_coord and dst in stage_to_coord:
            x1, y1 = stage_to_coord[src]
            x2, y2 = stage_to_coord[dst]
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=2.5, alpha=0.85, zorder=1)

    for stage in medoid_order:
        x, y = stage_to_coord[stage]
        color_idx = stage_to_color[stage]
        ax.scatter(
            [x],
            [y],
            c=[cmap_stages(color_idx)],
            s=220,
            marker="*",
            edgecolor="black",
            linewidth=2,
            zorder=10,
        )
        ax.text(
            x,
            y,
            STAGE_TO_ID[stage],
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=11,
        )

    ax.set_xlabel("MDS-1", fontsize=11)
    ax.set_ylabel("MDS-2", fontsize=11)
    ax.set_title(
        f"EM Flow Aggregate ({args.resolution})\n"
        "Fixed chain: 1-2-4-8-64",
        fontsize=12,
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_path}...", flush=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
