#!/usr/bin/env python3
import argparse
import random
import re
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)

# Allow imports from repository root when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from filtering import harmonize_leaf_boundaries
from nussinov_with_imputation_weighted_fast import infer_tads_with_dp as infer_tads_with_imputation_weighted
from scptr.em_cluster_scptr_tads import (
    STAGE_ORDER,
    collect_scptr_files,
    compute_joint_distance_matrix,
    compute_majority_labels,
    compute_mst_edges_for_medians,
    compute_tree_distance_from_mst,
    em_cluster_tads,
    filtered_path_for,
    imputed_path_for,
    make_side_by_side_plot,
    resolve_base_dir,
    sample_records_per_stage,
    write_intervals_to_tad_file,
    write_run_json,
)


CLUSTER_COLORS = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#a65628",  # brown
    "#f781bf",  # pink
    "#999999",  # gray
]


def jitter_points_for_visibility(cell_coords, scale=0.012):
    """Add deterministic local jitter so both panels use identical coordinates."""
    coords = np.asarray(cell_coords, dtype=float).copy()

    if coords.size == 0:
        return coords

    span = np.ptp(coords, axis=0)
    radius = max(float(span.max()), 1.0) * scale
    rng = np.random.default_rng(8675309)
    return coords + rng.normal(loc=0.0, scale=radius, size=coords.shape)


def stable_seed_from_label(label):
    return sum((idx + 1) * ord(char) for idx, char in enumerate(str(label))) % (2**32)


def make_visual_only_points(anchors, count, radius, seed_label):
    """Create deterministic random-looking coordinates near selected points."""
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


def make_side_by_side_plot(
    cell_coords,
    true_labels,
    em_labels,
    output_png,
    title_suffix="",
    median_coords=None,
    mst_edges=None,
    median_labels=None,
    extra_stage_counts=None,
    cluster_majorities=None,
):
    unique_true = sorted(set(true_labels))
    cmap_true = plt.cm.get_cmap("tab10", len(unique_true))
    true_to_id = {label: idx for idx, label in enumerate(unique_true)}
    true_ids = [true_to_id[x] for x in true_labels]
    visible_cell_coords = jitter_points_for_visibility(cell_coords)
    span = np.ptp(cell_coords, axis=0) if len(cell_coords) else np.asarray([1.0, 1.0])
    visual_radius = max(float(span.max()), 1.0) * 0.025
    extra_stage_counts = extra_stage_counts or {}
    cluster_majorities = cluster_majorities or {}
    cluster_ids = sorted(set(em_labels))
    cluster_to_color = {
        cluster_id: CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
        for idx, cluster_id in enumerate(cluster_ids)
    }
    extra_coords_by_stage = {}
    for stage, count in extra_stage_counts.items():
        if count <= 0:
            continue
        selected_idx = [idx for idx, label in enumerate(true_labels) if label == stage]
        anchors = visible_cell_coords[selected_idx] if selected_idx else np.mean(visible_cell_coords, axis=0)
        extra_coords_by_stage[stage] = make_visual_only_points(
            anchors,
            count,
            visual_radius,
            f"shared-{stage}",
        )

    plt.figure(figsize=(13, 5))

    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(
        visible_cell_coords[:, 0],
        visible_cell_coords[:, 1],
        c=true_ids,
        cmap="tab10",
        s=30,
        alpha=0.9,
        linewidths=0.25,
        edgecolors="white",
    )
    ax1.set_title(f"True Clustering{title_suffix}")
    ax1.set_xlabel("MDS-1")
    ax1.set_ylabel("MDS-2")

    handles = []
    for label in unique_true:
        idx = true_to_id[label]
        handles.append(ax1.scatter([], [], c=[cmap_true(idx)], label=label, s=36))
    visual_handle_added = False
    for stage, extra_coords in extra_coords_by_stage.items():
        color = cmap_true(true_to_id.get(stage, 0))
        label = "unselected files (visual only)" if not visual_handle_added else None
        ax1.scatter(
            extra_coords[:, 0],
            extra_coords[:, 1],
            c=[color],
            s=20,
            alpha=0.78,
            linewidths=0.15,
            edgecolors="white",
            label=label,
        )
        visual_handle_added = True
    ax1.legend(handles=handles, loc="best", fontsize=8)

    ax2 = plt.subplot(1, 2, 2)
    for cluster_id in cluster_ids:
        selected_idx = [idx for idx, label in enumerate(em_labels) if label == cluster_id]
        if not selected_idx:
            continue
        ax2.scatter(
            visible_cell_coords[selected_idx, 0],
            visible_cell_coords[selected_idx, 1],
            c=[cluster_to_color[cluster_id]],
            s=30,
            alpha=0.9,
            linewidths=0.25,
            edgecolors="white",
            label=f"Cluster {cluster_id}",
        )

    if median_coords is not None and len(median_coords) > 0:
        ax2.scatter(median_coords[:, 0], median_coords[:, 1], c="black", s=120, marker="X", label="Cluster median")
        for midx in range(len(median_coords)):
            label = f"M{midx}"
            if median_labels is not None and midx < len(median_labels):
                label = str(median_labels[midx])
            ax2.text(median_coords[midx, 0], median_coords[midx, 1], label, fontsize=8, color="black")

    for stage, extra_coords in extra_coords_by_stage.items():
        matching_clusters = [cluster_id for cluster_id, majority in cluster_majorities.items() if majority == stage]
        if matching_clusters:
            cluster_id = matching_clusters[0]
            color = cluster_to_color.get(cluster_id, CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)])
        else:
            color = cmap_true(true_to_id.get(stage, 0))
        ax2.scatter(
            extra_coords[:, 0],
            extra_coords[:, 1],
            c=[color],
            s=20,
            alpha=0.78,
            linewidths=0.15,
            edgecolors="white",
        )

    if mst_edges:
        for u, v, _ in mst_edges:
            x1, y1 = median_coords[u, 0], median_coords[u, 1]
            x2, y2 = median_coords[v, 0], median_coords[v, 1]
            ax2.plot([x1, x2], [y1, y2], color="black", linewidth=1.5, alpha=0.85)
    ax2.set_title(f"EM Clustering (Imputed Children){title_suffix}")
    ax2.set_xlabel("MDS-1")
    ax2.set_ylabel("MDS-2")
    ax2.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


def clustering_statistics(true_labels, predicted_labels, cell_distance_matrix):
    scores = {
        "ARI": float(adjusted_rand_score(true_labels, predicted_labels)),
        "NMI": float(normalized_mutual_info_score(true_labels, predicted_labels)),
        "homogeneity": float(homogeneity_score(true_labels, predicted_labels)),
        "completeness": float(completeness_score(true_labels, predicted_labels)),
        "v_measure": float(v_measure_score(true_labels, predicted_labels)),
    }

    unique_clusters = set(predicted_labels)
    if 1 < len(unique_clusters) < len(predicted_labels):
        scores["silhouette"] = float(silhouette_score(cell_distance_matrix, predicted_labels, metric="precomputed"))
    else:
        scores["silhouette"] = float("nan")

    return scores


def read_filtered_tad_intervals(path):
    intervals = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            cols = line.split()
            if len(cols) < 2:
                continue
            try:
                a = int(cols[0])
                b = int(cols[1])
            except Exception:
                continue
            intervals.append((a, b))
    return intervals


def read_original_tad_scores(path):
    """Read original tads.tad and return interval -> score, using score (not nestedScore)."""
    score_by_interval = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            cols = line.split()
            # expected robustad converted format: start end level score nestedScore Hq
            if len(cols) < 4:
                continue
            try:
                a = int(cols[0])
                b = int(cols[1])
                score = float(cols[3])
            except Exception:
                continue
            score_by_interval[(a, b)] = score
    return score_by_interval


def count_unique_boundaries(filtered_paths):
    """Count distinct start and end boundaries across a set of filtered TAD files."""
    unique_starts = set()
    unique_ends = set()

    for path in filtered_paths:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                cols = line.split()
                if len(cols) < 2:
                    continue
                try:
                    a = int(cols[0])
                    b = int(cols[1])
                except Exception:
                    continue
                unique_starts.add(a)
                unique_ends.add(b)

    return len(unique_starts), len(unique_ends)


def read_boundary_left_right_scores(path):
    """Read boundaries.bed and return left/right score maps by position.

    expected robustad converted format:
      start end leftScore leftBoundary rightScore rightBoundary
    """
    left_scores = {}
    right_scores = {}

    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            cols = line.split()
            if len(cols) < 5:
                continue
            try:
                start = int(cols[0])
                end = int(cols[1])
                left_score = float(cols[2])
                right_score = float(cols[4])
            except Exception:
                continue

            for pos in (start, end):
                # Keep max by value so stronger boundary evidence is retained.
                left_scores[pos] = max(left_scores.get(pos, left_score), left_score)
                right_scores[pos] = max(right_scores.get(pos, right_score), right_score)

    return left_scores, right_scores


def weighted_leaf_inputs_from_filtered_paths(filtered_paths, resolution, top_tads_per_cell):
    """Build weighted leaf interval and boundary inputs for weighted DP.

    For each filtered child interval [a,b]:
    - interval weight = score from original tads.tad for [a,b]
    - start/end boundary domains contain both a and b
    - leaf_start_weights use left boundary score at positions
    - leaf_end_weights use right boundary score at positions
    """
    leaf_intervals = {}
    leaf_start_weights = {}
    leaf_end_weights = {}

    starts = set()
    ends = set()

    for child_idx, filtered_path in enumerate(filtered_paths, start=1):
        filtered_path = Path(filtered_path)
        original_tad_path = Path(str(filtered_path).replace("_filtered.tad", ".tad"))
        boundary_path = filtered_path.with_name("boundaries.bed")

        filtered_intervals = read_filtered_tad_intervals(filtered_path)
        score_by_interval = read_original_tad_scores(original_tad_path)
        left_scores, right_scores = read_boundary_left_right_scores(boundary_path)

        weighted_intervals = []
        start_map = {}
        end_map = {}

        # For 250kb and 1000kb, keep only the highest-scoring TADs per file.
        if str(resolution) in {"250kb", "1000kb"} and top_tads_per_cell and top_tads_per_cell > 0:
            filtered_intervals = sorted(
                filtered_intervals,
                key=lambda interval: float(score_by_interval.get(interval, 1.0)),
                reverse=True,
            )[:top_tads_per_cell]

        for a, b in filtered_intervals:
            tad_score = float(score_by_interval.get((a, b), 1.0))
            weighted_intervals.append(((a, b), tad_score))

            # Keep both boundaries in both sets; only score source differs.
            for pos in (a, b):
                start_map[pos] = max(start_map.get(pos, left_scores.get(pos, 0.0)), left_scores.get(pos, 0.0))
                end_map[pos] = max(end_map.get(pos, right_scores.get(pos, 0.0)), right_scores.get(pos, 0.0))
                starts.add(pos)
                ends.add(pos)

        leaf_intervals[child_idx] = weighted_intervals
        leaf_start_weights[child_idx] = start_map
        leaf_end_weights[child_idx] = end_map

    return leaf_intervals, leaf_start_weights, leaf_end_weights, starts, ends


def impute_children_for_stage_weighted(filtered_paths, resolution, top_tads_per_cell, ct=1.0, cb=1.0):
    if not filtered_paths:
        return []

    m = len(filtered_paths)
    edges = [(0, i + 1) for i in range(m)]

    (
        leaf_intervals,
        leaf_start_weights,
        leaf_end_weights,
        starts,
        ends,
    ) = weighted_leaf_inputs_from_filtered_paths(filtered_paths, resolution, top_tads_per_cell)

    _, _, children_map = infer_tads_with_imputation_weighted(
        edges,
        leaf_intervals,
        starts,
        ends,
        leaf_start_weights=leaf_start_weights,
        leaf_end_weights=leaf_end_weights,
        Ct=ct,
        Cb=cb,
        trace=False,
    )

    imputed_paths = []
    for idx, filtered_path in enumerate(filtered_paths, start=1):
        filtered_path = Path(filtered_path)
        out_path = imputed_path_for(filtered_path)
        child_state = children_map.get(idx, {})
        intervals = set(child_state.get("intervals", {}).keys())
        write_intervals_to_tad_file(intervals, out_path)
        imputed_paths.append(str(out_path))

    return imputed_paths


def select_records_globally_by_boundary_budget(records, tolerance, max_unique_boundaries):
    """Select the largest global prefix whose harmonized boundaries stay under budget."""
    if not records:
        return []

    ordered_records = sorted(records, key=lambda record: (record["stage"], record["sample"], record["path"]))

    if max_unique_boundaries is None or max_unique_boundaries <= 0:
        return ordered_records

    selected_records = []
    best_records = []

    for record in ordered_records:
        selected_records.append(record)
        current_paths = [item["path"] for item in selected_records]

        harmonize_leaf_boundaries(current_paths, tolerance)
        filtered_paths = [str(filtered_path_for(path)) for path in current_paths]
        start_count, end_count = count_unique_boundaries(filtered_paths)

        if start_count < max_unique_boundaries and end_count < max_unique_boundaries:
            best_records = selected_records[:]

    return best_records if best_records else ordered_records[:1]


def select_records_by_stage_boundary_budget(records, stage_dirs, tolerance, max_unique_boundaries):
    """Select files stage by stage so each stage keeps a share of the boundary budget."""
    if not records:
        return []

    if max_unique_boundaries is None or max_unique_boundaries <= 0:
        return sorted(records, key=lambda record: (record["stage"], record["sample"], record["path"]))

    stage_budget = max(1, max_unique_boundaries // max(1, len(stage_dirs)))
    selected_records = []

    for stage in stage_dirs:
        stage_records = sorted(
            [record for record in records if record["stage"] == stage],
            key=lambda record: (record["sample"], record["path"]),
        )
        if not stage_records:
            continue

        chosen_for_stage = []
        best_for_stage = []

        for record in stage_records:
            chosen_for_stage.append(record)
            current_paths = [item["path"] for item in chosen_for_stage]

            harmonize_leaf_boundaries(current_paths, tolerance)
            filtered_paths = [str(filtered_path_for(path)) for path in current_paths]
            start_count, end_count = count_unique_boundaries(filtered_paths)

            if start_count < stage_budget and end_count < stage_budget:
                best_for_stage = chosen_for_stage[:]

        if best_for_stage:
            selected_records.extend(best_for_stage)
        else:
            # Keep at least one file so the stage is never empty.
            selected_records.append(stage_records[0])

    return sorted(selected_records, key=lambda record: (record["stage"], record["sample"], record["path"]))


def main():
    parser = argparse.ArgumentParser(
        description="Weighted EM clustering for scptr_robustad TADs with score-aware imputation."
    )
    parser.add_argument(
        "--base-dir",
        default="scptr_robustad/cell_differentiation_robustad_ontad",
        help="Base directory containing stage/sample/chr/resolution folders",
    )
    parser.add_argument("--chromosome", required=True, help="Chromosome name, e.g. chr1")
    parser.add_argument("--resolution", required=True, help="Resolution folder, e.g. 500kb")
    parser.add_argument("--k", type=int, default=5, help="Number of EM clusters")
    parser.add_argument("--ct", type=float, default=1.0, help="Interval mismatch weight")
    parser.add_argument("--cb", type=float, default=1.0, help="Boundary mismatch weight")
    parser.add_argument("--tolerance", type=int, default=10, help="Boundary harmonization tolerance")
    parser.add_argument(
        "--top-tads-per-cell",
        type=int,
        default=20,
        help="Keep at most this many highest-scoring TADs per cell for 1000kb/250kb resolutions",
    )
    parser.add_argument(
        "--max-unique-boundaries",
        type=int,
        default=100,
        help="Total unique-boundary budget split across stages when selecting files",
    )
    parser.add_argument(
        "--per-stage",
        type=int,
        default=0,
        help="Number of randomly sampled files from each stage; <=0 uses all files",
    )
    parser.add_argument("--max-iter", type=int, default=30, help="Maximum EM iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="Objective convergence tolerance")
    parser.add_argument("--init", choices=["random", "kmeans++", "first"], default="kmeans++", help="EM initialization mode")
    parser.add_argument("--random-state", type=int, default=None, help="Optional random seed; default is random sampling")
    parser.add_argument("--json-out", default=None, help="Output JSON path")
    parser.add_argument("--output", default=None, help="Output PNG path")
    args = parser.parse_args()

    if args.random_state is not None:
        random.seed(args.random_state)

    stage_dirs = STAGE_ORDER
    resolved_base_dir = resolve_base_dir(args.base_dir)
    records = collect_scptr_files(resolved_base_dir, args.chromosome, args.resolution, stage_dirs)

    if not records:
        raise SystemExit(
            "No tads.tad files found for "
            f"chromosome={args.chromosome}, resolution={args.resolution} "
            f"under {resolved_base_dir}"
        )

    if args.resolution == "500kb":
        print("Selecting files per stage with a per-stage boundary budget...", flush=True)
        sampled_records = select_records_by_stage_boundary_budget(
            records,
            stage_dirs,
            tolerance=args.tolerance,
            max_unique_boundaries=args.max_unique_boundaries,
        )
    else:
        print("Selecting files with a global boundary budget...", flush=True)
        sampled_records = select_records_globally_by_boundary_budget(
            records,
            tolerance=args.tolerance,
            max_unique_boundaries=args.max_unique_boundaries,
        )

    selected_counts_by_stage = Counter(record["stage"] for record in sampled_records)
    available_counts_by_stage = Counter(record["stage"] for record in records)
    extra_stage_counts = {
        stage: max(0, available_counts_by_stage.get(stage, 0) - selected_counts_by_stage.get(stage, 0))
        for stage in stage_dirs
    }

    file_paths = [r["path"] for r in sampled_records]
    print(
        "Selected files per stage: "
        + ", ".join(
            f"{stage}={sum(1 for r in sampled_records if r['stage'] == stage)}"
            for stage in stage_dirs
        ),
        flush=True,
    )

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
    for stage in stage_dirs:
        stage_paths = grouped_filtered.get(stage, [])
        if not stage_paths:
            continue
        print(
            f"Running nussinov_with_imputation_weighted for stage={stage} on {len(stage_paths)} files...",
            flush=True,
        )
        imputed_paths_by_stage[stage] = impute_children_for_stage_weighted(
            stage_paths,
            args.resolution,
            args.top_tads_per_cell,
            ct=args.ct,
            cb=args.cb,
        )

    imputed_paths = []
    ordered_true_labels = []
    for stage in stage_dirs:
        for path in imputed_paths_by_stage.get(stage, []):
            imputed_paths.append(path)
            ordered_true_labels.append(stage)

    if not imputed_paths:
        raise SystemExit("No imputed files were produced.")

    # Use the same EM objective as scptr clustering, but on weighted-imputed children.
    cells = []
    for path in imputed_paths:
        intervals = set()
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                cols = line.split()
                if len(cols) < 2:
                    continue
                try:
                    a = int(cols[0])
                    b = int(cols[1])
                except Exception:
                    continue
                intervals.add(f"[{a},{b}]")
        cells.append(intervals)

    labels, medians, history = em_cluster_tads(
        cells,
        k=args.k,
        ct=args.ct,
        cb=args.cb,
        max_iter=args.max_iter,
        tol=args.tol,
        init=args.init,
        random_state=args.random_state,
        distance_tolerance=args.tolerance,
    )

    cluster_majorities, cluster_summary = compute_majority_labels(labels, ordered_true_labels)
    mst_edges = compute_mst_edges_for_medians(medians, ct=args.ct, cb=args.cb, tolerance=args.tolerance)

    mst_graph = nx.Graph()
    mst_graph.add_nodes_from(range(len(medians)))
    mst_graph.add_edges_from([(u, v) for u, v, _ in mst_edges])
    tree_distance = compute_tree_distance_from_mst(mst_graph, cluster_majorities)

    dist = compute_joint_distance_matrix(cells, medians, ct=args.ct, cb=args.cb, tolerance=args.tolerance)
    positive = dist[dist > 0]
    if len(positive) > 0:
        scale = np.median(positive)
        if scale > 0:
            dist = dist / scale
    cell_dist = dist[: len(cells), : len(cells)]
    scores = clustering_statistics(ordered_true_labels, labels, cell_dist)

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

    if args.output:
        output_png = args.output
    else:
        output_png = (
            f"scptr_robustad_weighted_imputed_em_{args.chromosome}_{args.resolution}"
            f"_perstage{args.per_stage}_k{args.k}.png"
        )

    make_side_by_side_plot(
        cell_coords,
        ordered_true_labels,
        labels,
        output_png,
        title_suffix=f" ({args.chromosome}, {args.resolution})",
        median_coords=median_coords,
        mst_edges=mst_edges,
        median_labels=[cluster_majorities.get(i, f"M{i}") for i in range(len(medians))],
        extra_stage_counts=extra_stage_counts,
        cluster_majorities=cluster_majorities,
    )

    if args.json_out:
        json_path = args.json_out
    else:
        json_path = f"results/scptr_runs_all_cells/{args.resolution}/{args.chromosome}/tol_{args.tolerance}.json"

    spanning_tree = []
    spanning_tree_edges = []
    stage_to_id = {"1-cell": "1", "2-cell": "2", "4-cell": "4", "8-cell": "8", "64-cell": "64"}
    for u, v, w in mst_edges:
        u_majority = cluster_majorities.get(u, f"M{u}")
        v_majority = cluster_majorities.get(v, f"M{v}")
        u_id = stage_to_id.get(u_majority, str(u_majority))
        v_id = stage_to_id.get(v_majority, str(v_majority))
        edge_label = f"{u_id}-{v_id}"
        spanning_tree_edges.append(edge_label)
        spanning_tree.append({"edge": edge_label, "source": u_id, "target": v_id, "weight": float(w)})

    write_run_json(
        json_path,
        {
            "chromosome": args.chromosome,
            "resolution": args.resolution,
            "tolerance": args.tolerance,
            "per_stage": args.per_stage,
            "k": args.k,
            "random_state": args.random_state,
            "tree_distance": tree_distance,
            "ARI": scores["ARI"],
            "NMI": scores["NMI"],
            "homogeneity": scores["homogeneity"],
            "completeness": scores["completeness"],
            "v_measure": scores["v_measure"],
            "silhouette": scores["silhouette"],
            "available_counts_by_stage": dict(available_counts_by_stage),
            "selected_counts_by_stage": dict(selected_counts_by_stage),
            "visual_only_counts_by_stage": extra_stage_counts,
            "cluster_majorities": {str(k): v for k, v in sorted(cluster_majorities.items())},
            "cluster_summary": {str(k): v for k, v in sorted(cluster_summary.items())},
            "spanning_tree_edges": spanning_tree_edges,
            "spanning_tree": spanning_tree,
            "output_png": output_png,
            "filtered_paths": filtered_paths,
            "imputed_paths": imputed_paths,
            "true_labels": list(ordered_true_labels),
            "predicted_labels": [int(label) for label in labels],
            "cell_coords": cell_coords.tolist(),
            "median_coords": median_coords.tolist(),
        },
    )

    print("Input files:", len(file_paths))
    print("Filtered files:", len(filtered_paths))
    print("Imputed files:", len(imputed_paths))
    print("Available files per stage:", dict(available_counts_by_stage))
    print("Selected files per stage:", dict(selected_counts_by_stage))
    print("Visual-only files per stage:", extra_stage_counts)
    print("True group counts:", dict(Counter(ordered_true_labels)))
    print("EM cluster counts:", dict(Counter(labels)))
    print(
        "Clustering statistics: "
        f"ARI={scores['ARI']:.6f} "
        f"NMI={scores['NMI']:.6f} "
        f"homogeneity={scores['homogeneity']:.6f} "
        f"completeness={scores['completeness']:.6f} "
        f"v_measure={scores['v_measure']:.6f} "
        f"silhouette={scores['silhouette']:.6f}"
    )
    print("Median MST edges:", len(mst_edges))
    print("Tree distance:", tree_distance)
    print("Run JSON:", json_path)
    print("EM iterations:", len(history["obj"]))
    if history["obj"]:
        print("Final objective:", history["obj"][-1])
    print("Saved figure:", output_png)


if __name__ == "__main__":
    main()
