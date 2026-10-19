#!/usr/bin/env python3
"""Lambda-controlled non-imputed Nussinov EM clustering for merged all-chromosome OnTAD calls."""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.manifold import MDS

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from filtering import harmonize_leaf_boundaries
from scptr.em_cluster_scptr_tads import (
    STAGE_ORDER,
    compute_joint_distance_matrix,
    compute_majority_labels,
    compute_mst_edges_for_medians,
    compute_tree_distance_from_mst,
    em_cluster_tads,
    filtered_path_for,
    sample_records_per_stage,
    write_run_json,
)
from em_cluster_ontad_weighted import (
    MIN_FILES_PER_CATEGORY,
    clustering_statistics,
    effective_top_tads_per_cell,
    make_side_by_side_plot,
    read_filtered_tad_intervals,
    read_original_tad_scores,
)


CHROMS = [f"chr{i}" for i in range(1, 20)] + ["chrX"]


def collect_genomewide_records(work_base_dir, resolution, stage_dirs):
    records = []
    for stage in stage_dirs:
        stage_dir = work_base_dir / resolution / stage
        if not stage_dir.exists():
            continue
        for sample_dir in sorted(stage_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            tad_path = sample_dir / "all" / resolution / "tads.tad"
            if not tad_path.exists():
                continue
            records.append({"path": str(tad_path), "stage": stage, "sample": sample_dir.name})
    return records


def read_tad_cells(paths):
    cells = []
    for path in paths:
        intervals = set()
        with open(path) as handle:
            for line in handle:
                cols = line.strip().split()
                if len(cols) < 2:
                    continue
                try:
                    intervals.add(f"[{int(cols[0])},{int(cols[1])}]")
                except Exception:
                    continue
        cells.append(intervals)
    return cells


def read_top_scored_intervals(tad_path, top_tads_per_cell):
    intervals = []
    with open(tad_path) as handle:
        for line in handle:
            cols = line.strip().split()
            if len(cols) < 2:
                continue
            try:
                start = int(cols[0])
                end = int(cols[1])
                score = float(cols[4]) if len(cols) >= 5 else 1.0
            except Exception:
                continue
            intervals.append((start, end, score))

    limit = effective_top_tads_per_cell(top_tads_per_cell)
    if limit > 0:
        intervals = sorted(intervals, key=lambda item: item[2], reverse=True)[:limit]

    return [(start, end) for start, end, _ in intervals]


def raw_top_boundary_counts(records, top_tads_per_cell):
    starts = set()
    ends = set()
    for record in records:
        for start, end in read_top_scored_intervals(record["path"], top_tads_per_cell):
            starts.add(start)
            ends.add(end)
    return len(starts), len(ends)


def select_records_globally_by_top_boundary_budget(
    records,
    max_unique_boundaries,
    top_tads_per_cell,
    min_files_per_category=MIN_FILES_PER_CATEGORY,
):
    if not records:
        return []

    ordered_records = sorted(records, key=lambda record: (record["stage"], record["sample"], record["path"]))
    if max_unique_boundaries is None or max_unique_boundaries <= 0:
        return ordered_records

    selected_records = []
    stages = []
    for record in ordered_records:
        if record["stage"] not in stages:
            stages.append(record["stage"])

    remaining_by_stage = {}
    for stage in stages:
        stage_records = [record for record in ordered_records if record["stage"] == stage]
        keep_count = min(min_files_per_category, len(stage_records))
        selected_records.extend(stage_records[:keep_count])
        remaining_by_stage[stage] = stage_records[keep_count:]

    made_progress = True
    while made_progress:
        made_progress = False
        for stage in stages:
            remaining = remaining_by_stage.get(stage, [])
            if not remaining:
                continue

            record = remaining[0]
            candidate_records = selected_records + [record]
            start_count, end_count = raw_top_boundary_counts(candidate_records, top_tads_per_cell)
            if start_count <= max_unique_boundaries and end_count <= max_unique_boundaries:
                selected_records.append(record)
                remaining_by_stage[stage] = remaining[1:]
                made_progress = True

    return sorted(selected_records, key=lambda record: (record["stage"], record["sample"], record["path"]))


def within_stage_top_boundary_budget(records, max_unique_boundaries, top_tads_per_cell):
    if max_unique_boundaries is None or max_unique_boundaries <= 0:
        return True
    start_count, end_count = raw_top_boundary_counts(records, top_tads_per_cell)
    return start_count <= max_unique_boundaries and end_count <= max_unique_boundaries


def select_records_by_stage_top_boundary_budget(
    records,
    stage_dirs,
    max_unique_boundaries,
    top_tads_per_cell,
    min_files_per_category=MIN_FILES_PER_CATEGORY,
):
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

        keep_count = min(min_files_per_category, len(stage_records))
        selected_for_stage = stage_records[:keep_count]

        for record in stage_records[keep_count:]:
            candidate_for_stage = selected_for_stage + [record]
            if within_stage_top_boundary_budget(candidate_for_stage, stage_budget, top_tads_per_cell):
                selected_for_stage.append(record)

        selected_records.extend(selected_for_stage)

    return sorted(selected_records, key=lambda record: (record["stage"], record["sample"], record["path"]))


def write_top_tads_for_clustering(filtered_path, top_tads_per_cell):
    """Write a top-scoring copy of a filtered OnTAD file for clustering."""
    top_tad_limit = effective_top_tads_per_cell(top_tads_per_cell)
    if top_tad_limit <= 0:
        return str(filtered_path)

    filtered_path = Path(filtered_path)
    original_tad_path = Path(str(filtered_path).replace("_filtered.tad", ".tad"))
    intervals = read_filtered_tad_intervals(filtered_path)
    score_by_interval = read_original_tad_scores(original_tad_path)

    selected = sorted(
        intervals,
        key=lambda interval: float(score_by_interval.get(interval, 1.0)),
        reverse=True,
    )[:top_tad_limit]

    out_path = filtered_path.with_name(
        f"{filtered_path.stem}_top{top_tad_limit}{filtered_path.suffix}"
    )
    with open(out_path, "w") as handle:
        for a, b in selected:
            handle.write(f"{a}\t{b}\n")

    return str(out_path)


def top_tad_paths_for_clustering(filtered_paths, top_tads_per_cell):
    return [
        write_top_tads_for_clustering(path, top_tads_per_cell)
        for path in filtered_paths
    ]


def choose_known_indices_by_stage(records, label_fraction, random_state=None):
    label_fraction = min(1.0, max(0.0, float(label_fraction)))
    rng = random.Random(random_state)
    known_indices = set()

    for stage in STAGE_ORDER:
        stage_indices = [
            idx
            for idx, record in enumerate(records)
            if record["stage"] == stage
        ]
        if not stage_indices:
            continue

        known_count = int(round(label_fraction * len(stage_indices)))
        if label_fraction > 0 and known_count == 0:
            known_count = 1
        known_count = min(known_count, len(stage_indices))
        known_indices.update(rng.sample(stage_indices, known_count))

    return known_indices


def build_lambda_nonimputed_inputs(
    sampled_records,
    filtered_paths,
    label_fraction,
    random_state=None,
):
    known_indices = choose_known_indices_by_stage(
        sampled_records,
        label_fraction,
        random_state=random_state,
    )
    known_by_stage = {stage: [] for stage in STAGE_ORDER}

    for idx in sorted(known_indices):
        stage = sampled_records[idx]["stage"]
        path = filtered_paths[idx]
        known_by_stage.setdefault(stage, []).append(path)

    input_paths_for_clustering = list(filtered_paths)
    preprocessing_summary = []

    for stage in STAGE_ORDER:
        stage_paths = known_by_stage.get(stage, [])
        if not stage_paths:
            preprocessing_summary.append({
                "stage": stage,
                "known_cells": 0,
                "method": "none",
            })
            continue

        print(
            "Using non-imputed filtered TADs for known "
            f"stage={stage} on {len(stage_paths)} merged all-chromosome files.",
            flush=True,
        )

        preprocessing_summary.append({
            "stage": stage,
            "known_cells": len(stage_paths),
            "method": "non_imputed_filtered_tads",
        })

    known_counts_by_stage = Counter(
        sampled_records[idx]["stage"]
        for idx in known_indices
    )
    unknown_counts_by_stage = Counter(
        record["stage"]
        for idx, record in enumerate(sampled_records)
        if idx not in known_indices
    )

    return {
        "cells": read_tad_cells(input_paths_for_clustering),
        "input_paths_for_clustering": input_paths_for_clustering,
        "known_indices": sorted(known_indices),
        "known_counts_by_stage": dict(known_counts_by_stage),
        "unknown_counts_by_stage": dict(unknown_counts_by_stage),
        "preprocessing_summary": preprocessing_summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-base-dir", default="results_updated/scptr_merged_all_chr")
    parser.add_argument("--resolution", required=True, help="Resolution folder, e.g. 500kb")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ct", type=float, default=1.0)
    parser.add_argument("--cb", type=float, default=1.0)
    parser.add_argument("--tolerance", type=int, default=10)
    parser.add_argument(
        "--top-tads-per-cell",
        type=int,
        default=0,
        help="Keep at most this many highest-scoring filtered TADs per selected cell",
    )
    parser.add_argument(
        "--max-unique-boundaries",
        type=int,
        default=0,
        help="Unique-boundary budget for selecting all-chromosome input cells",
    )
    parser.add_argument(
        "--per-stage",
        type=int,
        default=0,
        help="Select up to this many all-chromosome files per stage before clustering",
    )
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--init", choices=["random", "kmeans++", "first"], default="kmeans++")
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument(
        "--lambda",
        dest="label_fraction",
        type=float,
        default=1.0,
        help="Fraction of cells tracked as known labels in the run metadata; clustering uses non-imputed filtered TADs",
    )
    parser.add_argument(
        "--label-fraction",
        dest="label_fraction",
        type=float,
        help="Alias for --lambda",
    )
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    label_fraction = min(1.0, max(0.0, float(args.label_fraction)))
    if args.random_state is not None:
        random.seed(args.random_state)

    script_dir = Path(__file__).resolve().parent
    work_base_dir = Path(args.work_base_dir)
    if not work_base_dir.is_absolute():
        work_base_dir = script_dir / work_base_dir

    stage_dirs = STAGE_ORDER
    records = collect_genomewide_records(work_base_dir, args.resolution, stage_dirs)
    if not records:
        raise SystemExit(
            f"No merged all-chromosome OnTAD files found for resolution={args.resolution} "
            f"under {work_base_dir}. Run merge_ontad_genomewide.py first."
        )

    print(f"Using merged all-chromosome OnTAD files for {len(records)} cells.", flush=True)
    if args.per_stage and args.per_stage > 0:
        print(f"Selecting up to {args.per_stage} all-chromosome files per stage...", flush=True)
        sampled_records = sample_records_per_stage(
            records,
            args.per_stage,
            random_state=args.random_state,
        )
        sampled_records = sorted(
            sampled_records,
            key=lambda record: (record["stage"], record["sample"], record["path"]),
        )
    elif args.resolution == "500kb":
        print(
            "Selecting all-chromosome files per stage with a raw top-TAD boundary budget...",
            flush=True,
        )
        sampled_records = select_records_by_stage_top_boundary_budget(
            records,
            stage_dirs,
            args.max_unique_boundaries,
            args.top_tads_per_cell,
        )
    else:
        print("Selecting all-chromosome files with a raw top-TAD global boundary budget...", flush=True)
        sampled_records = select_records_globally_by_top_boundary_budget(
            records,
            args.max_unique_boundaries,
            args.top_tads_per_cell,
        )

    selected_counts_by_stage = Counter(record["stage"] for record in sampled_records)
    available_counts_by_stage = Counter(record["stage"] for record in records)
    extra_stage_counts = {
        stage: max(0, available_counts_by_stage.get(stage, 0) - selected_counts_by_stage.get(stage, 0))
        for stage in stage_dirs
    }

    file_paths = [record["path"] for record in sampled_records]
    print(
        "Selected files per stage: "
        + ", ".join(f"{stage}={selected_counts_by_stage.get(stage, 0)}" for stage in stage_dirs),
        flush=True,
    )
    print(f"Harmonizing merged leaf boundaries with tolerance={args.tolerance}...", flush=True)
    harmonize_leaf_boundaries(file_paths, args.tolerance)

    filtered_paths = [str(filtered_path_for(path)) for path in file_paths]
    missing_filtered = [path for path in filtered_paths if not Path(path).exists()]
    if missing_filtered:
        raise FileNotFoundError("Expected filtered TAD files were not created: " + ", ".join(missing_filtered))

    clustering_filtered_paths = top_tad_paths_for_clustering(
        filtered_paths,
        args.top_tads_per_cell,
    )
    if args.top_tads_per_cell and args.top_tads_per_cell > 0:
        print(
            f"Using top {args.top_tads_per_cell} filtered TADs per selected all-chromosome cell.",
            flush=True,
        )

    print(
        f"Lambda label fraction={label_fraction:.3f}; "
        "known labels are tracked in metadata; clustering uses non-imputed filtered TADs.",
        flush=True,
    )
    lambda_inputs = build_lambda_nonimputed_inputs(
        sampled_records,
        clustering_filtered_paths,
        label_fraction,
        random_state=args.random_state,
    )
    cells = lambda_inputs["cells"]
    clustering_input_paths = lambda_inputs["input_paths_for_clustering"]
    ordered_true_labels = [record["stage"] for record in sampled_records]
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

    output_png = args.output or f"results_updated/scptr_runs_all_cells/{args.resolution}/all/tol_{args.tolerance}.png"
    json_path = args.json_out or f"results_updated/scptr_runs_all_cells/{args.resolution}/all/tol_{args.tolerance}.json"
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)

    make_side_by_side_plot(
        cell_coords,
        ordered_true_labels,
        labels,
        output_png,
        title_suffix=f" (all chromosomes, {args.resolution})",
        median_coords=median_coords,
        mst_edges=mst_edges,
        median_labels=[cluster_majorities.get(i, f"M{i}") for i in range(len(medians))],
        extra_stage_counts=extra_stage_counts,
        cluster_majorities=cluster_majorities,
    )

    stage_to_id = {"1-cell": "1", "2-cell": "2", "4-cell": "4", "8-cell": "8", "64-cell": "64"}
    spanning_tree = []
    spanning_tree_edges = []
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
            "mode": "ontad_lambda_metadata_nonimputed_nussinov_em_all_chr",
            "chromosome": "all",
            "chromosomes": CHROMS,
            "resolution": args.resolution,
            "tolerance": args.tolerance,
            "lambda": label_fraction,
            "k": args.k,
            "random_state": args.random_state,
            "top_tads_per_cell": args.top_tads_per_cell,
            "max_unique_boundaries": args.max_unique_boundaries,
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
            "known_counts_by_stage": lambda_inputs["known_counts_by_stage"],
            "unknown_counts_by_stage": lambda_inputs["unknown_counts_by_stage"],
            "known_indices": lambda_inputs["known_indices"],
            "preprocessing_summary": lambda_inputs["preprocessing_summary"],
            "cluster_majorities": {str(k): v for k, v in sorted(cluster_majorities.items())},
            "cluster_summary": {str(k): v for k, v in sorted(cluster_summary.items())},
            "spanning_tree_edges": spanning_tree_edges,
            "spanning_tree": spanning_tree,
            "output_png": output_png,
            "filtered_paths": filtered_paths,
            "top_tad_filtered_paths": clustering_filtered_paths,
            "clustering_input_paths": clustering_input_paths,
            "true_labels": list(ordered_true_labels),
            "predicted_labels": [int(label) for label in labels],
            "medians": [sorted(list(median)) for median in medians],
            "cell_coords": cell_coords.tolist(),
            "median_coords": median_coords.tolist(),
            "history": history,
        },
    )

    print("Input files:", len(file_paths))
    print("Filtered files:", len(filtered_paths))
    print("Known label count:", len(lambda_inputs["known_indices"]))
    print("Unknown label count:", len(file_paths) - len(lambda_inputs["known_indices"]))
    print("Available files per stage:", dict(available_counts_by_stage))
    print("Selected files per stage:", dict(selected_counts_by_stage))
    print("Known files per stage:", lambda_inputs["known_counts_by_stage"])
    print("Unknown files per stage:", lambda_inputs["unknown_counts_by_stage"])
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
    print("Tree distance:", tree_distance)
    print("Run JSON:", json_path)
    print("EM iterations:", len(history["obj"]))
    if history["obj"]:
        print("Final objective:", history["obj"][-1])
    print("Saved figure:", output_png)


if __name__ == "__main__":
    main()
