#!/usr/bin/env python3
"""Lambda-controlled partial-label preimputation before EM clustering."""

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
    collect_scptr_files,
    compute_joint_distance_matrix,
    compute_majority_labels,
    compute_mst_edges_for_medians,
    compute_tree_distance_from_mst,
    em_cluster_tads,
    filtered_path_for,
    resolve_base_dir,
    sample_records_per_stage,
    write_run_json,
)
from scptr_robustad.em_cluster_scptr_robustad_weighted import (
    clustering_statistics,
    count_unique_boundaries,
    impute_children_for_stage_weighted,
    make_side_by_side_plot,
    select_records_by_stage_boundary_budget,
    select_records_globally_by_boundary_budget,
)


def read_tad_set(path):
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
    return intervals


def summarize_filtered_boundaries(filtered_paths):
    start_count, end_count = count_unique_boundaries(filtered_paths)
    return {"unique_starts": start_count, "unique_ends": end_count}


def read_score_by_interval(tad_path):
    score_by_interval = {}
    with open(tad_path) as handle:
        for line in handle:
            cols = line.strip().split()
            if len(cols) < 2:
                continue
            try:
                interval = (int(cols[0]), int(cols[1]))
                score = float(cols[3]) if len(cols) > 3 else 1.0
            except Exception:
                continue
            score_by_interval[interval] = score
    return score_by_interval


def apply_top_tad_budget(filtered_paths, top_tads_per_cell):
    if top_tads_per_cell is None or top_tads_per_cell <= 0:
        return {"applied": False, "top_tads_per_cell": top_tads_per_cell, "files": []}

    summary = []
    for filtered_path in filtered_paths:
        filtered_path = Path(filtered_path)
        original_path = Path(str(filtered_path).replace("_filtered.tad", ".tad"))
        score_by_interval = read_score_by_interval(original_path)
        intervals = []

        with open(filtered_path) as handle:
            for line in handle:
                cols = line.strip().split()
                if len(cols) < 2:
                    continue
                try:
                    interval = (int(cols[0]), int(cols[1]))
                except Exception:
                    continue
                intervals.append(interval)

        retained = sorted(
            intervals,
            key=lambda interval: float(score_by_interval.get(interval, 1.0)),
            reverse=True,
        )[:top_tads_per_cell]

        with open(filtered_path, "w") as handle:
            for start, end in sorted(retained):
                handle.write(f"{start}\t{end}\n")

        summary.append({
            "path": str(filtered_path),
            "before": len(intervals),
            "after": len(retained),
        })

    return {
        "applied": True,
        "top_tads_per_cell": top_tads_per_cell,
        "files": summary,
    }


def should_apply_top_tad_budget(resolution):
    return str(resolution) == "250kb"


def select_records_by_stage_boundary_and_tad_budget(
    records,
    stage_dirs,
    tolerance,
    max_unique_boundaries,
    top_tads_per_cell,
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

        chosen_for_stage = []
        best_for_stage = []

        for record in stage_records:
            chosen_for_stage.append(record)
            current_paths = [item["path"] for item in chosen_for_stage]

            harmonize_leaf_boundaries(current_paths, tolerance)
            filtered_paths = [str(filtered_path_for(path)) for path in current_paths]
            apply_top_tad_budget(filtered_paths, top_tads_per_cell)
            start_count, end_count = count_unique_boundaries(filtered_paths)

            if start_count < stage_budget and end_count < stage_budget:
                best_for_stage = chosen_for_stage[:]

        if best_for_stage:
            selected_records.extend(best_for_stage)
        else:
            selected_records.append(stage_records[0])

    return sorted(selected_records, key=lambda record: (record["stage"], record["sample"], record["path"]))


def choose_known_indices_by_stage(records, label_fraction, random_state=None):
    """Choose the cells whose stage labels are allowed for preimputation."""
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


def build_lambda_preimputed_inputs(
    sampled_records,
    filtered_paths,
    label_fraction,
    resolution,
    top_tads_per_cell,
    ct=1.0,
    cb=1.0,
    random_state=None,
):
    """Preimpute only cells whose labels are known; keep unknown cells filtered."""
    known_indices = choose_known_indices_by_stage(
        sampled_records,
        label_fraction,
        random_state=random_state,
    )
    known_by_stage = {stage: [] for stage in STAGE_ORDER}
    known_index_by_path = {}

    for idx in sorted(known_indices):
        stage = sampled_records[idx]["stage"]
        path = filtered_paths[idx]
        known_by_stage.setdefault(stage, []).append(path)
        known_index_by_path[path] = idx

    input_paths_for_clustering = list(filtered_paths)
    preimputation_summary = []

    for stage in STAGE_ORDER:
        stage_paths = known_by_stage.get(stage, [])
        if not stage_paths:
            preimputation_summary.append({
                "stage": stage,
                "known_cells": 0,
                "method": "none",
            })
            continue

        print(
            "Running nussinov_with_imputation_weighted for known "
            f"stage={stage} on {len(stage_paths)} cells...",
            flush=True,
        )
        imputed_paths = impute_children_for_stage_weighted(
            stage_paths,
            resolution,
            top_tads_per_cell,
            ct=ct,
            cb=cb,
        )
        print(
            "Finished nussinov_with_imputation_weighted for known "
            f"stage={stage}.",
            flush=True,
        )

        for filtered_path, imputed_path in zip(stage_paths, imputed_paths):
            idx = known_index_by_path[filtered_path]
            input_paths_for_clustering[idx] = imputed_path

        preimputation_summary.append({
            "stage": stage,
            "known_cells": len(stage_paths),
            "method": "stage_preimputation",
        })

    cells = [read_tad_set(path) for path in input_paths_for_clustering]
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
        "cells": cells,
        "input_paths_for_clustering": input_paths_for_clustering,
        "known_indices": sorted(known_indices),
        "known_counts_by_stage": dict(known_counts_by_stage),
        "unknown_counts_by_stage": dict(unknown_counts_by_stage),
        "preimputation_summary": preimputation_summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default="scptr_robustad/cell_differentiation_robustad_ontad",
        help="Base directory containing stage/sample/chr/resolution folders",
    )
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--resolution", default="1000kb")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ct", type=float, default=1.0)
    parser.add_argument("--cb", type=float, default=1.0)
    parser.add_argument("--tolerance", type=int, default=1)
    parser.add_argument(
        "--top-tads-per-cell",
        type=int,
        default=20,
        help="Keep at most this many highest-scoring TADs per cell for 1000kb/250kb resolutions",
    )
    parser.add_argument("--max-unique-boundaries", type=int, default=100)
    parser.add_argument("--per-stage", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--init", choices=["random", "kmeans++", "first"], default="kmeans++")
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument(
        "--lambda",
        dest="label_fraction",
        type=float,
        default=1.0,
        help="Fraction of cells with known stage labels for preimputation; 0=no known labels, 1=full stage preimputation",
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

    stage_dirs = STAGE_ORDER
    resolved_base_dir = resolve_base_dir(args.base_dir)
    records = collect_scptr_files(
        resolved_base_dir,
        args.chromosome,
        args.resolution,
        stage_dirs,
    )
    if not records:
        raise SystemExit(
            "No tads.tad files found for "
            f"chromosome={args.chromosome}, resolution={args.resolution} under {resolved_base_dir}"
        )

    if args.per_stage and args.per_stage > 0:
        print(f"Selecting up to {args.per_stage} files per stage...", flush=True)
        sampled_records = sample_records_per_stage(
            records,
            args.per_stage,
            random_state=args.random_state,
        )
        sampled_records = sorted(
            sampled_records,
            key=lambda record: (record["stage"], record["sample"], record["path"]),
        )
    elif should_apply_top_tad_budget(args.resolution):
        print("Selecting files per stage with a top-TAD-capped boundary budget...", flush=True)
        sampled_records = select_records_by_stage_boundary_and_tad_budget(
            records,
            stage_dirs,
            tolerance=args.tolerance,
            max_unique_boundaries=args.max_unique_boundaries,
            top_tads_per_cell=args.top_tads_per_cell,
        )
    elif args.resolution == "500kb":
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

    file_paths = [record["path"] for record in sampled_records]
    true_labels = [record["stage"] for record in sampled_records]
    print(
        "Selected files per stage: "
        + ", ".join(f"{stage}={selected_counts_by_stage.get(stage, 0)}" for stage in stage_dirs),
        flush=True,
    )
    print(
        f"Lambda label fraction={label_fraction:.3f}; "
        "only this fraction uses known stage labels for preimputation.",
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

    if should_apply_top_tad_budget(args.resolution):
        print(
            f"Applying top-TAD budget of {args.top_tads_per_cell} per filtered chromosome file...",
            flush=True,
        )
        tad_budget_summary = apply_top_tad_budget(filtered_paths, args.top_tads_per_cell)
    else:
        tad_budget_summary = {
            "applied": False,
            "top_tads_per_cell": args.top_tads_per_cell,
            "reason": "top-TAD budget is only applied before clustering for 250kb runs",
            "files": [],
        }

    lambda_inputs = build_lambda_preimputed_inputs(
        sampled_records,
        filtered_paths,
        label_fraction,
        args.resolution,
        args.top_tads_per_cell,
        ct=args.ct,
        cb=args.cb,
        random_state=args.random_state,
    )
    cells = lambda_inputs["cells"]
    clustering_input_paths = lambda_inputs["input_paths_for_clustering"]

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

    cluster_majorities, cluster_summary = compute_majority_labels(labels, true_labels)
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
    scores = clustering_statistics(true_labels, labels, cell_dist)

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

    output_png = args.output or (
        f"results_updated/scptr_runs_all_cells/{args.resolution}/{args.chromosome}/"
        f"tol_{args.tolerance}.png"
    )
    json_path = args.json_out or (
        f"results_updated/scptr_runs_all_cells/{args.resolution}/{args.chromosome}/"
        f"tol_{args.tolerance}.json"
    )
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)

    make_side_by_side_plot(
        cell_coords,
        true_labels,
        labels,
        output_png,
        title_suffix=f" ({args.chromosome}, {args.resolution})",
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
            "mode": "lambda_partial_label_preimputation_then_em",
            "chromosome": args.chromosome,
            "resolution": args.resolution,
            "tolerance": args.tolerance,
            "lambda": label_fraction,
            "per_stage": args.per_stage,
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
            "preimputation_summary": lambda_inputs["preimputation_summary"],
            "tad_budget_summary": tad_budget_summary,
            "filtered_boundary_counts": summarize_filtered_boundaries(filtered_paths),
            "cluster_majorities": {str(k): v for k, v in sorted(cluster_majorities.items())},
            "cluster_summary": {str(k): v for k, v in sorted(cluster_summary.items())},
            "spanning_tree_edges": spanning_tree_edges,
            "spanning_tree": spanning_tree,
            "output_png": output_png,
            "input_paths": file_paths,
            "filtered_paths": filtered_paths,
            "clustering_input_paths": clustering_input_paths,
            "true_labels": list(true_labels),
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
    print("Visual-only files per stage:", extra_stage_counts)
    print("True group counts:", dict(Counter(true_labels)))
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
