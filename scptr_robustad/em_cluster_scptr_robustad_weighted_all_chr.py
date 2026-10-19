#!/usr/bin/env python3
"""Weighted EM clustering for scptr RobustAD TADs merged across chromosomes."""

import argparse
import math
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
from em_cluster_scptr_robustad_weighted import (
    clustering_statistics,
    count_unique_boundaries,
    impute_children_for_stage_weighted,
    make_side_by_side_plot,
    select_records_by_stage_boundary_budget,
    select_records_globally_by_boundary_budget,
)


CHROMS = [f"chr{i}" for i in range(1, 20)] + ["chrX"]


def resolution_to_bp(resolution):
    label = str(resolution).lower()
    if label.endswith("kb"):
        return int(label[:-2]) * 1000
    return int(label)


def resolution_to_data_dir(resolution):
    return str(resolution_to_bp(resolution) // 1000)


def load_chrom_offsets(genome_sizes_path, resolution_bp):
    sizes = {}
    with open(genome_sizes_path) as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 2:
                sizes[parts[0]] = int(parts[1])

    offsets = {}
    cursor = 0
    for chrom in CHROMS:
        if chrom not in sizes:
            raise KeyError(f"{chrom} is missing from {genome_sizes_path}")
        offsets[chrom] = cursor
        cursor += int(math.ceil(sizes[chrom] / resolution_bp)) + 1
    return offsets


def encode_position(chrom, position_bp, offsets, resolution_bp):
    return offsets[chrom] + int(float(position_bp)) // resolution_bp


def read_top_merged_tads(tad_path, offsets, resolution_bp, top_tads_per_cell):
    records = []
    with open(tad_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            cols = line.split()
            if len(cols) < 4 or cols[0] == "chrom":
                continue
            chrom = cols[0]
            if chrom not in offsets:
                continue
            try:
                start = encode_position(chrom, cols[1], offsets, resolution_bp)
                end = encode_position(chrom, cols[2], offsets, resolution_bp)
                score = float(cols[3])
                nested_score = float(cols[4]) if len(cols) > 4 else score
                hq = float(cols[5]) if len(cols) > 5 else 1.0
                level = float(cols[6]) if len(cols) > 6 else 0.0
            except Exception:
                continue
            if end <= start:
                continue
            records.append((start, end, level, score, nested_score, hq))

    records.sort(key=lambda item: item[3], reverse=True)
    if top_tads_per_cell and top_tads_per_cell > 0:
        records = records[:top_tads_per_cell]
    return records


def read_boundary_rows(boundary_path, offsets, resolution_bp, needed_positions):
    rows = []
    with open(boundary_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            cols = line.split()
            if len(cols) < 6:
                continue
            chrom = cols[0]
            if chrom not in offsets:
                continue
            try:
                start = encode_position(chrom, cols[1], offsets, resolution_bp)
                end = encode_position(chrom, cols[2], offsets, resolution_bp)
                left_score = float(cols[3])
                left_boundary = float(cols[4])
                right_score = float(cols[5])
                right_boundary = float(cols[6]) if len(cols) > 6 else 0.0
            except Exception:
                continue
            if start not in needed_positions and end not in needed_positions:
                continue
            rows.append((start, end, left_score, left_boundary, right_score, right_boundary))
    return rows


def write_merged_working_files(raw_sample_dir, out_dir, offsets, resolution_bp, top_tads_per_cell):
    tad_path = raw_sample_dir / "robustad_tads_all.bed"
    boundary_path = raw_sample_dir / "robustad_boundaries.bed"
    if not tad_path.exists() or not boundary_path.exists():
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_tads = read_top_merged_tads(tad_path, offsets, resolution_bp, top_tads_per_cell)
    if not merged_tads:
        return None

    needed_positions = set()
    for start, end, *_ in merged_tads:
        needed_positions.add(start)
        needed_positions.add(end)
    boundary_rows = read_boundary_rows(boundary_path, offsets, resolution_bp, needed_positions)

    out_tads = out_dir / "tads.tad"
    with open(out_tads, "w") as handle:
        for start, end, level, score, nested_score, hq in merged_tads:
            handle.write(f"{start}\t{end}\t{level}\t{score}\t{nested_score}\t{hq}\n")

    out_boundaries = out_dir / "boundaries.bed"
    with open(out_boundaries, "w") as handle:
        for row in boundary_rows:
            handle.write("\t".join(str(value) for value in row) + "\n")

    return out_tads


def prepare_merged_records(raw_base_dir, work_base_dir, resolution, genome_sizes_path, top_tads_per_cell, stage_dirs):
    resolution_bp = resolution_to_bp(resolution)
    data_resolution = resolution_to_data_dir(resolution)
    offsets = load_chrom_offsets(genome_sizes_path, resolution_bp)
    records = []

    for stage in stage_dirs:
        stage_dir = raw_base_dir / data_resolution / stage
        if not stage_dir.exists():
            continue
        for sample_dir in sorted(stage_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            out_dir = work_base_dir / resolution / stage / sample_dir.name / "all" / resolution
            out_tads = write_merged_working_files(
                sample_dir,
                out_dir,
                offsets,
                resolution_bp,
                top_tads_per_cell,
            )
            if out_tads is not None:
                records.append({"path": str(out_tads), "stage": stage, "sample": sample_dir.name})

    return records


def read_imputed_cells(imputed_paths):
    cells = []
    for path in imputed_paths:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-base-dir", default="robustad_scptr_data")
    parser.add_argument("--work-base-dir", default="results/scptr_merged_all_chr")
    parser.add_argument("--genome-sizes", default="../genome.chrom.sizes")
    parser.add_argument("--resolution", required=True, help="Resolution folder, e.g. 500kb")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ct", type=float, default=1.0)
    parser.add_argument("--cb", type=float, default=1.0)
    parser.add_argument("--tolerance", type=int, default=10)
    parser.add_argument("--top-tads-per-cell", type=int, default=20)
    parser.add_argument("--max-unique-boundaries", type=int, default=100)
    parser.add_argument(
        "--per-stage",
        type=int,
        default=0,
        help="Number of files to sample from each stage; <=0 uses per-stage boundary budget",
    )
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--init", choices=["random", "kmeans++", "first"], default="kmeans++")
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.random_state is not None:
        random.seed(args.random_state)

    script_dir = Path(__file__).resolve().parent
    raw_base_dir = Path(args.raw_base_dir)
    if not raw_base_dir.is_absolute():
        raw_base_dir = script_dir / raw_base_dir
    work_base_dir = Path(args.work_base_dir)
    if not work_base_dir.is_absolute():
        work_base_dir = script_dir / work_base_dir
    genome_sizes_path = Path(args.genome_sizes)
    if not genome_sizes_path.is_absolute():
        genome_sizes_path = script_dir / genome_sizes_path

    stage_dirs = STAGE_ORDER
    records = prepare_merged_records(
        raw_base_dir,
        work_base_dir,
        args.resolution,
        genome_sizes_path,
        args.top_tads_per_cell,
        stage_dirs,
    )
    if not records:
        raise SystemExit(f"No merged all-chromosome TAD files found for resolution={args.resolution}")

    print(f"Prepared merged all-chromosome files for {len(records)} cells.", flush=True)
    if args.per_stage and args.per_stage > 0:
        print(f"Selecting up to {args.per_stage} files per stage...", flush=True)
        sampled_records = sample_records_per_stage(
            records,
            args.per_stage,
            random_state=args.random_state,
        )
    else:
        print("Selecting files per stage with a per-stage boundary budget...", flush=True)
        sampled_records = select_records_by_stage_boundary_budget(
            records,
            stage_dirs,
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

    grouped_filtered = {}
    for record, filtered_path in zip(sampled_records, filtered_paths):
        grouped_filtered.setdefault(record["stage"], []).append(filtered_path)

    imputed_paths_by_stage = {}
    for stage in stage_dirs:
        stage_paths = grouped_filtered.get(stage, [])
        if not stage_paths:
            continue
        print(f"Running weighted Nussinov imputation for stage={stage} on {len(stage_paths)} files...", flush=True)
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

    cells = read_imputed_cells(imputed_paths)
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

    output_png = args.output or f"results/scptr_runs_all_cells/{args.resolution}/all/tol_{args.tolerance}.png"
    json_path = args.json_out or f"results/scptr_runs_all_cells/{args.resolution}/all/tol_{args.tolerance}.json"

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
            "chromosome": "all",
            "chromosomes": CHROMS,
            "resolution": args.resolution,
            "tolerance": args.tolerance,
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
