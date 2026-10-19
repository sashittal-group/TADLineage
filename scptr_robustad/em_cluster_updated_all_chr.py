#!/usr/bin/env python3
"""Lambda-controlled partial-label preimputation before all-chromosome EM clustering."""

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


def read_merged_tads(tad_path, offsets, resolution_bp):
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

    return records


def read_boundary_rows(boundary_path, offsets, resolution_bp):
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
            rows.append((start, end, left_score, left_boundary, right_score, right_boundary))
    return rows


def write_merged_working_files(raw_sample_dir, out_dir, offsets, resolution_bp):
    tad_path = raw_sample_dir / "robustad_tads_all.bed"
    boundary_path = raw_sample_dir / "robustad_boundaries.bed"
    if not tad_path.exists() or not boundary_path.exists():
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_tads = read_merged_tads(tad_path, offsets, resolution_bp)
    if not merged_tads:
        return None

    boundary_rows = read_boundary_rows(boundary_path, offsets, resolution_bp)

    out_tads = out_dir / "tads.tad"
    with open(out_tads, "w") as handle:
        for start, end, level, score, nested_score, hq in merged_tads:
            handle.write(f"{start}\t{end}\t{level}\t{score}\t{nested_score}\t{hq}\n")

    out_boundaries = out_dir / "boundaries.bed"
    with open(out_boundaries, "w") as handle:
        for row in boundary_rows:
            handle.write("\t".join(str(value) for value in row) + "\n")

    return out_tads


def prepare_merged_records(raw_base_dir, work_base_dir, resolution, genome_sizes_path, stage_dirs):
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
            )
            if out_tads is not None:
                records.append({"path": str(out_tads), "stage": stage, "sample": sample_dir.name})

    return records


def collect_full_matrix_records(work_base_dir, resolution, stage_dirs):
    records = []
    for stage in stage_dirs:
        stage_dir = work_base_dir / resolution / stage
        if not stage_dir.exists():
            continue
        for sample_dir in sorted(stage_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            tad_path = sample_dir / "all" / resolution / "tads.tad"
            manifest_path = sample_dir / "all" / resolution / "full_matrix_robustad_metadata.tsv"
            if not tad_path.exists() or not manifest_path.exists():
                continue
            manifest_text = manifest_path.read_text()
            if "input_scope\tfull_cell_hic_matrix_all_mm10_chromosomes" not in manifest_text:
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
    """Select files per stage after capping each filtered file to its top TADs."""
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


def select_records_globally_by_boundary_and_tad_budget(
    records,
    tolerance,
    max_unique_boundaries,
    top_tads_per_cell,
):
    """Select the largest global prefix after capping each filtered file to its top TADs."""
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
        apply_top_tad_budget(filtered_paths, top_tads_per_cell)
        start_count, end_count = count_unique_boundaries(filtered_paths)

        if start_count < max_unique_boundaries and end_count < max_unique_boundaries:
            best_records = selected_records[:]

    return best_records if best_records else ordered_records[:1]


def cap_records_per_stage(records, max_per_stage, random_state=None):
    if max_per_stage is None or max_per_stage <= 0:
        return sorted(records, key=lambda record: (record["stage"], record["sample"], record["path"]))

    rng = random.Random(random_state)
    capped_records = []
    for stage in STAGE_ORDER:
        stage_records = sorted(
            [record for record in records if record["stage"] == stage],
            key=lambda record: (record["sample"], record["path"]),
        )
        if len(stage_records) > max_per_stage:
            stage_records = rng.sample(stage_records, max_per_stage)
        capped_records.extend(stage_records)

    return sorted(capped_records, key=lambda record: (record["stage"], record["sample"], record["path"]))


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
            f"stage={stage} on {len(stage_paths)} merged all-chromosome files.",
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
        "preimputation_summary": preimputation_summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-base-dir", default="robustad_scptr_data")
    parser.add_argument("--work-base-dir", default="results_updated/scptr_merged_all_chr")
    parser.add_argument("--genome-sizes", default="../genome.chrom.sizes")
    parser.add_argument("--resolution", required=True, help="Resolution folder, e.g. 500kb")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ct", type=float, default=1.0)
    parser.add_argument("--cb", type=float, default=1.0)
    parser.add_argument("--tolerance", type=int, default=10)
    parser.add_argument(
        "--top-tads-per-cell",
        type=int,
        default=20,
        help="Keep at most this many highest-scoring TADs per selected all-chromosome file",
    )
    parser.add_argument(
        "--max-unique-boundaries",
        type=int,
        default=100,
        help="Unique-boundary budget used when selecting all-chromosome files",
    )
    parser.add_argument(
        "--per-stage",
        type=int,
        default=0,
        help="Number of randomly sampled files from each stage; <=0 uses boundary-budget selection",
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
    records = collect_full_matrix_records(work_base_dir, args.resolution, stage_dirs)
    if records:
        print(
            f"Using {len(records)} precomputed full-matrix all-chromosome RobustAD files "
            f"from {work_base_dir / args.resolution}.",
            flush=True,
        )
    else:
        records = prepare_merged_records(
            raw_base_dir,
            work_base_dir,
            args.resolution,
            genome_sizes_path,
            stage_dirs,
        )
    if not records:
        raise SystemExit(f"No merged all-chromosome TAD files found for resolution={args.resolution}")

    print(f"Prepared merged all-chromosome files for {len(records)} cells.", flush=True)
    print("Selecting merged all-chromosome files per stage with a boundary budget...", flush=True)
    if should_apply_top_tad_budget(args.resolution):
        print(
            "For 250kb, applying the top-TAD cap during boundary-budget selection.",
            flush=True,
        )
        sampled_records = select_records_by_stage_boundary_and_tad_budget(
            records,
            stage_dirs,
            tolerance=args.tolerance,
            max_unique_boundaries=args.max_unique_boundaries,
            top_tads_per_cell=args.top_tads_per_cell,
        )
    else:
        sampled_records = select_records_by_stage_boundary_budget(
            records,
            stage_dirs,
            tolerance=args.tolerance,
            max_unique_boundaries=args.max_unique_boundaries,
        )

    if args.per_stage and args.per_stage > 0:
        before_cap_counts = Counter(record["stage"] for record in sampled_records)
        sampled_records = cap_records_per_stage(
            sampled_records,
            args.per_stage,
            random_state=args.random_state,
        )
        after_cap_counts = Counter(record["stage"] for record in sampled_records)
        print(
            f"Capped selected files to at most {args.per_stage} per stage after boundary/top-TAD selection: "
            + ", ".join(
                f"{stage}={before_cap_counts.get(stage, 0)}->{after_cap_counts.get(stage, 0)}"
                for stage in stage_dirs
            ),
            flush=True,
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

    if should_apply_top_tad_budget(args.resolution):
        print(
            f"Applying top-TAD budget of {args.top_tads_per_cell} per all-chromosome file...",
            flush=True,
        )
        tad_budget_summary = apply_top_tad_budget(filtered_paths, args.top_tads_per_cell)
    else:
        tad_budget_summary = {
            "applied": False,
            "top_tads_per_cell": args.top_tads_per_cell,
            "reason": "top-TAD budget is only applied for 250kb all-chromosome runs",
            "files": [],
        }

    print(
        f"Lambda label fraction={label_fraction:.3f}; "
        "only this fraction uses known stage labels for preimputation.",
        flush=True,
    )
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
            "mode": "lambda_partial_label_preimputation_then_em_all_chr",
            "chromosome": "all",
            "chromosomes": CHROMS,
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
