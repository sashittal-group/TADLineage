#!/usr/bin/env python3
"""Shared weighted OnTAD helpers for scptr_ontad clustering."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MIN_FILES_PER_CATEGORY = 5
TARGET_TADS_PER_CELL = 5


def make_side_by_side_plot(*args, **kwargs):
    from scptr_robustad.em_cluster_scptr_robustad_weighted import (
        make_side_by_side_plot as robustad_make_side_by_side_plot,
    )

    return robustad_make_side_by_side_plot(*args, **kwargs)


def clustering_statistics(true_labels, predicted_labels, cell_distance_matrix):
    from sklearn.metrics import (
        adjusted_rand_score,
        completeness_score,
        homogeneity_score,
        normalized_mutual_info_score,
        silhouette_score,
        v_measure_score,
    )

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
                intervals.append((int(cols[0]), int(cols[1])))
            except Exception:
                continue
    return intervals


def read_original_tad_scores(path):
    """Read original OnTAD tads.tad and return interval -> TADscore."""
    score_by_interval = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            cols = line.split()
            # OnTAD format: startpos endpos TADlevel TADmean TADscore
            if len(cols) < 5:
                continue
            try:
                interval = (int(cols[0]), int(cols[1]))
                score_by_interval[interval] = float(cols[4])
            except Exception:
                continue
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
                    unique_starts.add(int(cols[0]))
                    unique_ends.add(int(cols[1]))
                except Exception:
                    continue

    return len(unique_starts), len(unique_ends)


def effective_top_tads_per_cell(top_tads_per_cell):
    """Use high-score TAD subsets near 20 TADs/cell for OnTAD-heavy inputs."""
    if top_tads_per_cell and top_tads_per_cell > 0:
        return min(int(top_tads_per_cell), TARGET_TADS_PER_CELL)
    return TARGET_TADS_PER_CELL


def top_scored_filtered_intervals(filtered_path, top_tads_per_cell):
    filtered_path = Path(filtered_path)
    original_tad_path = Path(str(filtered_path).replace("_filtered.tad", ".tad"))
    intervals = read_filtered_tad_intervals(filtered_path)
    score_by_interval = read_original_tad_scores(original_tad_path)
    limit = effective_top_tads_per_cell(top_tads_per_cell)

    return sorted(
        intervals,
        key=lambda interval: float(score_by_interval.get(interval, 1.0)),
        reverse=True,
    )[:limit]


def count_top_scored_unique_boundaries(filtered_paths, top_tads_per_cell):
    unique_starts = set()
    unique_ends = set()

    for path in filtered_paths:
        for start, end in top_scored_filtered_intervals(path, top_tads_per_cell):
            unique_starts.add(start)
            unique_ends.add(end)

    return len(unique_starts), len(unique_ends)


def read_boundary_left_right_scores(path):
    """Read optional boundaries.bed and return left/right score maps by position.

    Expected format:
      start end leftScore leftBoundary rightScore rightBoundary

    Native OnTAD output does not include boundary scores. In that case the file
    is absent and boundary weights should have no effect on weighted imputation.
    """
    left_scores = {}
    right_scores = {}

    if not Path(path).exists():
        return left_scores, right_scores

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
                left_scores[pos] = max(left_scores.get(pos, left_score), left_score)
                right_scores[pos] = max(right_scores.get(pos, right_score), right_score)

    return left_scores, right_scores


def weighted_leaf_inputs_from_filtered_paths(filtered_paths, resolution, top_tads_per_cell):
    """Build weighted leaf interval and boundary inputs for weighted DP."""
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

        top_tad_limit = effective_top_tads_per_cell(top_tads_per_cell)
        if top_tad_limit > 0:
            filtered_intervals = sorted(
                filtered_intervals,
                key=lambda interval: float(score_by_interval.get(interval, 1.0)),
                reverse=True,
            )[:top_tad_limit]

        for a, b in filtered_intervals:
            tad_score = float(score_by_interval.get((a, b), 1.0))
            weighted_intervals.append(((a, b), tad_score))

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
    from nussinov_with_imputation_weighted_fast import infer_tads_with_dp as infer_tads_with_imputation_weighted
    from scptr.em_cluster_scptr_tads import imputed_path_for, write_intervals_to_tad_file

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


def records_top_boundary_counts(records, tolerance, top_tads_per_cell):
    from filtering import harmonize_leaf_boundaries
    from scptr.em_cluster_scptr_tads import filtered_path_for

    current_paths = [item["path"] for item in records]
    harmonize_leaf_boundaries(current_paths, tolerance)
    filtered_paths = [str(filtered_path_for(path)) for path in current_paths]
    return count_top_scored_unique_boundaries(filtered_paths, top_tads_per_cell)


def within_boundary_budget(records, tolerance, max_unique_boundaries, top_tads_per_cell):
    if max_unique_boundaries is None or max_unique_boundaries <= 0:
        return True
    start_count, end_count = records_top_boundary_counts(records, tolerance, top_tads_per_cell)
    return start_count <= max_unique_boundaries and end_count <= max_unique_boundaries


def select_records_globally_by_boundary_budget(
    records,
    tolerance,
    max_unique_boundaries,
    top_tads_per_cell=None,
    min_files_per_category=MIN_FILES_PER_CATEGORY,
):
    """Select at least a per-stage baseline, then add files under a top-TAD boundary budget."""
    if not records:
        return []

    ordered_records = sorted(records, key=lambda record: (record["stage"], record["sample"], record["path"]))

    if max_unique_boundaries is None or max_unique_boundaries <= 0:
        return ordered_records

    stages = []
    for record in ordered_records:
        if record["stage"] not in stages:
            stages.append(record["stage"])

    selected_records = []
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

            candidate = remaining[0]
            candidate_records = selected_records + [candidate]
            if within_boundary_budget(candidate_records, tolerance, max_unique_boundaries, top_tads_per_cell):
                selected_records.append(candidate)
                remaining_by_stage[stage] = remaining[1:]
                made_progress = True

    return sorted(selected_records, key=lambda record: (record["stage"], record["sample"], record["path"]))


def select_records_by_stage_boundary_budget(
    records,
    stage_dirs,
    tolerance,
    max_unique_boundaries,
    top_tads_per_cell=None,
    min_files_per_category=MIN_FILES_PER_CATEGORY,
):
    """Keep at least 10 files per stage, then apply a top-scored-TAD boundary budget."""
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
        chosen_for_stage = stage_records[:keep_count]

        for record in stage_records[keep_count:]:
            candidate_for_stage = chosen_for_stage + [record]
            if within_boundary_budget(candidate_for_stage, tolerance, stage_budget, top_tads_per_cell):
                chosen_for_stage.append(record)

        selected_records.extend(chosen_for_stage)

    return sorted(selected_records, key=lambda record: (record["stage"], record["sample"], record["path"]))


if __name__ == "__main__":
    from em_cluster_ontad import main

    main()
