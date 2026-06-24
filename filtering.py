import os
from collections import defaultdict


def compute_median(cluster):
    """
    Returns integer median.
    If even length, uses lower median for determinism.
    """
    cluster = sorted(cluster)
    n = len(cluster)
    mid = n // 2

    if n % 2 == 1:
        return cluster[mid]
    else:
        return cluster[mid - 1]


def cluster_positions_median(positions, tolerance):
    """
    Cluster 1D positions using greedy sweep.
    Each cluster represented by its median.
    """
    if not positions:
        return {}

    positions = sorted(positions)
    mapping = {}

    cluster = [positions[0]]
    cluster_start = positions[0]

    for p in positions[1:]:
        if p - cluster_start <= tolerance:
            cluster.append(p)
        else:
            median = compute_median(cluster)
            for val in cluster:
                mapping[val] = median

            cluster = [p]
            cluster_start = p

    # finalize last cluster
    median = compute_median(cluster)
    for val in cluster:
        mapping[val] = median

    return mapping


def remove_duplicates(intervals):
    """
    Remove exact duplicate intervals and return sorted unique list.
    """
    return sorted(set(intervals))


def intervals_cross(interval1, interval2):
    a, b = interval1
    c, d = interval2
    return (a < c < b < d) or (c < a < d < b)


def find_crossing_pair(intervals):
    """
    Return the indices of one crossing pair, or None if the family is laminar.
    """
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            if intervals_cross(intervals[i], intervals[j]):
                return i, j
    return None


def restore_crossing_pairs_to_original(records):
    """
    Iteratively restore snapped intervals back to their original boundaries
    for any crossing pair until the interval family is laminar.
    """
    while True:
        current_intervals = [record["current"] for record in records]
        crossing_pair = find_crossing_pair(current_intervals)
        if crossing_pair is None:
            return

        i, j = crossing_pair
        records[i]["current"] = records[i]["original"]
        records[j]["current"] = records[j]["original"]


def harmonize_leaf_boundaries(leaf_files, tolerance):
    """
    Pipeline:
    1. Global boundary clustering (median)
    2. Remove duplicates
    3. Remove TADs whose BOTH boundaries are unique to one leaf
    4. Write *_filtered.tad files
    """

    all_starts = []
    all_ends = []
    file_data = {}

    # ---- Read files ----
    for path in leaf_files:
        intervals = []

        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                a = int(parts[0])
                b = int(parts[1])

                intervals.append((a, b))
                all_starts.append(a)
                all_ends.append(b)

        file_data[path] = intervals

    # ---- Snap boundaries globally ----
    start_map = cluster_positions_median(all_starts, tolerance)
    end_map = cluster_positions_median(all_ends, tolerance)

    snapped_records = {}

    for path, intervals in file_data.items():
        records = []

        for a, b in intervals:
            new_a = start_map[a]
            new_b = end_map[b]

            if new_a < new_b:
                records.append({
                    "original": (a, b),
                    "current": (new_a, new_b),
                })

        restore_crossing_pairs_to_original(records)
        snapped_records[path] = records

    snapped_data = {}
    for path, records in snapped_records.items():
        snapped_data[path] = remove_duplicates(record["current"] for record in records)

    # ---- Count boundary occurrences across leaves ----
    start_counts = defaultdict(int)
    end_counts = defaultdict(int)

    # Count per leaf (not per interval occurrence)
    for intervals in snapped_data.values():
        unique_starts = {a for a, _ in intervals}
        unique_ends = {b for _, b in intervals}

        for a in unique_starts:
            start_counts[a] += 1
        for b in unique_ends:
            end_counts[b] += 1

    # ---- Remove TADs whose BOTH boundaries are unique ----
    final_data = {}

    for path, intervals in snapped_data.items():
        filtered = []

        for a, b in intervals:
            if not (start_counts[a] == 1 and end_counts[b] == 1):
                filtered.append((a, b))

        final_data[path] = remove_duplicates(filtered)

    # ---- Write output files ----
    for path, intervals in final_data.items():
        base, _ = os.path.splitext(path)
        new_path = f"{base}_filtered.tad"

        with open(new_path, "w") as f:
            for a, b in intervals:
                f.write(f"{a}\t{b}\n")

        print(
            f"Saved: {new_path} "
            f"({len(intervals)} intervals retained)"
        )
