
import re
import os


def parse_interval(s):
    a, b = map(int, re.findall(r"\d+", s))
    return (a, b)


def boundaries_from_tads(tads):
    starts = {a for (a, b) in tads}
    ends   = {b for (a, b) in tads}
    return starts, ends

def is_laminar_set(intervals, interval_index=None, tolerance=0):
    """
    Return True if intervals are laminar (nested or disjoint).
    Touching at endpoints is allowed (e.g., [1,4] and [4,8]).
    """
    I = list(intervals)
    for i in range(len(I)):
        for j in range(i + 1, len(I)):
            a, b = I[i]
            c, d = I[j]
            # Proper overlap without containment -> not laminar
            if (a + tolerance < c and c + tolerance < b and b + tolerance < d) or (
                c + tolerance < a and a + tolerance < d and d + tolerance < b
            ):
                # if interval_index is None:
                #     print(f"Non-laminar intervals: [{a},{b}] and [{c},{d}]")
                # else:
                #     files_ab = sorted(interval_index.get((a, b), []))
                #     files_cd = sorted(interval_index.get((c, d), []))
                #     files_ad  = sorted(interval_index.get((a, d), []))
                #     files_cb  = sorted(interval_index.get((c, b), []))
                #     files_ad_text = ", ".join(files_ad) if files_ad else "not found"
                #     files_cb_text = ", ".join(files_cb) if files_cb else "not found"
                #     files_ab_text = ", ".join(files_ab) if files_ab else "not found"
                #     files_cd_text = ", ".join(files_cd) if files_cd else "not found"
                #     print(
                #         "Non-laminar intervals: "
                #         f"[{a},{b}] (files: {files_ab_text}) and "
                #         f"[{c},{d}] (files: {files_cd_text}) and "
                #         f"no files contain [{a},{d}] ({files_ad_text}) or "
                #         f"[{c},{b}] ({files_cb_text})"
                #     )
                return False
    return True
def tads_match_with_tolerance(tad1, tad2, tolerance=0):
    """
    Check if two TADs match within a given tolerance.
    tad1, tad2: tuples (start, end)
    tolerance: maximum difference allowed in bins
    Returns True if both start and end are within tolerance.
    """
    a1, b1 = tad1
    a2, b2 = tad2
    return abs(a1 - a2) <= tolerance and abs(b1 - b2) <= tolerance

def match_tads_within_tolerance(inferred_tads, provided_tads, tolerance=0):
    """
    Match inferred TADs to provided TADs with tolerance.
    Returns (matched_pairs, unmatched_inferred, unmatched_provided).
    """
    inferred_list = sorted(inferred_tads)
    provided_list = sorted(provided_tads)
    matched = []
    used_provided = set()
    
    for inf_tad in inferred_list:
        best_match = None
        best_dist = float('inf')
        
        for idx, prov_tad in enumerate(provided_list):
            if idx in used_provided:
                continue
            
            if tads_match_with_tolerance(inf_tad, prov_tad, tolerance):
                dist = abs(inf_tad[0] - prov_tad[0]) + abs(inf_tad[1] - prov_tad[1])
                if dist < best_dist:
                    best_dist = dist
                    best_match = idx
        
        if best_match is not None:
            matched.append((inf_tad, provided_list[best_match]))
            used_provided.add(best_match)
    
    unmatched_inferred = [
        t for t in inferred_list 
        if not any(t == m[0] for m in matched)
    ]
    unmatched_provided = [
        provided_list[i] for i in range(len(provided_list)) 
        if i not in used_provided
    ]
    
    return matched, unmatched_inferred, unmatched_provided

def compare_tad_files(tad_file_a, tad_file_b, tolerance=0, resolution=None, min_size=None, max_size=None):
    """
    Compare two .tad files and return simple overlap counts.

    Returns a dict with:
      - tads_common_from_a_to_b: count of intervals in A that match some interval in B
      - start_common: number of start boundaries matched within tolerance
      - end_common: number of end boundaries matched within tolerance
    """
    intervals_a = read_tad_intervals(tad_file_a, resolution=resolution, min_size=min_size, max_size=max_size)
    intervals_b = read_tad_intervals(tad_file_b, resolution=resolution, min_size=min_size, max_size=max_size)

    tads_a = {parse_interval(s) for s in intervals_a}
    tads_b = {parse_interval(s) for s in intervals_b}

    matched_a = 0
    for a_start, a_end in sorted(tads_a):
        for b_start, b_end in tads_b:
            if abs(a_start - b_start) <= tolerance and abs(a_end - b_end) <= tolerance:
                matched_a += 1
                break

    starts_a, ends_a = boundaries_from_tads(tads_a)
    starts_b, ends_b = boundaries_from_tads(tads_b)

    start_common = _match_points_within_tol(starts_a, starts_b, tolerance)
    end_common = _match_points_within_tol(ends_a, ends_b, tolerance)

    return {
        f"tads_in_{tad_file_a.split('/')[-1]}": len(tads_a),
        f"tads_in_{tad_file_b.split('/')[-1]}": len(tads_b),
        f"tads_common_from_{tad_file_a.split('/')[-1]}_to_{tad_file_b.split('/')[-1]}": matched_a,
        f"start_common": start_common,
        f"end_common": end_common,
        f"start_total_{tad_file_a.split('/')[-1]}": len(starts_a),
        f"start_total_{tad_file_b.split('/')[-1]}": len(starts_b),
        f"end_total_{tad_file_a.split('/')[-1]}": len(ends_a),
        f"end_total_{tad_file_b.split('/')[-1]}": len(ends_b),
        "tolerance": tolerance,
    }

def write_tads_to_file(tads, output_path):
    """
    Write a set of (start, end) interval tuples to a file.
    Each line: start<tab>end
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for a, b in sorted(tads):
            f.write(f"{a}\t{b}\n")

def _match_points_within_tol(a_points, b_points, tol):
    a_sorted = sorted(a_points)
    b_sorted = sorted(b_points)
    i = j = 0
    matched = 0
    while i < len(a_sorted) and j < len(b_sorted):
        if abs(a_sorted[i] - b_sorted[j]) <= tol:
            matched += 1
            i += 1
            j += 1
        elif a_sorted[i] < b_sorted[j] - tol:
            i += 1
        else:
            j += 1
    return matched

def relaxed_boundary_rf(Su, Sv, Eu, Ev, tol=0):
    if tol <= 0:
        return (
            len(Su - Sv) + len(Sv - Su)
            + len(Eu - Ev) + len(Ev - Eu)
        )

    match_s = _match_points_within_tol(Su, Sv, tol)
    match_e = _match_points_within_tol(Eu, Ev, tol)
    return (len(Su) + len(Sv) - 2 * match_s) + (len(Eu) + len(Ev) - 2 * match_e)


def relaxed_tad_rf(Tu, Tv, tol=0):
    if tol <= 0:
        return len(Tu - Tv) + len(Tv - Tu)

    tu_list = sorted(Tu)
    tv_list = list(Tv)
    used = [False] * len(tv_list)
    matched = 0

    for a, b in tu_list:
        best_idx = None
        best_dist = None
        for idx, (c, d) in enumerate(tv_list):
            if used[idx]:
                continue
            if abs(a - c) <= tol and abs(b - d) <= tol:
                dist = abs(a - c) + abs(b - d)
                if best_idx is None or dist < best_dist:
                    best_idx = idx
                    best_dist = dist
        if best_idx is not None:
            used[best_idx] = True
            matched += 1

    return len(tu_list) + len(tv_list) - 2 * matched

def read_tad_intervals(
    tad_file,
    resolution=None,
    min_size=None,
    max_size=None
):
    """
    Read an OnTAD .tad file and return a set of interval strings: {"[start,end]"}.

    Parameters
    ----------
    resolution : int or None
        If provided, convert bin indices to bp using OnTAD convention:
        start_bp = (start_bin-1) * resolution, end_bp = end_bin * resolution.
    min_size, max_size : int or None
        Optional TAD size filtering (applied after any resolution conversion).
    """
    intervals = set()

    with open(tad_file) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            fields = line.rstrip().split()
            if len(fields) < 2:
                continue

            try:
                start_bin = int(fields[0])
                end_bin = int(fields[1])
            except ValueError:
                continue

            if resolution is not None:
                start = (start_bin - 1) * resolution
                end = end_bin * resolution
            else:
                start = start_bin
                end = end_bin

            size = end - start
            if min_size is not None and size < min_size:
                continue
            if max_size is not None and size > max_size:
                continue

            intervals.add(f"[{start},{end}]")

    return intervals
