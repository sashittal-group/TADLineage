#!/usr/bin/env python3
"""
Analyze ground-truth TADs that were not inferred for CMP and MPP.

For each (chromosome, resolution, tolerance, leaf1, leaf2):
1) Compare inferred CMP against real CMP and collect unmatched real TADs.
2) Compare inferred MPP against real MPP and collect unmatched real TADs.
3) For each missed real TAD, record its OnTAD level, mean, score, and
   score percentile within the corresponding ground-truth .tad file.
"""

import os
import json
import argparse
from collections import defaultdict

import sys
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tad_utilities import read_tad_intervals, parse_interval, match_tads_within_tolerance


def load_tads(path):
    intervals = read_tad_intervals(path)
    return {parse_interval(s) for s in intervals}


def load_scored_tads(path):
    """
    Read an OnTAD .tad file and return interval -> metadata.
    Expected columns: start end level mean score
    """
    scored_tads = {}

    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue

            fields = line.rstrip().split()
            if len(fields) < 2:
                continue

            try:
                start = int(fields[0])
                end = int(fields[1])
            except ValueError:
                continue

            level = None
            mean = None
            score = None

            if len(fields) >= 3:
                try:
                    level = int(fields[2])
                except ValueError:
                    level = None
            if len(fields) >= 4:
                try:
                    mean = float(fields[3])
                except ValueError:
                    mean = None
            if len(fields) >= 5:
                try:
                    score = float(fields[4])
                except ValueError:
                    score = None

            scored_tads[(start, end)] = {
                "level": level,
                "mean": mean,
                "score": score,
            }

    return scored_tads


def unmatched_reference_with_tolerance(source_set, reference_set, tolerance):
    _, _, unmatched_reference = match_tads_within_tolerance(source_set, reference_set, tolerance)
    return set(unmatched_reference)


def resolve_leaf_name(leaf):
    if leaf == "GR":
        return "G"
    return leaf


def compute_percentile_map(scored_tads):
    """
    Return interval -> score percentile (0-100) within one ground-truth file.
    Higher OnTAD score maps to higher percentile.
    """
    score_groups = defaultdict(list)
    for interval, meta in scored_tads.items():
        score = meta.get("score")
        if score is not None:
            score_groups[score].append(interval)

    if not score_groups:
        return {}

    sorted_scores = sorted(score_groups.keys())
    total = sum(len(intervals) for intervals in score_groups.values())
    percentile_map = {}
    cumulative = 0

    for score in sorted_scores:
        intervals = score_groups[score]
        count = len(intervals)
        avg_rank = cumulative + (count + 1) / 2.0
        percentile = 100.0 if total == 1 else ((avg_rank - 1) / (total - 1)) * 100.0
        for interval in intervals:
            percentile_map[interval] = percentile
        cumulative += count

    return percentile_map


def median(values):
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def summarize_missing_tads(missing_tads, scored_tads, percentile_map):
    details = []
    percentile_values = []

    for interval in sorted(missing_tads):
        meta = scored_tads.get(interval, {})
        percentile = percentile_map.get(interval)
        if percentile is not None:
            percentile_values.append(percentile)

        details.append({
            "interval": f"[{interval[0]},{interval[1]}]",
            "level": meta.get("level"),
            "mean": meta.get("mean"),
            "score": meta.get("score"),
            "score_percentile": percentile,
        })

    summary = {
        "n_missing": len(missing_tads),
        "missing_tads": details,
        "score_percentiles": percentile_values,
    }

    if percentile_values:
        summary.update({
            "mean_score_percentile": sum(percentile_values) / len(percentile_values),
            "median_score_percentile": median(percentile_values),
            "min_score_percentile": min(percentile_values),
            "max_score_percentile": max(percentile_values),
        })
    else:
        summary.update({
            "mean_score_percentile": None,
            "median_score_percentile": None,
            "min_score_percentile": None,
            "max_score_percentile": None,
        })

    return summary


def run_analysis(leaf1, leaf2, chrom, res, tolerance, outdir="inferred"):
    leaf1_dir = resolve_leaf_name(leaf1)
    leaf2_dir = resolve_leaf_name(leaf2)

    cmp_inferred_path = os.path.join(
        outdir, "CMP", f"chr{chrom}", res, f"inferred_{leaf1}_{leaf2}_{tolerance}.tad"
    )
    mpp_inferred_path = os.path.join(
        outdir, "MPP", f"chr{chrom}", res, f"inferred_{leaf1}_{leaf2}_{tolerance}.tad"
    )
    leaf1_path = os.path.join(leaf1_dir, f"chr{chrom}", res, "tads.tad")
    leaf2_path = os.path.join(leaf2_dir, f"chr{chrom}", res, "tads.tad")
    cmp_real_path = os.path.join("CMP", f"chr{chrom}", res, "tads.tad")
    mpp_real_path = os.path.join("MPP", f"chr{chrom}", res, "tads.tad")

    needed_paths = [
        cmp_inferred_path,
        mpp_inferred_path,
        leaf1_path,
        leaf2_path,
        cmp_real_path,
        mpp_real_path,
    ]
    missing = [p for p in needed_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    cmp_inferred = load_tads(cmp_inferred_path)
    mpp_inferred = load_tads(mpp_inferred_path)
    cmp_real = load_tads(cmp_real_path)
    mpp_real = load_tads(mpp_real_path)
    cmp_real_scored = load_scored_tads(cmp_real_path)
    mpp_real_scored = load_scored_tads(mpp_real_path)
    cmp_real_percentiles = compute_percentile_map(cmp_real_scored)
    mpp_real_percentiles = compute_percentile_map(mpp_real_scored)

    cmp_missing = unmatched_reference_with_tolerance(cmp_inferred, cmp_real, tolerance)
    mpp_missing = unmatched_reference_with_tolerance(mpp_inferred, mpp_real, tolerance)

    return {
        "chrom": str(chrom),
        "resolution": res,
        "tolerance": int(tolerance),
        "leaf1": leaf1,
        "leaf2": leaf2,
        "paths": {
            "cmp_inferred": cmp_inferred_path,
            "mpp_inferred": mpp_inferred_path,
            "leaf1": leaf1_path,
            "leaf2": leaf2_path,
            "cmp_real": cmp_real_path,
            "mpp_real": mpp_real_path,
        },
        "CMP": {
            "n_inferred": len(cmp_inferred),
            "n_ground_truth": len(cmp_real),
            "missing_ground_truth": summarize_missing_tads(
                cmp_missing,
                cmp_real_scored,
                cmp_real_percentiles,
            ),
        },
        "MPP": {
            "n_inferred": len(mpp_inferred),
            "n_ground_truth": len(mpp_real),
            "missing_ground_truth": summarize_missing_tads(
                mpp_missing,
                mpp_real_scored,
                mpp_real_percentiles,
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze missed ground-truth TADs for CMP/MPP")
    parser.add_argument("--leaf1", required=True, help="one of MEP/MKP/MK")
    parser.add_argument("--leaf2", required=True, help="one of GMP/G (or GR alias)")
    parser.add_argument("--chrom", required=True, help="chromosome without chr prefix")
    parser.add_argument("--resolution", required=True, help="e.g. 250kb")
    parser.add_argument("--tolerance", required=True, type=int, help="matching tolerance in bins")
    parser.add_argument("--outdir", default="inferred", help="base inferred directory")
    parser.add_argument("--output", required=True, help="output json path")
    args = parser.parse_args()

    summary = run_analysis(
        leaf1=args.leaf1,
        leaf2=args.leaf2,
        chrom=args.chrom,
        res=args.resolution,
        tolerance=args.tolerance,
        outdir=args.outdir,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Saved missing TAD summary: {args.output}")


if __name__ == "__main__":
    main()
