#!/usr/bin/env python3
"""
Analyze internal-node inferred TADs that are not present in children.

For each (chromosome, resolution, threshold, leaf1, leaf2):
1) CMP: infer novelty in inferred CMP vs (leaf1 union leaf2)
2) MPP: infer novelty in inferred MPP vs (inferred CMP union CLP)
3) For each node, count how many novel inferred TADs appear in the real node set.
"""

import os
import json
import argparse

import sys
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath(".."))

from tad_utilities import read_tad_intervals, parse_interval, match_tads_within_tolerance


def load_tads(path):
    intervals = read_tad_intervals(path)
    return {parse_interval(s) for s in intervals}


def unmatched_with_tolerance(source_set, reference_set, tolerance):
    _, unmatched_source, _ = match_tads_within_tolerance(source_set, reference_set, tolerance)
    return set(unmatched_source)


def count_matches_with_tolerance(source_set, reference_set, tolerance):
    matched_pairs, _, _ = match_tads_within_tolerance(source_set, reference_set, tolerance)
    return len(matched_pairs)


def to_interval_strings(tads):
    return [f"[{a},{b}]" for (a, b) in sorted(tads)]


def resolve_leaf_name(leaf):
    # Some notes use GR while the folder is G.
    if leaf == "GR":
        return "G"
    return leaf


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
    clp_path = os.path.join("CLP", f"chr{chrom}", res, "tads.tad")
    cmp_real_path = os.path.join("CMP", f"chr{chrom}", res, "tads.tad")
    mpp_real_path = os.path.join("MPP", f"chr{chrom}", res, "tads.tad")

    needed_paths = [
        cmp_inferred_path,
        mpp_inferred_path,
        leaf1_path,
        leaf2_path,
        clp_path,
        cmp_real_path,
        mpp_real_path,
    ]
    missing = [p for p in needed_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    cmp_inferred = load_tads(cmp_inferred_path)
    mpp_inferred = load_tads(mpp_inferred_path)
    leaf1_tads = load_tads(leaf1_path)
    leaf2_tads = load_tads(leaf2_path)
    clp_tads = load_tads(clp_path)
    cmp_real = load_tads(cmp_real_path)
    mpp_real = load_tads(mpp_real_path)

    # CMP novelty: inferred CMP not found in (leaf1 union leaf2).
    cmp_children_union = leaf1_tads | leaf2_tads
    cmp_novel = unmatched_with_tolerance(cmp_inferred, cmp_children_union, tolerance)
    cmp_novel_in_real = count_matches_with_tolerance(cmp_novel, cmp_real, tolerance)

    # MPP novelty: inferred MPP not found in (inferred CMP union CLP).
    mpp_children_union = cmp_inferred | clp_tads
    mpp_novel = unmatched_with_tolerance(mpp_inferred, mpp_children_union, tolerance)
    mpp_novel_in_real = count_matches_with_tolerance(mpp_novel, mpp_real, tolerance)

    cmp_novel_fraction_in_real = (cmp_novel_in_real / len(cmp_novel)) if cmp_novel else 0.0
    mpp_novel_fraction_in_real = (mpp_novel_in_real / len(mpp_novel)) if mpp_novel else 0.0

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
            "clp": clp_path,
            "cmp_real": cmp_real_path,
            "mpp_real": mpp_real_path,
        },
        "CMP": {
            "n_inferred": len(cmp_inferred),
            "n_children_union": len(cmp_children_union),
            "n_novel_vs_children": len(cmp_novel),
            "n_novel_present_in_real": cmp_novel_in_real,
            "novel_fraction_present_in_real": cmp_novel_fraction_in_real,
            "novel_tads": to_interval_strings(cmp_novel),
        },
        "MPP": {
            "n_inferred": len(mpp_inferred),
            "n_children_union": len(mpp_children_union),
            "n_novel_vs_children": len(mpp_novel),
            "n_novel_present_in_real": mpp_novel_in_real,
            "novel_fraction_present_in_real": mpp_novel_fraction_in_real,
            "novel_tads": to_interval_strings(mpp_novel),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze internal-only inferred TADs for CMP/MPP")
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

    print(f"Saved internal novelty summary: {args.output}")


if __name__ == "__main__":
    main()
