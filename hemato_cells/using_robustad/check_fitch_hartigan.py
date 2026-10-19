#!/usr/bin/env python3
"""Infer hemato internal-node TADs with the Fitch-Hartigan helper.

This follows the same input handling, filtering, output layout, and summary
generation pattern as `check_intermediate.py`, but replaces the LP-based
internal-node inference with `rf_distance_fitch_hartigan.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tad_utilities import (
    read_tad_intervals,
    parse_interval,
    write_tads_to_file,
    match_tads_within_tolerance,
)
from filtering import harmonize_leaf_boundaries
from rf_distance_fitch_hartigan import infer_internal_tads_fitch_hartigan, infer_tree_rf_distance


def parse_path(path):
    """Parse a hemato path like `MKP/chr1/500kb/tads.tad`."""
    path = path.rstrip("/")
    m = re.search(r"([^/]+)/chr([^/]+)/([^/]+)/tads\.tad$", path)
    if not m:
        raise ValueError(f"Path not in expected format '<NODE>/chr<CHROM>/<RES>/tads.tad': {path}")
    node = m.group(1)
    chrom = m.group(2)
    res = m.group(3)
    return node, chrom, res


def load_tads_from_file(path):
    """Read a .tad file via read_tad_intervals and return set of (a,b) tuples."""
    intervals = read_tad_intervals(path)
    return {parse_interval(s) for s in intervals}


def compute_metrics(pred_set, ref_set, tolerance):
    """Return precision, recall, f1, and the matched interval pairs."""
    matched_pairs, unmatched_pred, unmatched_ref = match_tads_within_tolerance(pred_set, ref_set, tolerance)
    matched = len(matched_pairs)
    prec = matched / len(pred_set) if pred_set else 0.0
    rec = matched / len(ref_set) if ref_set else 0.0
    f1 = 0.0
    if (prec + rec) > 0:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1, matched_pairs


def build_temp_tree_for_selection(leaf1_node, leaf2_node):
    """Keep the same hemato tree layout used by check_intermediate.py."""
    edges = [(0, 2), (0, 1), (2, 3), (2, 4)]
    id_to_name = {0: "MPP", 1: "CLP", 2: "CMP", 3: leaf1_node, 4: leaf2_node}
    return edges, id_to_name


def infer_and_evaluate(leaf1_path, leaf2_path, tolerance=2, outbase="inferred_rf"):
    node1, chrom1, res1 = parse_path(leaf1_path)
    node2, chrom2, res2 = parse_path(leaf2_path)

    if chrom1 != chrom2 or res1 != res2:
        raise ValueError("Chromosome or resolution mismatch between supplied leaf files.")

    chrom = chrom1
    res = res1

    clp_path = os.path.join("CLP", f"chr{chrom}", res, "tads.tad")
    if not os.path.exists(clp_path):
        raise FileNotFoundError(f"Expected CLP file not found: {clp_path}")

    harmonize_list = [leaf1_path, leaf2_path, clp_path]
    print("Harmonizing boundaries for:", harmonize_list, f"tolerance={tolerance}")
    harmonize_leaf_boundaries(harmonize_list, tolerance)

    leaf1_filtered = leaf1_path.replace(".tad", "_filtered.tad")
    leaf2_filtered = leaf2_path.replace(".tad", "_filtered.tad")
    clp_filtered = clp_path.replace(".tad", "_filtered.tad")

    for p in (leaf1_filtered, leaf2_filtered, clp_filtered):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Filtered TAD file expected but missing: {p}")

    edges, id2name = build_temp_tree_for_selection(node1, node2)

    leaf_file_map = {
        1: clp_filtered,
        3: leaf1_filtered,
        4: leaf2_filtered,
    }

    leaf_intervals_for_fh = {}
    for nid, p in leaf_file_map.items():
        intervals = read_tad_intervals(p)
        leaf_intervals_for_fh[nid] = intervals

    print("Calling Fitch-Hartigan inference helper ...")
    node_assignments = infer_internal_tads_fitch_hartigan(edges, leaf_intervals_for_fh)
    rf_summary = infer_tree_rf_distance(edges, node_assignments, tolerance=tolerance)

    script_dir = Path(__file__).resolve().parent
    outbase_path = Path(outbase)
    if not outbase_path.is_absolute():
        outbase_path = script_dir / outbase_path
    inferred_dir_cmp = outbase_path / "CMP" / f"chr{chrom}" / res
    inferred_dir_mpp = outbase_path / "MPP" / f"chr{chrom}" / res
    inferred_dir_cmp.mkdir(parents=True, exist_ok=True)
    inferred_dir_mpp.mkdir(parents=True, exist_ok=True)

    tads_cmp = node_assignments.get(2, set())
    tads_mpp = node_assignments.get(0, set())

    fname_cmp = inferred_dir_cmp / f"inferred_{node1}_{node2}_{tolerance}.tad"
    fname_mpp = inferred_dir_mpp / f"inferred_{node1}_{node2}_{tolerance}.tad"
    write_tads_to_file(tads_cmp, str(fname_cmp))
    write_tads_to_file(tads_mpp, str(fname_mpp))

    print("Saved inferred (CMP) ->", fname_cmp)
    print("Saved inferred (MPP) ->", fname_mpp)

    true_cmp_path = os.path.join("CMP", f"chr{chrom}", res, "tads.tad")
    true_mpp_path = os.path.join("MPP", f"chr{chrom}", res, "tads.tad")

    if not os.path.exists(true_cmp_path):
        print("Warning: true CMP file not found:", true_cmp_path)
    if not os.path.exists(true_mpp_path):
        print("Warning: true MPP file not found:", true_mpp_path)

    true_cmp = load_tads_from_file(true_cmp_path) if os.path.exists(true_cmp_path) else set()
    true_mpp = load_tads_from_file(true_mpp_path) if os.path.exists(true_mpp_path) else set()

    cmp_prec, cmp_rec, cmp_f1, _ = compute_metrics(tads_cmp, true_cmp, tolerance)
    mpp_prec, mpp_rec, mpp_f1, _ = compute_metrics(tads_mpp, true_mpp, tolerance)

    summary = {
        "chrom": chrom,
        "resolution": res,
        "leaf1": node1,
        "leaf2": node2,
        "lp_cost": None,
        "is_integral": True,
        "tolerance": tolerance,
        "rf_score": rf_summary,
        "CMP": {
            "inferred_file": str(fname_cmp),
            "true_file": true_cmp_path,
            "precision": cmp_prec,
            "recall": cmp_rec,
            "f1": cmp_f1,
            "n_inferred": len(tads_cmp),
            "n_true": len(true_cmp),
        },
        "MPP": {
            "inferred_file": str(fname_mpp),
            "true_file": true_mpp_path,
            "precision": mpp_prec,
            "recall": mpp_rec,
            "f1": mpp_f1,
            "n_inferred": len(tads_mpp),
            "n_true": len(true_mpp),
        },
    }

    print("\n=== Summary ===")
    print(f"Chrom: {chrom}  Res: {res}")
    print(f"Leaves used: {node1}, {node2}, CLP")
    print("RF total:", rf_summary["total_rf"])
    print(
        "CMP: P={:.3f} R={:.3f} F1={:.3f} (inferred={}, true={})".format(
            cmp_prec, cmp_rec, cmp_f1, len(tads_cmp), len(true_cmp)
        )
    )
    print(
        "MPP: P={:.3f} R={:.3f} F1={:.3f} (inferred={}, true={})".format(
            mpp_prec, mpp_rec, mpp_f1, len(tads_mpp), len(true_mpp)
        )
    )
    print("=== End ===\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Infer hemato CMP/MPP using Fitch-Hartigan TAD reconstruction.")
    parser.add_argument("--leaf1", help="path to first hemato .tad (e.g. MKP/chr3/100kb/tads.tad)")
    parser.add_argument("--leaf2", help="path to second hemato .tad (e.g. GMP/chr3/100kb/tads.tad)")
    parser.add_argument("--tolerance", type=int, default=2, help="boundary/tad tolerance (bins)")
    parser.add_argument("--outdir", default="inferred_rf", help="output directory base, relative to hemato_cells/")
    args = parser.parse_args()

    summary = infer_and_evaluate(
        args.leaf1,
        args.leaf2,
        tolerance=args.tolerance,
        outbase=args.outdir,
    )

    leaf1_name = args.leaf1.split(os.sep)[0]
    leaf2_name = args.leaf2.split(os.sep)[0]

    outdir_path = Path(args.outdir)
    if not outdir_path.is_absolute():
        outdir_path = Path(__file__).resolve().parent / outdir_path

    out_json = os.path.join(
        str(outdir_path),
        f"jsons/summary_chr{summary['chrom']}_{summary['resolution']}_{leaf1_name}_{leaf2_name}_{args.tolerance}.json",
    )
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("Saved summary to", out_json)


if __name__ == "__main__":
    main()