#!/usr/bin/env python3
"""Harmonize hemato-cell TAD leaves and infer an MGR-style tree."""

import argparse
import json
import os
import re
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from filtering import harmonize_leaf_boundaries
from tad_utilities import parse_interval, read_tad_intervals, match_tads_within_tolerance
import solve_mgr


def infer_path_label(path):
    """Extract a stable leaf label from a leaf directory or .tad path."""
    path = Path(path)
    if path.suffix == ".tad" and len(path.parts) >= 4:
        return path.parts[-4]
    return path.name


def resolve_leaf_path(leaf_spec, chrom=None, res=None):
    """Resolve a leaf spec to an actual tads.tad path."""
    leaf_path = Path(leaf_spec)
    if leaf_path.suffix == ".tad":
        return leaf_path

    if chrom is None or res is None:
        raise ValueError("chrom and res are required when leaf inputs are directories or sample names")

    chrom_dir = chrom if str(chrom).startswith("chr") else f"chr{chrom}"
    return leaf_path / chrom_dir / res / "tads.tad"


def load_filtered_tads(path):
    """Load a filtered .tad file as a set of (start, end) tuples."""
    return {parse_interval(s) for s in read_tad_intervals(str(path))}


def build_node_name_map(leaf_labels, edges):
    """Map numeric node ids to readable labels."""
    node_names = {idx + 1: label for idx, label in enumerate(leaf_labels)}
    internal_ids = sorted({node for edge in edges for node in edge if node not in node_names})
    for offset, node_id in enumerate(internal_ids, start=1):
        node_names[node_id] = f"I{offset}"
    return node_names


def reverse_node_name_map(node_names):
    """Reverse mapping from names to node IDs."""
    return {v: k for k, v in node_names.items()}


def compute_metrics(pred_set, ref_set, tolerance=0):
    """Compute precision, recall, and F1 score."""
    matched_pairs, unmatched_pred, unmatched_ref = match_tads_within_tolerance(
        pred_set, ref_set, tolerance
    )
    matched = len(matched_pairs)
    prec = matched / len(pred_set) if pred_set else 0.0
    rec = matched / len(ref_set) if ref_set else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1



def infer_mgr_tree(leaf_specs, chrom=None, res=None, tolerance=2, Ct=1.0, Cb=1.0, outdir="mgr_inferred"):
    """Harmonize leaf boundaries, infer an MGR tree, and persist a summary."""
    raw_paths = [resolve_leaf_path(spec, chrom=chrom, res=res) for spec in leaf_specs]
    for path in raw_paths:
        if not path.exists():
            raise FileNotFoundError(f"Leaf file not found: {path}")

    raw_path_strings = [str(path) for path in raw_paths]
    print("Harmonizing boundaries for:", raw_path_strings, f"tolerance={tolerance}")
    harmonize_leaf_boundaries(raw_path_strings, tolerance)

    filtered_paths = [Path(str(path).replace(".tad", "_filtered.tad")) for path in raw_paths]
    for path in filtered_paths:
        if not path.exists():
            raise FileNotFoundError(f"Filtered TAD file expected but missing: {path}")

    leaf_labels = [infer_path_label(spec) for spec in leaf_specs]
    leaf_tads = {
        idx + 1: load_filtered_tads(path)
        for idx, path in enumerate(filtered_paths)
    }

    edges, nodes = solve_mgr.build_tree_mgr(leaf_tads, Ct=Ct, Cb=Cb)
    node_names = build_node_name_map(leaf_labels, edges)
    name_to_id = reverse_node_name_map(node_names)

    named_edges = solve_mgr.format_tree_edges(edges, node_names=node_names)
    out_root = Path(outdir)
    if chrom is None or res is None:
        chrom_tag = "unknown_chrom"
        res_tag = "unknown_res"
    else:
        chrom_tag = str(chrom)
        res_tag = str(res)

    tree_dir = out_root / chrom_tag / res_tag
    tree_dir.mkdir(parents=True, exist_ok=True)

    # Load true MPP and CMP tads for comparison
    chrom_dir = chrom if str(chrom).startswith("chr") else f"chr{chrom}"
    true_mpp_path = BASE_DIR / "MPP" / chrom_dir / res / "tads.tad"
    true_cmp_path = BASE_DIR / "CMP" / chrom_dir / res / "tads.tad"

    comparison_results = {}
    if "I1" in name_to_id and true_mpp_path.exists():
        i1_id = name_to_id["I1"]
        true_mpp = load_filtered_tads(true_mpp_path)
        inferred_i1 = nodes[i1_id]
        prec, rec, f1 = compute_metrics(inferred_i1, true_mpp, tolerance=tolerance)
        comparison_results["I1_vs_MPP"] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "inferred_count": len(inferred_i1),
            "true_count": len(true_mpp),
        }

    if "I2" in name_to_id and true_cmp_path.exists():
        i2_id = name_to_id["I2"]
        true_cmp = load_filtered_tads(true_cmp_path)
        inferred_i2 = nodes[i2_id]
        prec, rec, f1 = compute_metrics(inferred_i2, true_cmp, tolerance=tolerance)
        comparison_results["I2_vs_CMP"] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "inferred_count": len(inferred_i2),
            "true_count": len(true_cmp),
        }

    summary = {
        "chrom": chrom_tag,
        "resolution": res_tag,
        "tolerance": tolerance,
        "Ct": Ct,
        "Cb": Cb,
        "leafs": leaf_labels,
        "input_files": [str(path) for path in raw_paths],
        "filtered_files": [str(path) for path in filtered_paths],
        "edges": [[u, v] for u, v in named_edges],
        "nodes": {
            node_names.get(node_id, str(node_id)): sorted([[a, b] for a, b in tads])
            for node_id, tads in sorted(nodes.items())
        },
        "comparison": comparison_results,
    }

    summary_path = tree_dir / "mgr_tree_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    tree_text_path = tree_dir / "mgr_tree.txt"
    solve_mgr.write_tree_summary(edges, nodes, str(tree_text_path), node_names=node_names)

    print("\nInferred tree edges:")
    for u, v in named_edges:
        print(f"{u} -- {v}")

    if comparison_results:
        print("\n=== Comparison with True Genomes ===")
        for comparison_label, metrics in comparison_results.items():
            print(f"\n{comparison_label}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1:        {metrics['f1']:.4f}")
            print(f"  Inferred:  {metrics['inferred_count']} TADs")
            print(f"  True:      {metrics['true_count']} TADs")

    print(f"\nSaved summary to {summary_path}")
    print(f"Saved tree text to {tree_text_path}")

    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Infer an MGR-style tree from hemato-cell TAD files.")
    parser.add_argument("leaves", nargs="+", help="Leaf directories or direct .tad file paths")
    parser.add_argument("--chrom", help="Chromosome label such as chr4 or 4")
    parser.add_argument("--res", help="Resolution folder such as 250kb")
    parser.add_argument("--tolerance", type=int, default=2, help="Boundary harmonization tolerance in bins")
    parser.add_argument("--Ct", type=float, default=1.0, help="TAD interval weight")
    parser.add_argument("--Cb", type=float, default=1.0, help="Boundary weight")
    parser.add_argument("--outdir", default="mgr_inferred", help="Output directory root")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    infer_mgr_tree(
        args.leaves,
        chrom=args.chrom,
        res=args.res,
        tolerance=args.tolerance,
        Ct=args.Ct,
        Cb=args.Cb,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
