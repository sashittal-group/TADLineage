#!/usr/bin/env python3
"""
infer_pair_and_eval_given_root.py

Usage:
    python hemato_cells/check_intermediate_given_root.py \
        MKP/chr1/500kb/tads.tad GMP/chr1/500kb/tads.tad \
        --tolerance 5 --outdir inferred_given_root
"""

import os
import re
import argparse

import sys
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
from check_lp_constrained import solve_lp_from_edges_and_leaves


def parse_path(path):
    """
    Parse path like 'MKP/chr1/500kb/tads.tad' -> (node, chrom '1', res '500kb')
    Accepts both 'chr1' or '1' (be forgiving).
    Returns (node_name, chrom_str, res_str)
    """
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
    """
    Return (precision, recall, f1, matched_pairs)
      - matched_pairs is list of (pred, ref) tuples (both are (a,b))
    """
    matched_pairs, unmatched_pred, unmatched_ref = match_tads_within_tolerance(pred_set, ref_set, tolerance)
    matched = len(matched_pairs)
    prec = matched / len(pred_set) if pred_set else 0.0
    rec = matched / len(ref_set) if ref_set else 0.0
    f1 = 0.0
    if (prec + rec) > 0:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1, matched_pairs


def build_temp_tree_for_selection(leaf1_node, leaf2_node):
    """
    We map:
        0 -> LT (root, fixed)
        1 -> MPP
        2 -> CLP
        3 -> CMP
        4 -> leaf1_node
        5 -> leaf2_node

    Tree:
        LT-MPP
        MPP-CMP
        MPP-CLP
        CMP-leaf1
        CMP-leaf2
    """
    edges = [(0, 1), (1, 3), (1, 2), (3, 4), (3, 5)]
    id_to_name = {0: "LT", 1: "MPP", 2: "CLP", 3: "CMP", 4: leaf1_node, 5: leaf2_node}
    return edges, id_to_name


def infer_and_evaluate(
    leaf1_path,
    leaf2_path,
    tolerance=2,
    Ct=1.0,
    Cb=1.0,
    mode="convex",
    outbase="inferred_given_root",
):
    node1, chrom1, res1 = parse_path(leaf1_path)
    node2, chrom2, res2 = parse_path(leaf2_path)

    if chrom1 != chrom2 or res1 != res2:
        raise ValueError("Chromosome or resolution mismatch between supplied leaf files.")

    chrom = chrom1
    res = res1

    clp_path = os.path.join("CLP", f"chr{chrom}", res, "tads.tad")
    lt_path = os.path.join("LT", f"chr{chrom}", res, "tads.tad")

    if not os.path.exists(clp_path):
        raise FileNotFoundError(f"Expected CLP file not found: {clp_path}")
    if not os.path.exists(lt_path):
        raise FileNotFoundError(f"Expected LT file not found: {lt_path}")

    harmonize_list = [leaf1_path, leaf2_path, clp_path, lt_path]
    print("Harmonizing boundaries for:", harmonize_list, f"tolerance={tolerance}")
    harmonize_leaf_boundaries(harmonize_list, tolerance)

    leaf1_filtered = leaf1_path.replace(".tad", "_filtered.tad")
    leaf2_filtered = leaf2_path.replace(".tad", "_filtered.tad")
    clp_filtered = clp_path.replace(".tad", "_filtered.tad")
    lt_filtered = lt_path.replace(".tad", "_filtered.tad")

    for p in (leaf1_filtered, leaf2_filtered, clp_filtered, lt_filtered):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Filtered TAD file expected but missing: {p}")

    edges, id2name = build_temp_tree_for_selection(node1, node2)

    leaf_file_map = {
        2: clp_filtered,
        4: leaf1_filtered,
        5: leaf2_filtered,
    }

    leaf_intervals_for_lp = {}
    for nid, p in leaf_file_map.items():
        leaf_intervals_for_lp[nid] = read_tad_intervals(p)

    root_intervals_for_lp = read_tad_intervals(lt_filtered)

    print("Calling LP solver (small parsimony, given root intervals) ...")
    lp_cost, is_integral, node_assignments, x_vars, nodes, intervals = solve_lp_from_edges_and_leaves(
        edges,
        leaf_intervals_for_lp,
        Ct=Ct,
        Cb=Cb,
        mode=mode,
        require_laminar_internal=True,
        use_fitch_hartigan=False,
        verbose=True,
        root_intervals=root_intervals_for_lp,
    )

    if lp_cost is None:
        raise RuntimeError("LP solver failed to find a solution")

    print(f"LP returned cost={lp_cost:.4f} | integral={is_integral}")

    inferred_dir_cmp = os.path.join(outbase, "CMP", f"chr{chrom}", res)
    inferred_dir_mpp = os.path.join(outbase, "MPP", f"chr{chrom}", res)
    os.makedirs(inferred_dir_cmp, exist_ok=True)
    os.makedirs(inferred_dir_mpp, exist_ok=True)

    leaf1_label = node1
    leaf2_label = node2

    fname_cmp = os.path.join(inferred_dir_cmp, f"inferred_{leaf1_label}_{leaf2_label}_{tolerance}.tad")
    fname_mpp = os.path.join(inferred_dir_mpp, f"inferred_{leaf1_label}_{leaf2_label}_{tolerance}.tad")

    tads_cmp = node_assignments.get(3, set())
    tads_mpp = node_assignments.get(1, set())

    write_tads_to_file(tads_cmp, fname_cmp)
    write_tads_to_file(tads_mpp, fname_mpp)

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

    cmp_prec, cmp_rec, cmp_f1, cmp_matched = compute_metrics(tads_cmp, true_cmp, tolerance)
    mpp_prec, mpp_rec, mpp_f1, mpp_matched = compute_metrics(tads_mpp, true_mpp, tolerance)

    summary = {
        "chrom": chrom,
        "resolution": res,
        "leaf1": leaf1_label,
        "leaf2": leaf2_label,
        "given_root": "LT",
        "lp_cost": float(lp_cost),
        "is_integral": bool(is_integral),
        "tolerance": tolerance,
        "CMP": {
            "inferred_file": fname_cmp,
            "true_file": true_cmp_path,
            "precision": cmp_prec,
            "recall": cmp_rec,
            "f1": cmp_f1,
            "n_inferred": len(tads_cmp),
            "n_true": len(true_cmp),
        },
        "MPP": {
            "inferred_file": fname_mpp,
            "true_file": true_mpp_path,
            "precision": mpp_prec,
            "recall": mpp_rec,
            "f1": mpp_f1,
            "n_inferred": len(tads_mpp),
            "n_true": len(true_mpp),
        }
    }

    print("\n=== Summary ===")
    print(f"Chrom: {chrom}  Res: {res}")
    print(f"Leaves used: {leaf1_label}, {leaf2_label}, CLP | given root: LT")
    print("LP cost:", summary["lp_cost"], "integral:", summary["is_integral"])
    print("\nCMP: P={:.3f} R={:.3f} F1={:.3f} (inferred={}, true={})".format(
        cmp_prec, cmp_rec, cmp_f1, summary["CMP"]["n_inferred"], summary["CMP"]["n_true"]
    ))
    print("MPP: P={:.3f} R={:.3f} F1={:.3f} (inferred={}, true={})".format(
        mpp_prec, mpp_rec, mpp_f1, summary["MPP"]["n_inferred"], summary["MPP"]["n_true"]
    ))
    print("=== End ===\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Infer CMP/MPP from two selected leaves, CLP, and given LT root intervals.")
    parser.add_argument("leaf1", help="path to first leaf .tad (e.g. MKP/chr1/500kb/tads.tad)")
    parser.add_argument("leaf2", help="path to second leaf .tad (e.g. GMP/chr1/500kb/tads.tad)")
    parser.add_argument("--tolerance", type=int, default=2, help="boundary/tad tolerance (bins)")
    parser.add_argument("--Ct", type=float, default=1.0)
    parser.add_argument("--Cb", type=float, default=1.0)
    parser.add_argument("--mode", default="lex_boundary_first", help="LP objective mode")
    parser.add_argument("--outdir", default="inferred_given_root", help="output directory base")
    args = parser.parse_args()

    summary = infer_and_evaluate(
        args.leaf1,
        args.leaf2,
        tolerance=args.tolerance,
        Ct=args.Ct,
        Cb=args.Cb,
        mode=args.mode,
        outbase=args.outdir,
    )

    import json
    leaf1_name = args.leaf1.split(os.sep)[0]
    leaf2_name = args.leaf2.split(os.sep)[0]

    out_json = os.path.join(
        args.outdir,
        f"jsons/summary_chr{summary['chrom']}_{summary['resolution']}_{leaf1_name}_{leaf2_name}_{args.tolerance}.json"
    )
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("Saved summary to", out_json)


if __name__ == "__main__":
    main()
