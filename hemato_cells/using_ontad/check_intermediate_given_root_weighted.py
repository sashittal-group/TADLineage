#!/usr/bin/env python3
"""
infer_pair_and_eval_given_root.py

Usage:
    python hemato_cells/check_intermediate_given_root_weighted.py \
        MKP/chr1/500kb/tads.tad GMP/chr1/500kb/tads.tad \
        --tolerance 5 --outdir inferred_given_root
"""

import os
import re
import argparse
import csv

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
from check_lp_constrained_weighted import solve_lp_from_edges_and_leaves

  
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





def build_robustad_importance_function(
    chrom,
    res,
    tolerance,
    cmp_truth,
    mpp_truth,
    default_nontruth_value=0.1,
    lambda_weight=1.0,
    tad_kind="all",
):
    """
    Build importance(u, a, b) from RobusTAD TAD files.

    Reads:
      - robustad_predictions/MPP/chr<chrom>/<res>/MPP_chr<chrom>_<res>_tads_hq_with_bins.tsv (node id 1)
      - robustad_predictions/CMP/chr<chrom>/<res>/CMP_chr<chrom>_<res>_tads_hq_with_bins.tsv (node id 3)

    For each internal-node interval we look up a raw RobusTAD score (exact match,
    else best tolerance match, else ``default_nontruth_value``). The raw score is
    clamped to [0, 1] and then rescaled by ``lambda_weight`` around the neutral
    point 0.5:

        imp = 0.5 + lambda_weight * (clamp(raw, 0, 1) - 0.5)

    The LP objective's importance term ``x*imp + (1-x)*(1-imp)`` depends on the
    solution only through the per-interval coefficient ``(2*imp - 1) =
    2*lambda_weight*(norm - 0.5)``, so ``lambda_weight`` is a clean strength knob:

        lambda_weight = 0  -> imp == 0.5 everywhere -> term is solution-independent,
                              so the LP reduces to pure parsimony (== given_root).
        lambda_weight = 1  -> RobusTAD score used directly (normalized).
        lambda_weight > 1  -> RobusTAD guidance dominates parsimony more strongly.

    This keeps the shared solver untouched: the strength is folded entirely into
    the returned importance values.
    """

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    chrom_label = f"chr{chrom}"
    mpp_tsv = os.path.join(
        repo_root,
        "robustad_predictions",
        "MPP",
        chrom_label,
        res,
        f"MPP_{chrom_label}_{res}_tads_{tad_kind}_with_bins.tsv",
    )
    cmp_tsv = os.path.join(
        repo_root,
        "robustad_predictions",
        "CMP",
        chrom_label,
        res,
        f"CMP_{chrom_label}_{res}_tads_{tad_kind}_with_bins.tsv",
    )

    def _load_tad_scores(tsv_path):
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"Missing RobustAD TAD file: {tsv_path}")

        exact_scores = {}
        interval_rows = []
        with open(tsv_path, "r", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            required = {"start_bin", "end_bin", "score"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{tsv_path} missing columns: {sorted(missing)}")

            for row in reader:
                a = int(float(row["start_bin"]))
                b = int(float(row["end_bin"]))
                score = float(row["score"])
                key = (a, b)
                if score > exact_scores.get(key, float("-inf")):
                    exact_scores[key] = score
                interval_rows.append((a, b, score))
        return exact_scores, interval_rows

    mpp_exact, mpp_rows = _load_tad_scores(mpp_tsv)
    cmp_exact, cmp_rows = _load_tad_scores(cmp_tsv)

    node_tad_scores = {
        1: {"exact": mpp_exact, "rows": mpp_rows},
        3: {"exact": cmp_exact, "rows": cmp_rows},
    }

    def _raw_score(u, a, b):
        score_map = node_tad_scores.get(u)
        if score_map is None:
            return default_nontruth_value

        exact = score_map["exact"]
        if (a, b) in exact:
            return exact[(a, b)]

        best = float("-inf")
        for ta, tb, s in score_map["rows"]:
            if abs(a - ta) <= tolerance and abs(b - tb) <= tolerance:
                if s > best:
                    best = s + 0.1

        if best == float("-inf"):
            return default_nontruth_value
        return best

    cache = {}

    def importance(u, a, b):
        key = (u, a, b)
        if key in cache:
            return cache[key]

        raw = _raw_score(u, a, b)
        norm = min(1.0, max(0.0, raw))
        val = 0.5 + lambda_weight * (norm - 0.5)

        cache[key] = val
        return val

    return importance


def infer_and_evaluate(
    leaf1_path,
    leaf2_path,
    tolerance=2,
    Ct=1.0,
    Cb=1.0,
    mode="convex",
    importance_source="refhic",
    lambda_weight=1.0,
    tad_kind="all",
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

    true_cmp_path = os.path.join("CMP", f"chr{chrom}", res, "tads.tad")
    true_mpp_path = os.path.join("MPP", f"chr{chrom}", res, "tads.tad")

    if not os.path.exists(true_cmp_path):
        print("Warning: true CMP file not found:", true_cmp_path)
    if not os.path.exists(true_mpp_path):
        print("Warning: true MPP file not found:", true_mpp_path)

    true_cmp = load_tads_from_file(true_cmp_path) if os.path.exists(true_cmp_path) else set()
    true_mpp = load_tads_from_file(true_mpp_path) if os.path.exists(true_mpp_path) else set()

    if importance_source == "robustad":
        importance_fn = build_robustad_importance_function(
            chrom=chrom,
            res=res,
            tolerance=tolerance,
            cmp_truth=true_cmp,
            mpp_truth=true_mpp,
            default_nontruth_value=0.1,
            lambda_weight=lambda_weight,
            tad_kind=tad_kind,
        )
    else:
        raise ValueError(f"Unsupported importance_source: {importance_source}")
    # importance_fn = build_internal_importance_function(
    #     cmp_truth=true_cmp,
    #     mpp_truth=true_mpp,
    #     tolerance=tolerance,
    # )

    print("Calling LP solver (small parsimony, given root intervals, weighted importance) ...")
    lp_cost, is_integral, node_assignments, x_vars, nodes, intervals = solve_lp_from_edges_and_leaves(
        edges,
        leaf_intervals_for_lp,
        Ct=Ct,
        Cb=Cb,
        importance=importance_fn,
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
    parser.add_argument(
        "--importance-source",
        choices=["refhic", "robustad"],
        default="robustad",
        help="boundary importance source",
    )
    parser.add_argument(
        "--importance-weight",
        type=float,
        default=1.0,
        help="strength lambda of the RobusTAD importance term (0 == pure parsimony)",
    )
    parser.add_argument(
        "--tad-kind",
        choices=["all", "hq"],
        default="all",
        help="which RobusTAD prediction set to weight against (tads_all vs tads_hq)",
    )
    parser.add_argument("--outdir", default="inferred_given_root_weighted", help="output directory base")
    args = parser.parse_args()

    summary = infer_and_evaluate(
        args.leaf1,
        args.leaf2,
        tolerance=args.tolerance,
        Ct=args.Ct,
        Cb=args.Cb,
        mode=args.mode,
        importance_source=args.importance_source,
        lambda_weight=args.importance_weight,
        tad_kind=args.tad_kind,
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
