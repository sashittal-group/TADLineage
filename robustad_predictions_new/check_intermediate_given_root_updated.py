#!/usr/bin/env python3
"""
Infer CMP/MPP with LT fixed as root, using RobusTAD calls as inputs/truth.

Examples
--------
python check_intermediate_given_root_updated.py \
    robustad_tads/GMP/chr1/500kb/tads_all_with_bins.tsv \
    robustad_tads/MKP/chr1/500kb/tads_all_with_bins.tsv \
    --tolerance 2 --outdir inferred_given_root_robustad

The positional leaf paths identify sample/chromosome/resolution and, when
possible, whether to use RobusTAD all or hq TADs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from tad_utilities import match_tads_within_tolerance, write_tads_to_file
from filtering import harmonize_leaf_boundaries
from check_lp_constrained_weighted_updated import solve_lp_from_edges_and_leaves


ROBUSTAD_ROOT = SCRIPT_DIR / "robustad_tads"


def parse_path(path: str) -> tuple[str, str, str]:
    """
    Parse paths like:
      robustad_predictions_new/robustad_tads/GMP/chr1/500kb/tads_all.bed
      robustad_predictions_new/robustad_tads/GMP/chr1/500kb/tads_all_with_bins.tsv

    Returns (sample, chrom_without_chr, resolution_label).
    """
    path = path.rstrip("/")
    m = re.search(r"([^/]+)/chr([^/]+)/([^/]+)/(?:tads\.tad|tads_(?:all|hq)(?:_with_bins)?\.(?:bed|tsv))$", path)
    if not m:
        raise ValueError(
            "Path not in expected format '<SAMPLE>/chr<CHROM>/<RES>/tads.tad' "
            f"or RobusTAD tads path: {path}"
        )
    return m.group(1), m.group(2), m.group(3)


def parse_tad_kind(path: str) -> str | None:
    name = Path(path).name
    if name.startswith("tads_all"):
        return "all"
    if name.startswith("tads_hq"):
        return "hq"
    return None


def robustad_dir(sample: str, chrom: str, res: str) -> Path:
    return ROBUSTAD_ROOT / sample / f"chr{chrom}" / res


def robustad_tads_path(sample: str, chrom: str, res: str, tad_kind: str) -> Path:
    name = f"tads_{tad_kind}_with_bins.tsv"
    return robustad_dir(sample, chrom, res) / name


def robustad_boundary_path(sample: str, chrom: str, res: str) -> Path:
    return robustad_dir(sample, chrom, res) / "boundary_with_bins.tsv"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            "Run RobusTAD inference and then robustad_predictions_new/convert_robustad_bins.py."
        )


def load_robustad_tads(sample: str, chrom: str, res: str, tad_kind: str) -> set[tuple[int, int]]:
    path = robustad_tads_path(sample, chrom, res, tad_kind)
    require_file(path)

    tads: set[tuple[int, int]] = set()
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"start_bin", "end_bin"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")

        for row in reader:
            if not row.get("start_bin") or not row.get("end_bin"):
                continue
            a = int(float(row["start_bin"]))
            b = int(float(row["end_bin"]))
            if a < b:
                tads.add((a, b))
    return tads


def interval_strings(tads: set[tuple[int, int]]) -> set[str]:
    return {f"[{a},{b}]" for a, b in tads}


def read_simple_tad_file(path: Path) -> set[tuple[int, int]]:
    tads: set[tuple[int, int]] = set()
    with path.open() as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                a = int(parts[0])
                b = int(parts[1])
            except ValueError:
                continue
            if a < b:
                tads.add((a, b))
    return tads


def materialize_robustad_tads_for_harmonization(
    sample: str,
    chrom: str,
    res: str,
    tad_kind: str,
    tads: set[tuple[int, int]],
    outbase: str,
) -> Path:
    out = Path(outbase) / "_harmonize_inputs" / sample / f"chr{chrom}" / res / f"tads_{tad_kind}.tad"
    write_tads_to_file(tads, str(out))
    return out


def load_boundary_scores(sample: str, chrom: str, res: str) -> tuple[dict[int, float], dict[int, float]]:
    path = robustad_boundary_path(sample, chrom, res)
    require_file(path)

    left_scores: dict[int, float] = {}
    right_scores: dict[int, float] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"bin", "leftScore", "rightScore"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")

        for row in reader:
            if not row.get("bin"):
                continue
            b = int(float(row["bin"]))
            try:
                left_scores[b] = float(row["leftScore"])
            except (TypeError, ValueError):
                pass
            try:
                right_scores[b] = float(row["rightScore"])
            except (TypeError, ValueError):
                pass
    return left_scores, right_scores


def load_tad_scores(sample: str, chrom: str, res: str, tad_kind: str) -> dict[tuple[int, int], float]:
    path = robustad_tads_path(sample, chrom, res, tad_kind)
    require_file(path)

    scores: dict[tuple[int, int], float] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"start_bin", "end_bin", "score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")

        for row in reader:
            if not row.get("start_bin") or not row.get("end_bin"):
                continue
            a = int(float(row["start_bin"]))
            b = int(float(row["end_bin"]))
            if a >= b:
                continue
            try:
                score = float(row["score"])
            except (TypeError, ValueError):
                continue
            key = (a, b)
            if score > scores.get(key, float("-inf")):
                scores[key] = score
    return scores


def build_leaf_importance(
    node_to_sample: dict[int, str],
    leaf_intervals: dict[int, set[str]],
    chrom: str,
    res: str,
    tad_kind: str,
    match_tolerance: int,
    default_tad_score: float | None = None,
    default_boundary_score: float | None = None,
) -> dict[int, dict[tuple[int, int], dict[str, float]]]:
    """
    Build the leaf_importance object expected by check_lp_constrained_weighted_updated.

    For interval [a,b]:
      x = RobusTAD TAD score from tads_*_with_bins.tsv
      s = leftScore at boundary bin a from boundary_with_bins.tsv
      e = rightScore at boundary bin b from boundary_with_bins.tsv
    """
    leaf_importance: dict[int, dict[tuple[int, int], dict[str, float]]] = {}

    for node, sample in node_to_sample.items():
        tad_scores = load_tad_scores(sample, chrom, res, tad_kind)
        tad_rows = list(tad_scores.items())
        left_scores, right_scores = load_boundary_scores(sample, chrom, res)

        node_scores: dict[tuple[int, int], dict[str, float]] = {}
        for interval_str in leaf_intervals[node]:
            a, b = map(int, re.findall(r"\d+", interval_str))
            entry: dict[str, float] = {}

            x_score = tad_scores.get((a, b), None)
            if x_score is None:
                best = float("-inf")
                for (ta, tb), score in tad_rows:
                    if abs(a - ta) <= match_tolerance and abs(b - tb) <= match_tolerance:
                        best = max(best, score)
                x_score = default_tad_score if best == float("-inf") else best
            s_score = left_scores.get(a, default_boundary_score)
            e_score = right_scores.get(b, default_boundary_score)

            if x_score is not None:
                entry["x"] = float(x_score)
            if s_score is not None:
                entry["s"] = float(s_score)
            if e_score is not None:
                entry["e"] = float(e_score)

            if entry:
                node_scores[(a, b)] = entry

        leaf_importance[node] = node_scores

    return leaf_importance


def compute_metrics(
    pred_set: set[tuple[int, int]],
    ref_set: set[tuple[int, int]],
    tolerance: int,
) -> tuple[float, float, float, list[tuple[tuple[int, int], tuple[int, int]]]]:
    matched_pairs, _, _ = match_tads_within_tolerance(pred_set, ref_set, tolerance)
    matched = len(matched_pairs)
    precision = matched / len(pred_set) if pred_set else 0.0
    recall = matched / len(ref_set) if ref_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, matched_pairs


def build_temp_tree_for_selection(leaf1_node: str, leaf2_node: str):
    edges = [(0, 1), (1, 3), (1, 2), (3, 4), (3, 5)]
    id_to_name = {0: "LT", 1: "MPP", 2: "CLP", 3: "CMP", 4: leaf1_node, 5: leaf2_node}
    return edges, id_to_name


def infer_and_evaluate(
    leaf1_path: str,
    leaf2_path: str,
    tolerance: int = 2,
    Ct: float = 1.0,
    Cb: float = 1.0,
    mode: str = "lex_boundary_first",
    tad_kind: str | None = None,
    outbase: str = "inferred_given_root_robustad",
    leaf_importance_weight: float = 1.0,
    default_tad_score: float | None = None,
    default_boundary_score: float | None = None,
    harmonize: bool = True,
):
    node1, chrom1, res1 = parse_path(leaf1_path)
    node2, chrom2, res2 = parse_path(leaf2_path)

    if chrom1 != chrom2 or res1 != res2:
        raise ValueError("Chromosome or resolution mismatch between supplied leaf paths.")

    leaf1_kind = parse_tad_kind(leaf1_path)
    leaf2_kind = parse_tad_kind(leaf2_path)
    if tad_kind is None:
        if leaf1_kind and leaf2_kind and leaf1_kind != leaf2_kind:
            raise ValueError(f"Input TAD kinds differ: {leaf1_kind} vs {leaf2_kind}")
        tad_kind = leaf1_kind or leaf2_kind or "all"

    if tad_kind not in {"all", "hq"}:
        raise ValueError("--tad-kind must be 'all' or 'hq'")

    chrom = chrom1
    res = res1

    edges, id_to_name = build_temp_tree_for_selection(node1, node2)

    print(f"Loading RobusTAD {tad_kind} TADs for chr{chrom} {res}")
    clp_tads = load_robustad_tads("CLP", chrom, res, tad_kind)
    leaf1_tads = load_robustad_tads(node1, chrom, res, tad_kind)
    leaf2_tads = load_robustad_tads(node2, chrom, res, tad_kind)
    lt_tads = load_robustad_tads("LT", chrom, res, tad_kind)
    true_cmp = load_robustad_tads("CMP", chrom, res, tad_kind)
    true_mpp = load_robustad_tads("MPP", chrom, res, tad_kind)

    if harmonize:
        harmonize_paths = {
            "CLP": materialize_robustad_tads_for_harmonization("CLP", chrom, res, tad_kind, clp_tads, outbase),
            node1: materialize_robustad_tads_for_harmonization(node1, chrom, res, tad_kind, leaf1_tads, outbase),
            node2: materialize_robustad_tads_for_harmonization(node2, chrom, res, tad_kind, leaf2_tads, outbase),
            "LT": materialize_robustad_tads_for_harmonization("LT", chrom, res, tad_kind, lt_tads, outbase),
        }
        ordered_paths = [str(harmonize_paths[node1]), str(harmonize_paths[node2]), str(harmonize_paths["CLP"]), str(harmonize_paths["LT"])]
        print("Harmonizing RobusTAD boundaries for:", ordered_paths, f"tolerance={tolerance}")
        harmonize_leaf_boundaries(ordered_paths, tolerance)

        def filtered_path(path: Path) -> Path:
            return path.with_name(path.stem + "_filtered.tad")

        leaf1_tads = read_simple_tad_file(filtered_path(harmonize_paths[node1]))
        leaf2_tads = read_simple_tad_file(filtered_path(harmonize_paths[node2]))
        clp_tads = read_simple_tad_file(filtered_path(harmonize_paths["CLP"]))
        lt_tads = read_simple_tad_file(filtered_path(harmonize_paths["LT"]))

    leaf_intervals_for_lp = {
        2: interval_strings(clp_tads),
        4: interval_strings(leaf1_tads),
        5: interval_strings(leaf2_tads),
    }
    root_intervals_for_lp = interval_strings(lt_tads)

    leaf_importance = build_leaf_importance(
        node_to_sample={2: "CLP", 4: node1, 5: node2},
        leaf_intervals=leaf_intervals_for_lp,
        chrom=chrom,
        res=res,
        tad_kind=tad_kind,
        match_tolerance=tolerance,
        default_tad_score=default_tad_score,
        default_boundary_score=default_boundary_score,
    )

    print("Calling updated LP solver with RobusTAD leaf importance ...")
    lp_cost, is_integral, node_assignments, x_vars, nodes, intervals = solve_lp_from_edges_and_leaves(
        edges,
        leaf_intervals_for_lp,
        Ct=Ct,
        Cb=Cb,
        leaf_importance=leaf_importance,
        mode=mode,
        require_laminar_internal=True,
        use_fitch_hartigan=False,
        verbose=True,
        root_intervals=root_intervals_for_lp,
        ground_truth_intervals={1: true_mpp, 3: true_cmp},
        leaf_importance_weight=leaf_importance_weight,
    )

    if lp_cost is None:
        raise RuntimeError("LP solver failed to find a solution")

    outbase_path = Path(outbase)
    inferred_dir_cmp = outbase_path / "CMP" / f"chr{chrom}" / res
    inferred_dir_mpp = outbase_path / "MPP" / f"chr{chrom}" / res
    inferred_dir_cmp.mkdir(parents=True, exist_ok=True)
    inferred_dir_mpp.mkdir(parents=True, exist_ok=True)

    tads_cmp = node_assignments.get(3, set())
    tads_mpp = node_assignments.get(1, set())

    fname_cmp = inferred_dir_cmp / f"inferred_{node1}_{node2}_{tolerance}_{tad_kind}.tad"
    fname_mpp = inferred_dir_mpp / f"inferred_{node1}_{node2}_{tolerance}_{tad_kind}.tad"
    write_tads_to_file(tads_cmp, str(fname_cmp))
    write_tads_to_file(tads_mpp, str(fname_mpp))

    cmp_prec, cmp_rec, cmp_f1, _ = compute_metrics(tads_cmp, true_cmp, tolerance)
    mpp_prec, mpp_rec, mpp_f1, _ = compute_metrics(tads_mpp, true_mpp, tolerance)

    summary = {
        "chrom": chrom,
        "resolution": res,
        "tad_kind": tad_kind,
        "leaf1": node1,
        "leaf2": node2,
        "given_root": "LT",
        "lp_cost": float(lp_cost),
        "is_integral": bool(is_integral),
        "tolerance": tolerance,
        "leaf_importance_weight": leaf_importance_weight,
        "harmonized_inputs": harmonize,
        "inputs": {
            "LT": str(robustad_tads_path("LT", chrom, res, tad_kind)),
            "CLP": str(robustad_tads_path("CLP", chrom, res, tad_kind)),
            node1: str(robustad_tads_path(node1, chrom, res, tad_kind)),
            node2: str(robustad_tads_path(node2, chrom, res, tad_kind)),
            "CMP_truth": str(robustad_tads_path("CMP", chrom, res, tad_kind)),
            "MPP_truth": str(robustad_tads_path("MPP", chrom, res, tad_kind)),
        },
        "counts": {
            "LT_root": len(lt_tads),
            "CLP_leaf": len(clp_tads),
            f"{node1}_leaf": len(leaf1_tads),
            f"{node2}_leaf": len(leaf2_tads),
            "candidate_intervals": len(intervals),
        },
        "CMP": {
            "inferred_file": str(fname_cmp),
            "true_file": str(robustad_tads_path("CMP", chrom, res, tad_kind)),
            "precision": cmp_prec,
            "recall": cmp_rec,
            "f1": cmp_f1,
            "n_inferred": len(tads_cmp),
            "n_true": len(true_cmp),
        },
        "MPP": {
            "inferred_file": str(fname_mpp),
            "true_file": str(robustad_tads_path("MPP", chrom, res, tad_kind)),
            "precision": mpp_prec,
            "recall": mpp_rec,
            "f1": mpp_f1,
            "n_inferred": len(tads_mpp),
            "n_true": len(true_mpp),
        },
    }

    print("\n=== Summary ===")
    print(f"Chrom: {chrom}  Res: {res}  RobusTAD TADs: {tad_kind}")
    print(f"Leaves used: {node1}, {node2}, CLP | given root: LT")
    print("LP cost:", summary["lp_cost"], "integral:", summary["is_integral"])
    print(
        "CMP: P={:.3f} R={:.3f} F1={:.3f} (inferred={}, robustad={})".format(
            cmp_prec, cmp_rec, cmp_f1, len(tads_cmp), len(true_cmp)
        )
    )
    print(
        "MPP: P={:.3f} R={:.3f} F1={:.3f} (inferred={}, robustad={})".format(
            mpp_prec, mpp_rec, mpp_f1, len(tads_mpp), len(true_mpp)
        )
    )
    print("=== End ===\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer CMP/MPP from selected leaves and fixed LT root using RobusTAD TADs and importance scores."
    )
    parser.add_argument(
        "leaf1",
        help="First RobusTAD leaf TAD file, e.g. robustad_tads/MKP/chr1/500kb/tads_all_with_bins.tsv",
    )
    parser.add_argument(
        "leaf2",
        help="Second RobusTAD leaf TAD file, e.g. robustad_tads/GMP/chr1/500kb/tads_all_with_bins.tsv",
    )
    parser.add_argument("--tolerance", type=int, default=2, help="Boundary/TAD tolerance in bins")
    parser.add_argument("--Ct", type=float, default=1.0)
    parser.add_argument("--Cb", type=float, default=1.0)
    parser.add_argument("--mode", default="lex_boundary_first")
    parser.add_argument("--tad-kind", choices=["all", "hq"], default="all", help="Defaults to kind inferred from input filenames")
    parser.add_argument("--outdir", default="inferred_given_root_robustad")
    parser.add_argument("--leaf-importance-weight", type=float, default=0.1)
    parser.add_argument("--default-tad-score", type=float, default=None)
    parser.add_argument("--default-boundary-score", type=float, default=None)
    parser.add_argument("--no-harmonize", action="store_true", help="Skip boundary harmonization/filtering before LP")
    args = parser.parse_args()

    summary = infer_and_evaluate(
        args.leaf1,
        args.leaf2,
        tolerance=args.tolerance,
        Ct=args.Ct,
        Cb=args.Cb,
        mode=args.mode,
        tad_kind=args.tad_kind,
        outbase=args.outdir,
        leaf_importance_weight=args.leaf_importance_weight,
        default_tad_score=args.default_tad_score,
        default_boundary_score=args.default_boundary_score,
    )

    out_json = (
        Path(args.outdir)
        / "jsons"
        / f"summary_chr{summary['chrom']}_{summary['resolution']}_{summary['leaf1']}_{summary['leaf2']}_{args.tolerance}_{args.tad_kind}.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print("Saved summary to", out_json)


if __name__ == "__main__":
    main()
