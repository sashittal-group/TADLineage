#!/usr/bin/env python3
"""
Infer CMP/MPP with LT fixed as root using hemato_cells TADs, while weighting
leaf TAD/boundary terms with OnTAD-biased RobusTAD scores.

Example
-------
From the repository root:
    python hemato_cells/check_intermediate_given_root_weighted_updated.py \
        hemato_cells/MKP/chr3/100kb/tads.tad \
        hemato_cells/GMP/chr3/100kb/tads.tad \
        --tolerance 2

From inside hemato_cells:
    python check_intermediate_given_root_weighted_updated.py \
        MKP/chr3/100kb/tads.tad GMP/chr3/100kb/tads.tad \
        --tolerance 2
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
HEMATO_ROOT = SCRIPT_DIR
ROBUSTAD_ROOT = PROJECT_ROOT / "robustad_predictions"

sys.path.append(str(PROJECT_ROOT))

from filtering import harmonize_leaf_boundaries
from tad_utilities import (
    match_tads_within_tolerance,
    parse_interval,
    read_tad_intervals,
    write_tads_to_file,
)
from check_lp_constrained_weighted_updated import solve_lp_from_edges_and_leaves


def resolve_hemato_path(path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        return p.resolve()
    candidate = HEMATO_ROOT / p
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Missing hemato TAD file: {path}")


def parse_path(path: str | Path) -> tuple[str, str, str]:
    """
    Parse paths like:
      MKP/chr1/500kb/tads.tad
      hemato_cells/MKP/chr1/500kb/tads.tad

    Returns (sample, chrom_without_chr, resolution_label).
    """
    normalized = str(path).rstrip("/")
    m = re.search(r"([^/]+)/chr([^/]+)/([^/]+)/tads\.tad$", normalized)
    if not m:
        raise ValueError(f"Path not in expected format '<NODE>/chr<CHROM>/<RES>/tads.tad': {path}")
    return m.group(1), m.group(2), m.group(3)


def load_tads_from_file(path: str | Path) -> set[tuple[int, int]]:
    intervals = read_tad_intervals(str(path))
    return {parse_interval(s) for s in intervals}


def interval_strings(tads: set[tuple[int, int]]) -> set[str]:
    return {f"[{a},{b}]" for a, b in tads}


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


def robustad_dir(sample: str, chrom: str, res: str) -> Path:
    return ROBUSTAD_ROOT / sample / f"chr{chrom}" / res


def robustad_tads_path(sample: str, chrom: str, res: str, tad_kind: str) -> Path:
    chrom_label = f"chr{chrom}"
    return robustad_dir(sample, chrom, res) / f"{sample}_{chrom_label}_{res}_tads_{tad_kind}_with_bins.tsv"


def robustad_boundary_path(sample: str, chrom: str, res: str) -> Path:
    chrom_label = f"chr{chrom}"
    return robustad_dir(sample, chrom, res) / f"{sample}_{chrom_label}_{res}_boundary_with_bins.tsv"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required RobusTAD score file: {path}\n"
            "Run robustad_predictions/build_ontad_panel_and_infer.py and make sure *_with_bins.tsv files exist."
        )


def load_boundary_scores(sample: str, chrom: str, res: str) -> tuple[dict[int, float], dict[int, float]]:
    path = robustad_boundary_path(sample, chrom, res)
    require_file(path)

    left_scores: dict[int, float] = {}
    right_scores: dict[int, float] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = set(reader.fieldnames or [])
        if "leftScore" not in fieldnames or "rightScore" not in fieldnames:
            raise ValueError(f"{path} missing leftScore/rightScore columns")

        bin_column = "bin" if "bin" in fieldnames else None
        if bin_column is None:
            bin_column = next((name for name in fieldnames if name.startswith("bin")), None)
        if bin_column is None and "start" not in fieldnames:
            raise ValueError(f"{path} missing a bin column and start column")

        res_bp = resolution_to_bp(res)
        for row in reader:
            try:
                if bin_column is not None and row.get(bin_column):
                    b = int(float(row[bin_column]))
                else:
                    b = int(float(row["start"])) // res_bp
            except (TypeError, ValueError):
                continue

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
            try:
                a = int(float(row["start_bin"]))
                b = int(float(row["end_bin"]))
                score = float(row["score"])
            except (TypeError, ValueError):
                continue
            if a >= b:
                continue
            key = (a, b)
            if score > scores.get(key, float("-inf")):
                scores[key] = score
    return scores


def resolution_to_bp(res: str) -> int:
    label = res.lower()
    if label.endswith("kb"):
        return int(float(label[:-2]) * 1_000)
    if label.endswith("mb"):
        return int(float(label[:-2]) * 1_000_000)
    return int(label)


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
    For each hemato leaf interval [a,b], attach OnTAD-biased RobusTAD scores:
      x = TAD score from robustad_predictions/*_tads_{all,hq}_with_bins.tsv
      s = leftScore at start bin a
      e = rightScore at end bin b
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

            x_score = tad_scores.get((a, b))
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


def infer_and_evaluate(
    leaf1_path: str,
    leaf2_path: str,
    tolerance: int = 2,
    Ct: float = 1.0,
    Cb: float = 1.0,
    mode: str = "lex_boundary_first",
    tad_kind: str = "all",
    outbase: str = "inferred_given_root_weighted_updated",
    leaf_importance_weight: float = 0.1,
    default_tad_score: float | None = None,
    default_boundary_score: float | None = None,
    harmonize: bool = True,
):
    leaf1_resolved = resolve_hemato_path(leaf1_path)
    leaf2_resolved = resolve_hemato_path(leaf2_path)
    node1, chrom1, res1 = parse_path(leaf1_resolved)
    node2, chrom2, res2 = parse_path(leaf2_resolved)

    if chrom1 != chrom2 or res1 != res2:
        raise ValueError("Chromosome or resolution mismatch between supplied leaf files.")
    if tad_kind not in {"all", "hq"}:
        raise ValueError("--tad-kind must be 'all' or 'hq'")

    chrom = chrom1
    res = res1

    clp_path = resolve_hemato_path(Path("CLP") / f"chr{chrom}" / res / "tads.tad")
    lt_path = resolve_hemato_path(Path("LT") / f"chr{chrom}" / res / "tads.tad")
    true_cmp_path = resolve_hemato_path(Path("CMP") / f"chr{chrom}" / res / "tads.tad")
    true_mpp_path = resolve_hemato_path(Path("MPP") / f"chr{chrom}" / res / "tads.tad")

    if harmonize:
        harmonize_list = [str(leaf1_resolved), str(leaf2_resolved), str(clp_path), str(lt_path)]
        print("Harmonizing hemato boundaries for:", harmonize_list, f"tolerance={tolerance}")
        harmonize_leaf_boundaries(harmonize_list, tolerance)

        leaf1_for_lp = leaf1_resolved.with_name(leaf1_resolved.stem + "_filtered.tad")
        leaf2_for_lp = leaf2_resolved.with_name(leaf2_resolved.stem + "_filtered.tad")
        clp_for_lp = clp_path.with_name(clp_path.stem + "_filtered.tad")
        lt_for_lp = lt_path.with_name(lt_path.stem + "_filtered.tad")
    else:
        leaf1_for_lp = leaf1_resolved
        leaf2_for_lp = leaf2_resolved
        clp_for_lp = clp_path
        lt_for_lp = lt_path

    for path in (leaf1_for_lp, leaf2_for_lp, clp_for_lp, lt_for_lp):
        if not path.exists():
            raise FileNotFoundError(f"Filtered/input TAD file expected but missing: {path}")

    edges, id_to_name = build_temp_tree_for_selection(node1, node2)

    leaf_intervals_for_lp = {
        2: read_tad_intervals(str(clp_for_lp)),
        4: read_tad_intervals(str(leaf1_for_lp)),
        5: read_tad_intervals(str(leaf2_for_lp)),
    }
    root_intervals_for_lp = read_tad_intervals(str(lt_for_lp))

    true_cmp = load_tads_from_file(true_cmp_path)
    true_mpp = load_tads_from_file(true_mpp_path)

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


    print("Calling updated LP solver with OnTAD-biased RobusTAD leaf scores ...")
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

    filename_suffix = "" if tad_kind == "all" else f"_{tad_kind}"
    fname_cmp = inferred_dir_cmp / f"inferred_{node1}_{node2}_{tolerance}{filename_suffix}.tad"
    fname_mpp = inferred_dir_mpp / f"inferred_{node1}_{node2}_{tolerance}{filename_suffix}.tad"
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
        "hemato_inputs": {
            "LT": str(lt_path),
            "CLP": str(clp_path),
            node1: str(leaf1_resolved),
            node2: str(leaf2_resolved),
            "CMP_truth": str(true_cmp_path),
            "MPP_truth": str(true_mpp_path),
        },
        "robustad_score_inputs": {
            "CLP_tads": str(robustad_tads_path("CLP", chrom, res, tad_kind)),
            "CLP_boundary": str(robustad_boundary_path("CLP", chrom, res)),
            f"{node1}_tads": str(robustad_tads_path(node1, chrom, res, tad_kind)),
            f"{node1}_boundary": str(robustad_boundary_path(node1, chrom, res)),
            f"{node2}_tads": str(robustad_tads_path(node2, chrom, res, tad_kind)),
            f"{node2}_boundary": str(robustad_boundary_path(node2, chrom, res)),
        },
        "counts": {
            "LT_root": len(root_intervals_for_lp),
            "CLP_leaf": len(leaf_intervals_for_lp[2]),
            f"{node1}_leaf": len(leaf_intervals_for_lp[4]),
            f"{node2}_leaf": len(leaf_intervals_for_lp[5]),
            "candidate_intervals": len(intervals),
        },
        "CMP": {
            "inferred_file": str(fname_cmp),
            "true_file": str(true_cmp_path),
            "precision": cmp_prec,
            "recall": cmp_rec,
            "f1": cmp_f1,
            "n_inferred": len(tads_cmp),
            "n_true": len(true_cmp),
        },
        "MPP": {
            "inferred_file": str(fname_mpp),
            "true_file": str(true_mpp_path),
            "precision": mpp_prec,
            "recall": mpp_rec,
            "f1": mpp_f1,
            "n_inferred": len(tads_mpp),
            "n_true": len(true_mpp),
        },
    }

    print("\n=== Summary ===")
    print(f"Chrom: {chrom}  Res: {res}  RobusTAD score source: {tad_kind}")
    print(f"Leaves used: {node1}, {node2}, CLP | given root: LT")
    print("LP cost:", summary["lp_cost"], "integral:", summary["is_integral"])
    print(
        "CMP: P={:.3f} R={:.3f} F1={:.3f} (inferred={}, hemato={})".format(
            cmp_prec, cmp_rec, cmp_f1, len(tads_cmp), len(true_cmp)
        )
    )
    print(
        "MPP: P={:.3f} R={:.3f} F1={:.3f} (inferred={}, hemato={})".format(
            mpp_prec, mpp_rec, mpp_f1, len(tads_mpp), len(true_mpp)
        )
    )
    print("=== End ===\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Infer CMP/MPP from hemato TAD leaves and fixed LT root, using "
            "OnTAD-biased RobusTAD leaf scores from robustad_predictions."
        )
    )
    parser.add_argument("leaf1", help="first hemato .tad, e.g. MKP/chr3/100kb/tads.tad")
    parser.add_argument("leaf2", help="second hemato .tad, e.g. GMP/chr3/100kb/tads.tad")
    parser.add_argument("--tolerance", type=int, default=2, help="Boundary/TAD tolerance in bins")
    parser.add_argument("--Ct", type=float, default=1.0)
    parser.add_argument("--Cb", type=float, default=1.0)
    parser.add_argument("--mode", default="lex_boundary_first")
    parser.add_argument("--tad-kind", choices=["all", "hq"], default="all")
    parser.add_argument("--outdir", default="inferred_given_root_weighted_updated")
    parser.add_argument("--leaf-importance-weight", type=float, default=0.1)
    parser.add_argument("--default-tad-score", type=float, default=None)
    parser.add_argument("--default-boundary-score", type=float, default=None)
    parser.add_argument("--no-harmonize", action="store_true", help="Skip hemato boundary harmonization/filtering")
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
        harmonize=not args.no_harmonize,
    )

    filename_suffix = "" if args.tad_kind == "all" else f"_{args.tad_kind}"
    out_json = (
        Path(args.outdir)
        / "jsons"
        / (
            f"summary_chr{summary['chrom']}_{summary['resolution']}_"
            f"{summary['leaf1']}_{summary['leaf2']}_{args.tolerance}{filename_suffix}.json"
        )
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print("Saved summary to", out_json)


if __name__ == "__main__":
    main()
