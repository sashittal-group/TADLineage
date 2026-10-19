#!/usr/bin/env python3
"""
infer_quartet_tree.py

Infer the hematopoietic tree from four observed TAD hierarchies with the large
parsimony ILP, and check whether it recovers the known topology.

The small parsimony pipeline uses the tree
    LT -> MPP -> {CLP, CMP -> {leaf1, leaf2}}
with LT pinned as the root. Unrooted, MPP and CMP both have degree 3, so this is
the binary quartet {LT, CLP} | {leaf1, leaf2}, and the large parsimony objective
does not depend on where the root sits. We therefore give the ILP the four
observed hierarchies LT, CLP, leaf1, leaf2 as leaves and no tree.

    correct  <=>  the inferred tree groups LT with CLP (and leaf1 with leaf2)

The ILP places the root on the edge above LT (``--free-root`` lifts this). The
objective ignores the root, and every unrooted quartet has an edge to LT, so this
excludes no topology; it only removes the 6 rootings per quartet the solver would
otherwise branch over.

Besides the tree the ILP returns (the free solve), the script scores all three
quartets with the same model (``--no-rank`` skips this), so the JSON records
whether the optimum is unique, tied, or wrong, and by how much.

If the inferred tree is correct, the internal vertex next to LT is MPP and the
other is CMP; both are written out and scored against the held-out calls, as in
check_intermediate_given_root.py.

--weighting adds the importance term of one of the weighted small parsimony
pipelines to the objective (convex mode). The scores are built by the builder
function of that pipeline's own script, imported from it, so they are identical:

  none               plain parsimony (the ontad_unweighted analysis).
  ontad_weighted     hemato_cells/using_ontad/check_intermediate_given_root_weighted.py:
                     RobusTAD TAD scores of MPP and CMP weight the TADs of the internal
                     vertices, - sum_t [x*imp + (1-x)*(1-imp)]. The root sits on LT's
                     edge, so vertex 6 is the vertex next to LT and gets the MPP scores,
                     and vertex 5 is the cherry parent and gets the CMP scores, in all
                     three quartets. Data: --data-dir ../../using_ontad.
  robustad_weighted  hemato_cells/using_robustad/check_intermediate_given_root_weighted_updated.py:
                     RobusTAD TAD and boundary scores of the leaves CLP, leaf1 and leaf2
                     reward their parent for keeping the leaf's TADs and boundaries,
                     w * [c + q * (1 - 2c)] with c = 3 * score. LT gets no scores, as in
                     that pipeline, where LT is the root rather than a leaf.
                     Data: --data-dir ../../using_robustad.

Usage (from an analysis folder, e.g. hemato_cells/large_parsimony/ontad_unweighted):
    python ../infer_quartet_tree.py --data-dir ../.. --chrom 1 --resolution 1mb \
        --leaf1 MEP --leaf2 GMP --tolerance 1 \
        --output results/jsons/quartet_chr1_1mb_MEP_GMP_1.json
"""

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from tad_utilities import read_tad_intervals, parse_interval, write_tads_to_file, match_tads_within_tolerance  # noqa: E402
from filtering import harmonize_leaf_boundaries  # noqa: E402
from large_parsimony_ilp import (  # noqa: E402
    LargeParsimonyILP,
    quartet_partner,
    unrooted_adjacency,
)

ROOT_NAME = "LT"
SISTER_NAME = "CLP"
WEIGHTINGS = ("none", "ontad_weighted", "robustad_weighted")
# Vertex roles with the root on LT's edge (k = 4): 7 -> {LT, 6}, 6 -> {5, leaf}, 5 -> {leaf, leaf}
VERTEX_NEXT_TO_LT, VERTEX_CHERRY_PARENT = 6, 5
WEIGHTED_SCRIPTS = {
    "ontad_weighted": REPO_ROOT / "hemato_cells" / "using_ontad" / "check_intermediate_given_root_weighted.py",
    "robustad_weighted": REPO_ROOT / "hemato_cells" / "using_robustad"
                         / "check_intermediate_given_root_weighted_updated.py",
}


def load_script(path):
    """Import a pipeline script by path (the two weighted scripts share a module name)."""
    spec = importlib.util.spec_from_file_location(f"weighted_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tad_path(data_dir, node, chrom, res):
    return os.path.join(data_dir, node, f"chr{chrom}", res, "tads.tad")


def load_tads(path):
    return {parse_interval(s) for s in read_tad_intervals(path)}


def harmonize_private(paths_by_name, tolerance):
    """harmonize_leaf_boundaries writes *_filtered.tad next to its inputs. Run it on
    private copies so concurrent jobs (other tolerances / leaf pairs sharing LT and
    CLP) never read each other's filtered files."""
    with tempfile.TemporaryDirectory(prefix="lp_harmonize_") as tmp:
        copies = []
        for name, src in paths_by_name.items():
            dst = os.path.join(tmp, f"{name}.tad")
            shutil.copyfile(src, dst)
            copies.append(dst)
        harmonize_leaf_boundaries(copies, tolerance)
        return {name: load_tads(os.path.join(tmp, f"{name}_filtered.tad")) for name in paths_by_name}


def compute_metrics(pred_set, ref_set, tolerance):
    matched_pairs, _, _ = match_tads_within_tolerance(pred_set, ref_set, tolerance)
    matched = len(matched_pairs)
    prec = matched / len(pred_set) if pred_set else 0.0
    rec = matched / len(ref_set) if ref_set else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def split_label(names, partner):
    """'LT,CLP|MEP,GMP' for the quartet that pairs leaf 1 with ``partner``."""
    rest = [names[v] for v in (2, 3, 4) if v != partner]
    return f"{names[1]},{names[partner]}|{rest[0]},{rest[1]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="..", help="folder holding <NODE>/chr<C>/<RES>/tads.tad (default: ..)")
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--resolution", required=True)
    ap.add_argument("--leaf1", required=True, help="leaf of the MEP line (MEP, MKP, MK)")
    ap.add_argument("--leaf2", required=True, help="leaf of the GMP line (GMP, G)")
    ap.add_argument("--tolerance", type=int, default=1, help="boundary tolerance in bins")
    ap.add_argument("--Ct", type=float, default=1.0)
    ap.add_argument("--Cb", type=float, default=1.0)
    ap.add_argument("--mode", default="convex", choices=["convex", "lex_boundary_first", "lex_tad_first"])
    ap.add_argument("--threads", type=int, default=0, help="Gurobi threads (0 = Gurobi default)")
    ap.add_argument("--time-limit", type=float, default=0, help="seconds per Gurobi call (0 = none)")
    ap.add_argument("--no-rank", action="store_true", help="skip scoring the two other quartets")
    ap.add_argument("--lp-method", type=int, default=-1,
                    help="Gurobi Method for LP relaxations: -1 auto, 1 dual simplex, 2 barrier, 3 concurrent")
    ap.add_argument("--solver-log", action="store_true", help="print Gurobi's progress log")
    ap.add_argument("--free-root", action="store_true",
                    help="let the ILP place the root anywhere (same optimum, much slower); "
                         "by default the root sits on the edge above LT")
    ap.add_argument("--weighting", default="none", choices=WEIGHTINGS,
                    help="importance term of a weighted small parsimony pipeline (see above)")
    ap.add_argument("--importance-weight", type=float, default=1.0,
                    help="ontad_weighted: lambda of the internal-vertex importance (pipeline default 1.0)")
    ap.add_argument("--leaf-importance-weight", type=float, default=0.1,
                    help="robustad_weighted: weight of the leaf importance term (pipeline default 0.1)")
    ap.add_argument("--tad-kind", default="all", choices=["all", "hq"],
                    help="which RobusTAD TAD score file the weights come from (pipeline default all)")
    ap.add_argument("--inferred-dir", default="results/inferred",
                    help="where the inferred MPP/CMP .tad files go when the tree is correct")
    ap.add_argument("--output", required=True, help="summary JSON path")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    chrom = str(args.chrom).removeprefix("chr")
    res, tol = args.resolution, args.tolerance

    # Leaf vertex ids 1..4 in the ILP. LT is leaf 1, so a quartet is named by LT's partner.
    names = {1: ROOT_NAME, 2: SISTER_NAME, 3: args.leaf1, 4: args.leaf2}
    ids = {name: v for v, name in names.items()}
    correct_partner = ids[SISTER_NAME]

    raw_paths = {names[v]: tad_path(args.data_dir, names[v], chrom, res) for v in (1, 2, 3, 4)}
    for p in raw_paths.values():
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    print(f"chr{chrom} {res} tol={tol}: leaves {', '.join(names[v] for v in (1, 2, 3, 4))}")
    harmonized = harmonize_private(raw_paths, tol)
    leaf_tads = [harmonized[names[v]] for v in (1, 2, 3, 4)]

    summary = {
        "chrom": chrom,
        "resolution": res,
        "tolerance": tol,
        "leaf1": args.leaf1,
        "leaf2": args.leaf2,
        "leaves": [names[v] for v in (1, 2, 3, 4)],
        "mode": args.mode,
        "Ct": args.Ct,
        "Cb": args.Cb,
        "root_on_lt_edge": not args.free_root,
        "n_tads_after_harmonization": {names[v]: len(leaf_tads[v - 1]) for v in (1, 2, 3, 4)},
        "correct_split": split_label(names, correct_partner),
    }

    t_start = time.time()
    weights = {}
    if args.weighting != "none":
        if args.free_root:
            raise SystemExit("--weighting needs the root on LT's edge (vertex roles); drop --free-root")
        if args.mode != "convex":
            raise SystemExit("--weighting is implemented for --mode convex only")
        script = load_script(WEIGHTED_SCRIPTS[args.weighting])
        if args.weighting == "ontad_weighted":
            # the pipeline's importance(u, a, b) knows MPP as node 1 and CMP as node 3
            imp = script.build_robustad_importance_function(
                chrom=chrom, res=res, tolerance=tol, cmp_truth=None, mpp_truth=None,
                default_nontruth_value=0.1, lambda_weight=args.importance_weight, tad_kind=args.tad_kind,
            )
            weights["vertex_importance"] = {
                VERTEX_NEXT_TO_LT: lambda a, b: imp(1, a, b),
                VERTEX_CHERRY_PARENT: lambda a, b: imp(3, a, b),
            }
        else:
            scored = {v: names[v] for v in (1, 2, 3, 4) if names[v] != ROOT_NAME}
            weights["leaf_importance"] = script.build_leaf_importance(
                node_to_sample=scored,
                leaf_intervals={v: {f"[{a},{b}]" for a, b in leaf_tads[v - 1]} for v in scored},
                chrom=chrom, res=res, tad_kind=args.tad_kind, match_tolerance=tol,
            )
            weights["leaf_importance_weight"] = args.leaf_importance_weight
        summary.update({
            "weighting": args.weighting,
            "weight_script": str(WEIGHTED_SCRIPTS[args.weighting]),
            "importance_weight": args.importance_weight if args.weighting == "ontad_weighted" else None,
            "leaf_importance_weight": (args.leaf_importance_weight
                                       if args.weighting == "robustad_weighted" else None),
            "tad_kind": args.tad_kind,
        })
    else:
        summary["weighting"] = "none"

    ilp = LargeParsimonyILP(
        leaf_tads, Ct=args.Ct, Cb=args.Cb, mode=args.mode,
        threads=args.threads or None, time_limit=args.time_limit or None,
        root_leaf=None if args.free_root else ids[ROOT_NAME],
        lp_method=None if args.lp_method == -1 else args.lp_method,
        solver_log=args.solver_log, verbose=args.verbose, **weights,
    )
    summary.update({
        "n_candidate_tads": len(ilp.candidates),
        "n_candidate_starts": len(ilp.starts),
        "n_candidate_ends": len(ilp.ends),
        "build_seconds": ilp.build_seconds,
    })

    # ---- 1. The large parsimony solve: no tree given --------------------------
    free = ilp.solve()
    summary["status"] = free["status"]
    if free["status"] != "optimal":
        summary["seconds_total"] = time.time() - t_start
        write_json(summary, args.output)
        print(f"Large parsimony solve did not finish: {free['status']}")
        return

    partner = quartet_partner(free["arcs"])
    inferred_correct = partner == correct_partner
    vname = lambda v: names.get(v, f"I{v}")  # noqa: E731
    summary.update({
        "inferred_split": split_label(names, partner),
        "inferred_correct": inferred_correct,
        "cost": free["cost"],
        "parsimony_cost": free["parsimony_cost"],
        "importance_term": free["importance_term"],
        "tad_changes": free["tad_changes"],
        "boundary_changes": free["boundary_changes"],
        "rooted_arcs": [[vname(u), vname(v)] for (u, v) in free["arcs"]],
        "free_solve_seconds": free["seconds"],
        "free_solve_cut_rounds": free["cut_rounds"],
    })
    print(f"Inferred {summary['inferred_split']}  (cost={free['cost']:.6g}, "
          f"T={free['tad_changes']}, B={free['boundary_changes']})  correct={inferred_correct}")

    # ---- 2. Score every quartet with the same model ---------------------------
    if not args.no_rank:
        quartets = {partner: free}
        for c in (2, 3, 4):
            if c != partner:
                quartets[c] = ilp.solve(pin_partner=c)
        failed = [c for c, r in quartets.items() if r["status"] != "optimal"]
        summary["quartets"] = {
            split_label(names, c): (
                {"status": r["status"]} if r["status"] != "optimal" else {
                    "status": "optimal", "cost": r["cost"], "parsimony_cost": r["parsimony_cost"],
                    "importance_term": r["importance_term"], "tad_changes": r["tad_changes"],
                    "boundary_changes": r["boundary_changes"], "seconds": r["seconds"],
                }
            )
            for c, r in sorted(quartets.items())
        }
        if failed:
            summary["optimum_class"] = "rank_incomplete"
        else:
            keys = {c: r["key"] for c, r in quartets.items()}
            best = min(keys.values())
            # A pinned optimum below the free optimum would mean a solver failure.
            if keys[partner] != best:
                summary["rank_inconsistent"] = True
                print("WARNING: a pinned quartet beat the free optimum:", keys)
            optimal = sorted(split_label(names, c) for c, key in keys.items() if key == best)
            best_wrong = min(key for c, key in keys.items() if c != correct_partner)
            if keys[correct_partner] < best_wrong:
                cls = "unique_correct"
            elif keys[correct_partner] == best:
                cls = "tie_with_correct"
            else:
                cls = "wrong"
            summary["optimal_splits"] = optimal
            summary["optimum_class"] = cls
            # margin > 0: the correct quartet beats the best wrong one by this much
            summary["margin"] = [round(w - c, 9) for w, c in zip(best_wrong, keys[correct_partner])]
            summary["cost_correct"] = quartets[correct_partner]["cost"]
            summary["cost_best_wrong"] = min(r["cost"] for c, r in quartets.items() if c != correct_partner)
            print(f"Quartet class: {cls}; optimal: {optimal}; margin: {summary['margin']}")

    # ---- 3. Reconstruct MPP / CMP when the topology is right ------------------
    summary["MPP"] = summary["CMP"] = None
    if inferred_correct:
        adj = unrooted_adjacency(free["arcs"])
        (mpp,) = [w for w in adj[ids[ROOT_NAME]] if w > 4]
        (cmp_,) = [w for w in adj if w > 4 and w != mpp]
        assert ids[SISTER_NAME] in adj[mpp] and {ids[args.leaf1], ids[args.leaf2]} <= adj[cmp_]
        for label, vertex in (("MPP", mpp), ("CMP", cmp_)):
            inferred = free["tads"][vertex]
            out_tad = os.path.join(args.inferred_dir, label, f"chr{chrom}", res,
                                   f"inferred_{args.leaf1}_{args.leaf2}_{tol}.tad")
            write_tads_to_file(inferred, out_tad)
            truth_path = tad_path(args.data_dir, label, chrom, res)
            entry = {"inferred_file": out_tad, "n_inferred": len(inferred)}
            if os.path.exists(truth_path):
                truth = load_tads(truth_path)
                p, r, f1 = compute_metrics(inferred, truth, tol)
                entry.update({"true_file": truth_path, "n_true": len(truth),
                              "precision": p, "recall": r, "f1": f1})
                print(f"{label}: P={p:.3f} R={r:.3f} F1={f1:.3f} (inferred={len(inferred)}, true={len(truth)})")
            summary[label] = entry

    summary["seconds_total"] = time.time() - t_start
    write_json(summary, args.output)


def write_json(summary, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("Saved", path)


if __name__ == "__main__":
    main()
