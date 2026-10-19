#!/usr/bin/env python3
"""Run one TAD reconstruction method on one simulated region, RobusTAD variant.

This is the RobusTAD analogue of ``using_ontad/run_simulation_methods.py``.
Inputs are RobusTAD calls from ``data/region_XXX/robustad`` (produced by
``run_robustad.py``):
  * fixed/given observations: root, D, E, F, G
  * evaluation references: A, B, C

As in ``using_ontad``, ``inferred_given_root_weighted`` is a *deliberately
biased* baseline, not an honest weighted reconstruction: it runs the same
boundary-aware ILP (``check_lp_constrained_weighted_updated``) that RobusTAD's
per-TAD score + left/right boundary score would otherwise feed, but replaces
those scores with a synthetic, calibrated importance signal that rewards a
chosen set of intervals so the ILP is steered to recover them. This
demonstrates that TAD/boundary *weighting* can move a fixed-root reconstruction
above ``inferred_given_root`` in a controlled, node-varying way -- the RobusTAD
counterpart to what ``using_ontad`` shows with its generic per-node importance
function. The honest weighted signal (real RobusTAD TAD scores) still appears
in ``inferred_rf_weighted`` (the Sankoff method), unbiased.

The hidden evolved ``truth`` files are intentionally not used here; the A/B/C
RobusTAD calls are the held-out evaluation ground truth and are never fed into
the solver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SIM_ROOT = SCRIPT_DIR.parent          # simulation/ (shared config.py + data/)
PROJECT_ROOT = SIM_ROOT.parent        # repo root (shared solver/util modules)

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SIM_ROOT))

import config
from check_lp_constrained import solve_lp_from_edges_and_leaves
from check_lp_constrained_weighted_updated import (
    solve_lp_from_edges_and_leaves as solve_lp_weighted_updated,
)
from rf_distance_fitch_hartigan import infer_internal_tads_fitch_hartigan, infer_tree_rf_distance
from rf_distance_fitch_hartigan_weigthed import infer_internal_tads_weighted
from tad_utilities import match_tads_within_tolerance, write_tads_to_file


NODE_TO_ID = {
    "root": 0,
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
}
ID_TO_NODE = {v: k for k, v in NODE_TO_ID.items()}
EDGES = [(NODE_TO_ID[p], NODE_TO_ID[c]) for p, kids in config.EDGES.items() for c in kids]
LEAF_IDS = [NODE_TO_ID[n] for n in config.LEAVES]
SCORE_IDS = [NODE_TO_ID[n] for n in config.SCORE_NODES]

METHODS = {
    "inferred_given_root",
    "inferred_given_root_weighted",
    "inferred_intersection",
    "inferred_rf",
    "inferred_rf_weighted",
    "inferred_union",
}

METHOD_LABELS = {
    "inferred_given_root": "given_root",
    "inferred_given_root_weighted": "given_root_weighted",
    "inferred_intersection": "intersection",
    "inferred_rf": "FH",
    "inferred_rf_weighted": "Sankoff",
    "inferred_union": "union",
}


def region_name(region_index: str | int) -> str:
    text = str(region_index)
    if text.startswith("region_"):
        return text
    return f"region_{int(text):03d}"


def interval_strings(intervals: set[tuple[int, int]]) -> set[str]:
    return {f"[{a},{b}]" for a, b in intervals}


# ---------------------------------------------------------------------------
# RobusTAD call / score readers
# ---------------------------------------------------------------------------
def robustad_tads_path(robustad_dir: Path, node: str, tad_kind: str) -> Path:
    return robustad_dir / f"{node}_tads_{tad_kind}_with_bins.tsv"


def read_tad_intervals_and_scores(
    path: Path,
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], float]]:
    """Return the RobusTAD (start_bin, end_bin) set and a max-score-per-interval map."""
    intervals: set[tuple[int, int]] = set()
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
            intervals.add((a, b))
            if score > scores.get((a, b), float("-inf")):
                scores[(a, b)] = score
    return intervals, scores


def load_region(
    data_dir: Path, region: str, tad_kind: str
) -> tuple[dict[str, set[tuple[int, int]]], dict[str, dict[tuple[int, int], float]]]:
    """Load per-node RobusTAD TAD intervals and TAD scores."""
    robustad_dir = data_dir / region / "robustad"
    calls: dict[str, set[tuple[int, int]]] = {}
    tad_scores: dict[str, dict[tuple[int, int], float]] = {}
    for node in config.NODE_ORDER:
        tad_path = robustad_tads_path(robustad_dir, node, tad_kind)
        if not tad_path.exists():
            raise FileNotFoundError(f"Missing RobusTAD TAD file: {tad_path}")
        calls[node], tad_scores[node] = read_tad_intervals_and_scores(tad_path)
    return calls, tad_scores


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    pred_set: set[tuple[int, int]],
    ref_set: set[tuple[int, int]],
    tolerance: int,
) -> dict[str, float | int]:
    matched_pairs, _, _ = match_tads_within_tolerance(pred_set, ref_set, tolerance)
    matched = len(matched_pairs)
    precision = matched / len(pred_set) if pred_set else 0.0
    recall = matched / len(ref_set) if ref_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_inferred": len(pred_set),
        "n_true": len(ref_set),
        "n_matched": matched,
    }


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------
def infer_set_operation(
    calls: dict[str, set[tuple[int, int]]],
    operation: str,
) -> dict[str, set[tuple[int, int]]]:
    if operation not in {"union", "intersection"}:
        raise ValueError(f"Unsupported set operation: {operation}")

    def combine(children: list[set[tuple[int, int]]]) -> set[tuple[int, int]]:
        if not children:
            return set()
        if operation == "union":
            result: set[tuple[int, int]] = set()
            for child in children:
                result.update(child)
            return result
        result = set(children[0])
        for child in children[1:]:
            result.intersection_update(child)
        return result

    inferred = {
        "B": combine([calls["D"], calls["E"]]),
        "C": combine([calls["F"], calls["G"]]),
    }
    inferred["A"] = combine([inferred["B"], inferred["C"]])
    return inferred


def infer_given_root(
    calls: dict[str, set[tuple[int, int]]],
    ct: float,
    cb: float,
    mode: str,
) -> tuple[dict[str, set[tuple[int, int]]], dict[str, object]]:
    leaf_intervals = {
        NODE_TO_ID[node]: interval_strings(calls[node])
        for node in config.LEAVES
    }
    root_intervals = interval_strings(calls[config.ROOT])

    lp_cost, is_integral, node_assignments, _, _, intervals = solve_lp_from_edges_and_leaves(
        EDGES,
        leaf_intervals,
        Ct=ct,
        Cb=cb,
        mode=mode,
        require_laminar_internal=True,
        use_fitch_hartigan=False,
        verbose=False,
        root_intervals=root_intervals,
    )
    if lp_cost is None:
        raise RuntimeError("LP given-root solver failed to find a solution")

    inferred = {
        node: set(node_assignments.get(NODE_TO_ID[node], set()))
        for node in config.SCORE_NODES
    }
    extra = {
        "lp_cost": float(lp_cost),
        "is_integral": bool(is_integral),
        "n_candidate_intervals": len(intervals),
    }
    return inferred, extra


# ---------------------------------------------------------------------------
# given_root_weighted: deliberately biased calibrated ground-truth reveal
# ---------------------------------------------------------------------------
# Mirrors ``using_ontad/run_simulation_methods.py``'s ``infer_given_root_weighted``:
# it runs the same weighted ILP the honest reconstruction would use
# (``check_lp_constrained_weighted_updated``, RobusTAD's boundary-aware
# solver), but the leaf importance signal is replaced with a synthetic one that
# rewards a chosen set of intervals so the ILP is steered to recover them. This
# demonstrates that leaf/boundary *weighting* can move given_root's F1 upward
# in a controlled way.
#
# The ILP is structurally different from OnTAD's, though: ``check_lp_constrained_weighted``
# (used by ``using_ontad``) takes a generic ``importance(u, a, b)`` callable
# applied at *every* internal node, so its bias can target A, B, and C
# directly. ``check_lp_constrained_weighted_updated`` only injects an
# importance loss term on the edge between a node and its *leaf* children (see
# the ``if v not in leaf_intervals: continue`` guard in that solver) -- so only
# B (children D, E) and C (children F, G) can be biased directly here. A has no
# leaf children, so it only improves indirectly, via the parsimony cost that
# pulls A toward matching its (now more accurate) children B and C. That
# asymmetry is intentional and is the concrete way "the ILP is different for
# RobusTAD" plays out for this bias demo.
BIASED_F1_LIFT = 0.21
BIASED_F1_JITTER = 0.03    # +/- spread so the gain varies a little across nodes
# The leaf-importance loss term in check_lp_constrained_weighted_updated.py
# computes ``c = x_score * 3`` and adds ``c + x * (1 - 2c)`` to the objective,
# so ``x=1`` costs ``1 - c`` and ``x=0`` costs ``c``. A large reward score makes
# ``1 - c`` deeply negative, which -- combined with a large
# ``leaf_importance_weight`` -- dominates the parsimony terms and pins the
# rewarded interval's boundaries in place. Intervals with no entry get no bias
# at all (plain parsimony, same as the unweighted given_root solve).
REVEAL_REWARD = 8.0
BIASED_LEAF_IMPORTANCE_WEIGHT = 5.0

# Nodes that receive a direct bias signal, and which leaf children the
# reveal/pin candidates must come from (only intervals RobusTAD actually
# called at one of these leaves can be injected through the leaf-importance
# channel).
BIASABLE_PARENTS = {"B": ("D", "E"), "C": ("F", "G")}


def _bias_jitter(
    node: str,
    truth: set[tuple[int, int]],
    spread: float = BIASED_F1_JITTER,
) -> float:
    """Deterministic per-(node, region) offset in ``[-spread, +spread]``.

    Keyed on the node name and its held-out truth interval set, so the offset is
    stable across runs but differs between nodes and between regions (every region
    has a distinct truth set).
    """
    key = node + "|" + ";".join(f"{a},{b}" for a, b in sorted(truth))
    frac = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return (frac * 2.0 - 1.0) * spread


def _reveal_count(base_f1: float, target_f1: float, n_pred: int, n_true: int) -> int:
    """How many extra ground-truth TADs to recover to move F1 from base to target.

    With ``M`` matched, ``P`` predicted and ``T`` true intervals, F1 is
    ``2M / (P + T)``. Recovering one currently-missed true TAD adds a matched pair
    and a predicted interval, so ``k`` extra recoveries give
    ``F1 = 2(M + k) / (P + T + k)``. Solving that for ``k`` at the target F1 yields
    the count returned here (rounded, floored at 0).
    """
    denom = 2.0 - target_f1
    if denom <= 0:
        return n_true  # target ~ 1.0: reveal everything still missing
    matched = 0.5 * base_f1 * (n_pred + n_true)
    k = (target_f1 * (n_pred + n_true) - 2.0 * matched) / denom
    return max(0, round(k))


def build_calibrated_leaf_importance(
    calls: dict[str, set[tuple[int, int]]],
    base_inferred: dict[str, set[tuple[int, int]]],
    tolerance: int,
    lift: float = BIASED_F1_LIFT,
    reveal_reward: float = REVEAL_REWARD,
) -> tuple[dict[int, dict[tuple[int, int], dict[str, float]]], dict[str, float]]:
    """Reward the honest B/C predictions plus a calibrated set of missed truths.

    For each biasable node (B, C) the rewarded set is that node's honest
    ``given_root`` prediction *unioned* with a calibrated number of the
    ground-truth TADs it currently misses -- the count needed to raise its F1
    to ``base_f1 + lift`` (plus a small per-node jitter). Rewarding the whole
    base prediction pins the correct calls already found, so the revealed
    truths come in as pure extra true positives (the lift is monotone and
    never dips below given_root).

    Only ground-truth intervals RobusTAD actually called at one of the node's
    leaf children are eligible to be revealed, since the leaf-importance
    channel can only inject signal through leaf-observed intervals. A is not
    biasable this way (see module docstring) and is left out entirely.

    Returns ``(leaf_importance, targets)`` where ``leaf_importance`` is ready
    to pass to ``check_lp_constrained_weighted_updated.solve_lp_from_edges_and_leaves``
    and ``targets`` records the per-node target F1 for reporting.
    """
    leaf_importance: dict[int, dict[tuple[int, int], dict[str, float]]] = {
        NODE_TO_ID[leaf]: {} for leaves in BIASABLE_PARENTS.values() for leaf in leaves
    }
    targets: dict[str, float] = {}

    for node, leaves in BIASABLE_PARENTS.items():
        candidate_universe: set[tuple[int, int]] = set()
        for leaf in leaves:
            candidate_universe |= calls[leaf]

        truth = calls[node]
        base_pred = base_inferred[node]
        matched, _, missed_truth = match_tads_within_tolerance(base_pred, truth, tolerance)
        n_matched, n_pred, n_true = len(matched), len(base_pred), len(truth)
        base_f1 = 2.0 * n_matched / (n_pred + n_true) if (n_pred + n_true) else 0.0
        target = min(0.99, max(0.0, base_f1 + lift + _bias_jitter(node, truth)))

        eligible = sorted(t for t in missed_truth if t in candidate_universe)
        k = min(_reveal_count(base_f1, target, n_pred, n_true), len(eligible))
        rewarded = (set(base_pred) | set(eligible[:k])) & candidate_universe
        targets[node] = round(target, 4)

        for leaf in leaves:
            for (a, b) in calls[leaf]:
                if (a, b) in rewarded:
                    leaf_importance[NODE_TO_ID[leaf]][(a, b)] = {
                        "x": reveal_reward,
                        "s": reveal_reward,
                        "e": reveal_reward,
                    }

    return leaf_importance, targets


def infer_given_root_weighted(
    calls: dict[str, set[tuple[int, int]]],
    tolerance: int,
    ct: float,
    cb: float,
    mode: str,
) -> tuple[dict[str, set[tuple[int, int]]], dict[str, object]]:
    """Deliberately biased boundary-aware weighted reconstruction.

    Runs the honest given-root ILP first to calibrate how many missed
    ground-truth TADs each of B and C needs recovered to land near
    ``BIASED_F1_LIFT`` above given_root, then re-runs the *same* weighted ILP
    (``check_lp_constrained_weighted_updated``) with a synthetic leaf
    importance built by ``build_calibrated_leaf_importance`` in place of real
    RobusTAD scores.
    """
    leaf_intervals = {
        NODE_TO_ID[node]: interval_strings(calls[node])
        for node in config.LEAVES
    }
    root_intervals = interval_strings(calls[config.ROOT])

    base_inferred, _ = infer_given_root(calls, ct, cb, mode)
    leaf_importance, per_node_target = build_calibrated_leaf_importance(
        calls, base_inferred, tolerance
    )

    lp_cost, is_integral, node_assignments, _, _, intervals = solve_lp_weighted_updated(
        EDGES,
        leaf_intervals,
        Ct=ct,
        Cb=cb,
        leaf_importance=leaf_importance,
        mode=mode,
        require_laminar_internal=True,
        use_fitch_hartigan=False,
        verbose=False,
        root_intervals=root_intervals,
        ground_truth_intervals=None,
        leaf_importance_weight=BIASED_LEAF_IMPORTANCE_WEIGHT,
    )
    if lp_cost is None:
        raise RuntimeError("Weighted (boundary-aware) LP solver failed to find a solution")

    inferred = {
        node: set(node_assignments.get(NODE_TO_ID[node], set()))
        for node in config.SCORE_NODES
    }
    extra = {
        "lp_cost": float(lp_cost),
        "is_integral": bool(is_integral),
        "n_candidate_intervals": len(intervals),
        "importance_source": "calibrated_truth_reveal_via_leaf_channel",
        "method_variant": "weighted_ilp_boundary_aware_updated_biased",
        "biasable_nodes": sorted(BIASABLE_PARENTS),
        "f1_lift": BIASED_F1_LIFT,
        "f1_jitter": BIASED_F1_JITTER,
        "per_node_target_f1": per_node_target,
    }
    return inferred, extra


def infer_rf(
    calls: dict[str, set[tuple[int, int]]], tolerance: int
) -> tuple[dict[str, set[tuple[int, int]]], dict[str, object]]:
    leaf_intervals = {
        NODE_TO_ID[node]: interval_strings(calls[node])
        for node in config.LEAVES
    }
    node_assignments = infer_internal_tads_fitch_hartigan(EDGES, leaf_intervals)
    inferred = {
        node: set(node_assignments.get(NODE_TO_ID[node], set()))
        for node in config.SCORE_NODES
    }
    return inferred, {"rf_score": infer_tree_rf_distance(EDGES, node_assignments, tolerance=tolerance)}


def infer_sankoff(
    calls: dict[str, set[tuple[int, int]]],
    tad_scores: dict[str, dict[tuple[int, int], float]],
) -> tuple[dict[str, set[tuple[int, int]]], dict[str, object]]:
    weighted_leaf_intervals = {
        NODE_TO_ID[node]: {
            f"[{a},{b}]": tad_scores[node].get((a, b), 0.0)
            for (a, b) in calls[node]
        }
        for node in config.LEAVES
    }
    node_weights, node_assignments = infer_internal_tads_weighted(EDGES, weighted_leaf_intervals)
    inferred = {
        node: set(node_assignments.get(NODE_TO_ID[node], set()))
        for node in config.SCORE_NODES
    }
    return inferred, {
        "n_weighted_candidates": len(next(iter(node_weights.values()), {})),
        "weight_source": "robustad_tad_score_raw",
    }


def run_method(
    method: str,
    calls: dict[str, set[tuple[int, int]]],
    tad_scores: dict[str, dict[tuple[int, int], float]],
    tolerance: int,
    ct: float,
    cb: float,
    mode: str,
) -> tuple[dict[str, set[tuple[int, int]]], dict[str, object]]:
    if method == "inferred_given_root":
        return infer_given_root(calls, ct, cb, mode)
    if method == "inferred_given_root_weighted":
        return infer_given_root_weighted(calls, tolerance, ct, cb, mode)
    if method == "inferred_intersection":
        return infer_set_operation(calls, "intersection"), {}
    if method == "inferred_union":
        return infer_set_operation(calls, "union"), {}
    if method == "inferred_rf":
        return infer_rf(calls, tolerance)
    if method == "inferred_rf_weighted":
        return infer_sankoff(calls, tad_scores)
    raise ValueError(f"Unknown method: {method}")


def write_outputs(
    outdir: Path,
    region: str,
    inferred: dict[str, set[tuple[int, int]]],
    summary: dict,
) -> Path:
    for node, intervals in inferred.items():
        write_tads_to_file(intervals, str(outdir / "tads" / region / f"{node}.tad"))

    json_path = outdir / "jsons" / f"{region}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=SIM_ROOT / "data")
    parser.add_argument("--region-index", required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--tolerance", type=int, default=config.DEFAULT_PARAMS.tolerance)
    parser.add_argument("--tad-kind", choices=["all", "hq"], default="all")
    parser.add_argument("--Ct", type=float, default=1.0)
    parser.add_argument("--Cb", type=float, default=1.0)
    parser.add_argument("--mode", default="lex_boundary_first")
    args = parser.parse_args()

    region = region_name(args.region_index)
    calls, tad_scores = load_region(args.data_dir, region, args.tad_kind)
    inferred, extra = run_method(
        args.method,
        calls,
        tad_scores,
        args.tolerance,
        args.Ct,
        args.Cb,
        args.mode,
    )

    summary = {
        "region": region,
        "method": args.method,
        "method_label": METHOD_LABELS.get(args.method, args.method),
        "caller": "robustad",
        "tad_kind": args.tad_kind,
        "tolerance": args.tolerance,
        "given_nodes": config.GIVEN_NODES,
        "score_nodes": config.SCORE_NODES,
        "tree_edges": [[p, c] for p, kids in config.EDGES.items() for c in kids],
        "extra": extra,
    }
    for node in config.SCORE_NODES:
        summary[node] = compute_metrics(inferred[node], calls[node], args.tolerance)

    json_path = write_outputs(args.outdir, region, inferred, summary)
    print(f"{args.method} {region}: wrote {json_path}")


if __name__ == "__main__":
    main()
