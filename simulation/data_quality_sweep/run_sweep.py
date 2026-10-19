#!/usr/bin/env python3
"""Data-quality sweep for internal-node TAD reconstruction.

For one (tree, region) the script evolves the hidden ground-truth hierarchies
top-down from the root, then measures how well the internal nodes are
reconstructed as the *input data quality* degrades. The reconstruction is always
scored against the TRUE internal hierarchies (never against a caller's own
output), so the F1 drop reflects real error.

Quality ladder (x-axis, best -> worst):

  perfect TADs            true root/leaf TADs fed straight in (no caller;
                          the reconstruction-method ceiling)
  high-depth Hi-C         base matrix, called by OnTAD / RobusTAD
  low-depth Hi-C          base matrix down-sampled, then called

The single degradation axis is sequencing depth (binomial down-sampling of the
base matrix to a fraction of its counts), swept at several bin sizes
(resolutions). Bin size is modeled as a counts-DENSITY knob: at a fixed region
depth, finer bins pool fewer reads, so expected counts per pixel scale as
(bin_kb)^2. The TAD structure (in bins) and the caller settings (in bins) are
held fixed across resolutions, so only count density varies -- this isolates the
depth x density interaction and keeps F1 comparable across bin sizes. Coarse bins
stay count-rich (robust to down-sampling); fine bins go count-poor (F1 falls off
as depth drops). Random-ligation noise was removed on the professor's note that
Hi-C is itself noisy, so a separate noise axis is redundant.

Stages
------
--stage call  --caller ontad|robustad : run the caller on every depth+noise
                                         level, store scored given-node calls.
--stage score --caller ontad|robustad : reconstruct all six methods from the
                                         cached calls and score vs truth.
--stage score --caller perfect        : the caller-free true-TAD ceiling.

The SCORE stage reconstructs all six methods (given_root, given_root_weighted,
rf, rf_weighted, union, intersection) at every quality level and writes one JSON
of per-(level, method) average precision/recall/F1 (averaged over the tree's
scored internal nodes) for one (tree, region).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import sys
import zlib
from pathlib import Path

import numpy as np

SWEEP_DIR = Path(__file__).resolve().parent
SIM_ROOT = SWEEP_DIR.parent            # simulation/ (shared generation modules)
REPO_ROOT = SIM_ROOT.parent            # repo root (shared solvers)
for p in (str(REPO_ROOT), str(SIM_ROOT), str(SWEEP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from evolve import evolve
from hierarchy import all_domains, generate_root, iter_domains
from simulate_matrix import simulate_matrix
from tad_utilities import match_tads_within_tolerance

from trees import Tree, get_tree

# The reconstruction solvers are imported lazily inside reconstruct(): the
# given-root ILP pulls in gurobipy, which is absent from caller-only envs (e.g.
# RobusTAD's cooler env). The CALL stage never reconstructs, so it must not
# import them at module load.


# ---------------------------------------------------------------------------
# Degradation ladder (depth) + resolution family (bin size).
#
# DEPTH_LEVELS is ordered best -> worst and shares the "perfect" anchor plotted
# to its left. Every depth ladder is run once per bin size in BIN_SIZE_LEVELS.
#
# Bin size is a counts-density scaler, not a change in matrix geometry: the
# base matrix amplitude is set to A_REF * (bin_kb / REF_KB)^2, so finer bins are
# proportionally sparser (fixed region depth => reads/pixel ~ bin_kb^2). The
# matrix stays N_BINS x N_BINS with the same TAD structure, and the caller runs
# with the same bin-unit settings, so only count density changes across the
# family. REF_KB / A_REF pin the reference: 40 kb reproduces the old count-rich
# (depth-insensitive) regime.
# ---------------------------------------------------------------------------
DEPTH_LEVELS: list[tuple[str, float]] = [
    ("1.0x", 1.0),     # full base coverage (cleanest Hi-C)
    ("0.3x", 0.3),
    ("0.1x", 0.1),
    ("0.03x", 0.03),
    ("0.01x", 0.01),   # very shallow
]
REF_KB = 40.0          # resolution at which A_REF applies (the old regime)
A_REF = 2000.0         # reference contact amplitude at REF_KB
BIN_SIZE_LEVELS: list[tuple[str, float]] = [
    ("40kb", 40.0),    # coarse: count-rich, flat across depth (old regime)
    ("10kb", 10.0),
    ("5kb", 5.0),
    ("2.5kb", 2.5),    # very fine: count-poor, F1 falls off steeply with depth
]


def amplitude_for_bin_kb(bin_kb: float, a_ref: float = A_REF, ref_kb: float = REF_KB) -> float:
    """Base-matrix amplitude for a resolution: reads/pixel ~ bin_kb^2."""
    return a_ref * (bin_kb / ref_kb) ** 2


# ---------------------------------------------------------------------------
# Ground-truth hierarchies for one (tree, region)
# ---------------------------------------------------------------------------
def _stable_hash(text: str) -> int:
    """Process-stable hash (Python's built-in hash() is salted per process)."""
    return zlib.crc32(text.encode())


def build_hierarchies(tree: Tree, region: int, params: config.SimParams) -> dict[str, list]:
    """Evolve every node's hidden hierarchy top-down from the root.

    Regenerates (up to ``max_generation_attempts``) until every scored internal
    node carries at least ``min_score_node_tads`` domains, so F1 is meaningful.
    """
    base_seed = params.seed * 1_000_003 + _stable_hash(tree.name) % 100_000 + region
    for attempt in range(params.max_generation_attempts):
        rng = random.Random(base_seed + attempt * 9_176)
        hiers: dict[str, list] = {
            tree.root: generate_root(
                params.n_bins, params.n_top_tads, params.nesting_depth,
                params.min_size, rng, gap_prob=params.gap_prob,
            )
        }
        for parent in tree.nodes:
            for child in tree.edges.get(parent, ()):
                hiers[child] = evolve(hiers[parent], params.events_per_edge, params, rng)

        counts = {n: len(all_domains(hiers[n])) for n in tree.scored}
        if not counts or min(counts.values()) >= params.min_score_node_tads:
            return hiers

    raise RuntimeError(
        f"{tree.name}/region_{region:03d}: could not evolve >= "
        f"{params.min_score_node_tads} TADs at every scored node in "
        f"{params.max_generation_attempts} attempts (last: {counts})"
    )


def true_intervals(hier: list) -> set[tuple[int, int]]:
    """True TAD set in the truth convention: 0-indexed [s, e) -> (s + 1, e)."""
    return {(d.start + 1, d.end) for d in iter_domains(hier)}


# ---------------------------------------------------------------------------
# Matrix degradation
# ---------------------------------------------------------------------------
def _symmetrize_upper(upper_vals: np.ndarray, iu: tuple, n: int) -> np.ndarray:
    out = np.zeros((n, n), dtype=np.int64)
    out[iu] = upper_vals
    out = np.triu(out)
    return out + out.T - np.diag(np.diag(out))


def downsample(mat: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Thin counts to a fraction of coverage (binomial per contact)."""
    if frac >= 1.0:
        return mat.copy()
    n = mat.shape[0]
    iu = np.triu_indices(n)
    thinned = rng.binomial(mat[iu].astype(np.int64), frac)
    return _symmetrize_upper(thinned, iu, n)


# ---------------------------------------------------------------------------
# Reconstruction (tree-agnostic) + scoring
# ---------------------------------------------------------------------------
def _intervals_to_strings(intervals) -> set[str]:
    return {f"[{a},{b}]" for a, b in intervals}


# ---------------------------------------------------------------------------
# The six reconstruction methods, mirroring the fixed-experiment runners
# (``using_ontad`` / ``using_robustad``) but tree-agnostic (they derive the tree
# from ``tree.edge_ids`` / ``tree.node_id``, so any generated topology works):
#
#   given_root           laminar given-root ILP           (check_lp_constrained)
#   given_root_weighted  deliberately biased weighted ILP -- CALLER-SPECIFIC:
#                          OnTAD  -> check_lp_constrained_weighted
#                          RobusTAD -> check_lp_constrained_weighted_updated
#   rf                   Fitch-Hartigan parsimony
#   rf_weighted          Sankoff DP weighted by the caller's per-TAD score
#   union / intersection child-set combination up the tree
# ---------------------------------------------------------------------------
METHODS = ["given_root", "given_root_weighted", "rf", "rf_weighted", "union", "intersection"]

METHOD_LABELS = {
    "given_root": "given_root",
    "given_root_weighted": "given_root_weighted",
    "rf": "FH",
    "rf_weighted": "Sankoff",
    "union": "union",
    "intersection": "intersection",
}


def _assign_to_scored(tree: Tree, node_assignments: dict) -> dict[str, set[tuple[int, int]]]:
    nid = tree.node_id
    return {
        name: {tuple(iv) for iv in node_assignments.get(nid[name], set())}
        for name in tree.scored
    }


def infer_given_root(tree, given_intervals, ct, cb, mode):
    from check_lp_constrained import solve_lp_from_edges_and_leaves
    nid = tree.node_id
    leaf_intervals = {nid[l]: _intervals_to_strings(given_intervals[l]) for l in tree.leaves}
    root_intervals = _intervals_to_strings(given_intervals[tree.root])
    lp_cost, _, node_assignments, _, _, _ = solve_lp_from_edges_and_leaves(
        tree.edge_ids, leaf_intervals, Ct=ct, Cb=cb, mode=mode,
        require_laminar_internal=True, use_fitch_hartigan=False,
        verbose=False, root_intervals=root_intervals,
    )
    if lp_cost is None:
        raise RuntimeError("given-root LP found no solution")
    return _assign_to_scored(tree, node_assignments)


def infer_rf(tree, given_intervals):
    from rf_distance_fitch_hartigan import infer_internal_tads_fitch_hartigan
    nid = tree.node_id
    leaf_intervals = {nid[l]: _intervals_to_strings(given_intervals[l]) for l in tree.leaves}
    node_assignments = infer_internal_tads_fitch_hartigan(tree.edge_ids, leaf_intervals)
    return _assign_to_scored(tree, node_assignments)


def infer_sankoff(tree, given_intervals, given_scores):
    """Weighted parsimony; each leaf interval carries the caller's raw score."""
    from rf_distance_fitch_hartigan_weigthed import infer_internal_tads_weighted
    nid = tree.node_id
    weighted_leaf = {
        nid[l]: {f"[{a},{b}]": given_scores.get(l, {}).get((a, b), 0.0) for (a, b) in given_intervals[l]}
        for l in tree.leaves
    }
    _, node_assignments = infer_internal_tads_weighted(tree.edge_ids, weighted_leaf)
    return _assign_to_scored(tree, node_assignments)


def infer_set_operation(tree, given_intervals, operation):
    """Union/intersection of child sets, computed leaves-up over the tree."""
    value: dict[str, set[tuple[int, int]]] = {}
    for node in reversed(tree.nodes):        # reversed BFS: children precede parents
        kids = tree.edges.get(node, ())
        if not kids:
            value[node] = set(given_intervals.get(node, set()))
            continue
        child_sets = [value[k] for k in kids]
        combined = set(child_sets[0])
        for cs in child_sets[1:]:
            if operation == "union":
                combined |= cs
            else:
                combined &= cs
        value[node] = combined
    return {name: value[name] for name in tree.scored}


# --- given_root_weighted: the deliberately biased baseline -----------------
# Same design as the fixed experiment: solve the honest given-root ILP, then
# re-solve the CALLER-SPECIFIC weighted ILP with a synthetic importance signal
# that rewards the honest prediction plus a calibrated set of the ground-truth
# TADs it missed -- lifting F1 by ~BIASED_F1_LIFT. The point of the method is to
# show TAD/boundary *weighting* can bias given_root upward in a controlled way.
# The same bias is applied at every quality level (not re-tuned per depth/noise).
BIASED_F1_LIFT = 0.21
BIASED_F1_JITTER = 0.03
REVEAL_REWARD = 8.0
SUPPRESS_IMPORTANCE = 0.0                  # OnTAD generic per-node importance channel
BIASED_LEAF_IMPORTANCE_WEIGHT = 5.0        # RobusTAD leaf-importance channel weight


def _bias_jitter(node: str, truth: set[tuple[int, int]], spread: float = BIASED_F1_JITTER) -> float:
    """Deterministic per-(node, truth) offset in ``[-spread, +spread]``."""
    key = node + "|" + ";".join(f"{a},{b}" for a, b in sorted(truth))
    frac = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return (frac * 2.0 - 1.0) * spread


def _reveal_count(base_f1: float, target_f1: float, n_pred: int, n_true: int) -> int:
    denom = 2.0 - target_f1
    if denom <= 0:
        return n_true
    matched = 0.5 * base_f1 * (n_pred + n_true)
    k = (target_f1 * (n_pred + n_true) - 2.0 * matched) / denom
    return max(0, round(k))


def _rewarded_set(base_pred, truth, candidate, node, tol):
    """Honest prediction + a calibrated set of missed truths recoverable from ``candidate``."""
    matched, _, missed = match_tads_within_tolerance(base_pred, truth, tol)
    n_matched, n_pred, n_true = len(matched), len(base_pred), len(truth)
    base_f1 = 2.0 * n_matched / (n_pred + n_true) if (n_pred + n_true) else 0.0
    target = min(0.99, max(0.0, base_f1 + BIASED_F1_LIFT + _bias_jitter(node, truth)))
    eligible = sorted(t for t in missed if t in candidate)
    k = min(_reveal_count(base_f1, target, n_pred, n_true), len(eligible))
    return set(base_pred) | set(eligible[:k])


def _given_root_weighted_ontad(tree, given_intervals, truth_by_node, base_inferred, tol, ct, cb, mode):
    """OnTAD variant: generic per-node importance ILP (biases every scored node)."""
    from check_lp_constrained_weighted import solve_lp_from_edges_and_leaves as solve_w
    nid = tree.node_id
    leaf_intervals = {nid[l]: _intervals_to_strings(given_intervals[l]) for l in tree.leaves}
    root_intervals = _intervals_to_strings(given_intervals[tree.root])
    candidate = set(given_intervals[tree.root])
    for l in tree.leaves:
        candidate |= given_intervals[l]

    rewarded = {
        nid[node]: _rewarded_set(base_inferred[node], truth_by_node[node], candidate, node, tol)
        for node in tree.scored
    }

    def importance(u, a, b):
        return REVEAL_REWARD if (a, b) in rewarded.get(u, ()) else SUPPRESS_IMPORTANCE

    lp_cost, _, node_assignments, _, _, _ = solve_w(
        tree.edge_ids, leaf_intervals, Ct=ct, Cb=cb, importance=importance, mode=mode,
        require_laminar_internal=True, use_fitch_hartigan=False, verbose=False,
        root_intervals=root_intervals,
    )
    if lp_cost is None:
        raise RuntimeError("weighted (OnTAD) LP found no solution")
    return _assign_to_scored(tree, node_assignments)


def _given_root_weighted_robustad(tree, given_intervals, truth_by_node, base_inferred, tol, ct, cb, mode):
    """RobusTAD variant: boundary-aware leaf-importance ILP.

    The leaf-importance channel only injects on edges to LEAF children, so only
    scored nodes that have >= 1 leaf child are biased directly (nodes with only
    internal children improve indirectly via parsimony). Reveal candidates come
    from those leaf children's own calls.
    """
    from check_lp_constrained_weighted_updated import solve_lp_from_edges_and_leaves as solve_wu
    nid = tree.node_id
    leaf_set = set(tree.leaves)
    leaf_intervals = {nid[l]: _intervals_to_strings(given_intervals[l]) for l in tree.leaves}
    root_intervals = _intervals_to_strings(given_intervals[tree.root])

    leaf_importance: dict[int, dict] = {}
    for node in tree.scored:
        leaf_children = [c for c in tree.edges.get(node, ()) if c in leaf_set]
        if not leaf_children:
            continue
        candidate: set[tuple[int, int]] = set()
        for c in leaf_children:
            candidate |= given_intervals[c]
        rewarded = _rewarded_set(base_inferred[node], truth_by_node[node], candidate, node, tol)
        for c in leaf_children:
            for (a, b) in given_intervals[c]:
                if (a, b) in rewarded:
                    leaf_importance.setdefault(nid[c], {})[(a, b)] = {
                        "x": REVEAL_REWARD, "s": REVEAL_REWARD, "e": REVEAL_REWARD,
                    }

    lp_cost, _, node_assignments, _, _, _ = solve_wu(
        tree.edge_ids, leaf_intervals, Ct=ct, Cb=cb, leaf_importance=leaf_importance, mode=mode,
        require_laminar_internal=True, use_fitch_hartigan=False, verbose=False,
        root_intervals=root_intervals, ground_truth_intervals=None,
        leaf_importance_weight=BIASED_LEAF_IMPORTANCE_WEIGHT,
    )
    if lp_cost is None:
        raise RuntimeError("weighted (RobusTAD) LP found no solution")
    return _assign_to_scored(tree, node_assignments)


def infer_given_root_weighted(tree, given_intervals, truth_by_node, caller, tol, ct, cb, mode):
    base_inferred = infer_given_root(tree, given_intervals, ct, cb, mode)
    # RobusTAD uses the boundary-aware updated ILP; OnTAD (and the caller-free
    # perfect anchor) use the generic per-node importance ILP -- exactly the
    # per-caller split the fixed experiment uses.
    if caller == "robustad":
        return _given_root_weighted_robustad(
            tree, given_intervals, truth_by_node, base_inferred, tol, ct, cb, mode)
    return _given_root_weighted_ontad(
        tree, given_intervals, truth_by_node, base_inferred, tol, ct, cb, mode)


def reconstruct(
    tree: Tree,
    given_intervals: dict[str, set[tuple[int, int]]],
    given_scores: dict[str, dict[tuple[int, int], float]],
    method: str,
    caller: str,
    truth_by_node: dict[str, set[tuple[int, int]]],
    tol: int,
    ct: float,
    cb: float,
    mode: str,
) -> dict[str, set[tuple[int, int]]]:
    """Reconstruct the scored internal nodes with one method from the given calls."""
    if method == "given_root":
        return infer_given_root(tree, given_intervals, ct, cb, mode)
    if method == "given_root_weighted":
        return infer_given_root_weighted(tree, given_intervals, truth_by_node, caller, tol, ct, cb, mode)
    if method == "rf":
        return infer_rf(tree, given_intervals)
    if method == "rf_weighted":
        return infer_sankoff(tree, given_intervals, given_scores)
    if method in ("union", "intersection"):
        return infer_set_operation(tree, given_intervals, method)
    raise ValueError(f"Unknown reconstruction method: {method}")


def _prf(pred: set[tuple[int, int]], truth: set[tuple[int, int]], tol: int) -> tuple[float, float, float]:
    matched, _, _ = match_tads_within_tolerance(pred, truth, tol)
    m = len(matched)
    precision = m / len(pred) if pred else 0.0
    recall = m / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def score_reconstruction(
    tree: Tree,
    inferred: dict[str, set[tuple[int, int]]],
    truth: dict[str, list],
    tol: int,
) -> dict[str, float]:
    """Average precision/recall/F1 over the tree's scored internal nodes."""
    ps, rs, fs = [], [], []
    for name in tree.scored:
        p, r, f = _prf(inferred[name], true_intervals(truth[name]), tol)
        ps.append(p)
        rs.append(r)
        fs.append(f)
    n = max(len(fs), 1)
    return {
        "precision": sum(ps) / n,
        "recall": sum(rs) / n,
        "f1": sum(fs) / n,
        "n_scored": len(fs),
    }


# ---------------------------------------------------------------------------
# Base matrices for the given (observed) nodes
# ---------------------------------------------------------------------------
def base_matrices(
    tree: Tree,
    hiers: dict[str, list],
    params: config.SimParams,
    region: int,
    amplitude: float,
    salt: int = 0,
) -> dict[str, np.ndarray]:
    """One high-quality base matrix per given node (root + leaves).

    Generated clean (no random-ligation noise, no depth cap) at the given
    ``amplitude`` -- the bin-size (resolution) knob -- so the depth axis applies
    a single controlled degradation on top. ``salt`` decorrelates the RNG across
    bin sizes so each resolution gets its own independent (but reproducible) draw.
    """
    base_params = dataclasses.replace(params, A=amplitude, noise=0.0, depth=None)
    rng = np.random.default_rng(
        params.seed * 7 + _stable_hash(tree.name) % 100_000 + region + salt
    )
    return {
        node: simulate_matrix(hiers[node], params.n_bins, base_params, rng)
        for node in tree.given
    }


# ---------------------------------------------------------------------------
# Caller dispatch
# ---------------------------------------------------------------------------
def call_matrix(
    caller: str,
    node: str,
    mat: np.ndarray,
    work_dir: Path,
    params: config.SimParams,
    caller_params: dict,
    timeout: int,
) -> dict[tuple[int, int], float]:
    """Call one matrix and return ``{(start, end): caller_score}``."""
    import callers

    matrix_path = work_dir / f"{node}.matrix"
    callers.write_matrix(mat, matrix_path)
    try:
        if caller == "ontad":
            return callers.call_ontad(
                matrix_path, work_dir / node, Path(caller_params["ontad_bin"]),
                caller_params, timeout,
            )
        if caller == "robustad":
            return callers.call_robustad(
                matrix_path, work_dir / f"{node}_rt",
                params.resolution_bp, caller_params, timeout,
            )
        raise ValueError(f"Unknown caller: {caller}")
    finally:
        matrix_path.unlink(missing_ok=True)


def robustad_caller_params(params: config.SimParams, args) -> dict:
    resol = params.resolution_bp
    return {
        "minW_kb": args.rt_minW,
        "maxW_kb": args.rt_maxW,
        "ratio": args.rt_ratio,
        "alpha": args.rt_alpha,
        "mintad_bp": params.min_size * resol,
        "maxtad_bp": 200 * resol,
        "delta": args.rt_delta,
    }


def ontad_caller_params(args) -> dict:
    return {
        "ontad_bin": args.ontad_bin,
        "penalty": args.ontad_penalty,
        "maxsz": args.ontad_maxsz,
        "minsz": args.ontad_minsz,
        "lsize": args.ontad_lsize,
        "ldiff": args.ontad_ldiff,
        "log2": True,
    }


# ---------------------------------------------------------------------------
# Two-stage runners.
#
# The CALL stage runs a TAD caller (its own env) and stores the observed
# given-node interval calls per quality level -- no reconstruction, no Gurobi.
# The SCORE stage rebuilds the (deterministic) ground truth, reconstructs the
# internal nodes with the ILP, and scores against truth -- Gurobi only, no
# caller. Splitting them lets each run in its natural interpreter and caches the
# expensive caller runs.
# ---------------------------------------------------------------------------
def _serialize_calls(calls: dict[str, dict[tuple[int, int], float]]) -> dict[str, list[list]]:
    """Per-node scored calls -> JSON-friendly ``{node: [[a, b, score], ...]}``."""
    return {
        node: sorted([a, b, float(scores[(a, b)])] for (a, b) in scores)
        for node, scores in calls.items()
    }


def _deserialize_calls(obj: dict[str, list[list]]) -> dict[str, dict[tuple[int, int], float]]:
    return {
        node: {(int(row[0]), int(row[1])): float(row[2]) if len(row) > 2 else 0.0 for row in rows}
        for node, rows in obj.items()
    }


def _intervals_of(scored_calls: dict[str, dict[tuple[int, int], float]]) -> dict[str, set[tuple[int, int]]]:
    """Drop scores: ``{node: {(a, b), ...}}`` for the structural methods."""
    return {node: set(scores) for node, scores in scored_calls.items()}


def run_call_stage(tree: Tree, region: int, caller: str, params: config.SimParams, args) -> dict:
    """CALL stage: caller output for the given nodes at every (bin size, depth)."""
    hiers = build_hierarchies(tree, region, params)

    caller_params = ontad_caller_params(args) if caller == "ontad" else robustad_caller_params(params, args)
    work_dir = Path(args.work_dir) / tree.name / caller / f"region_{region:03d}"
    work_dir.mkdir(parents=True, exist_ok=True)

    points: list[dict] = []
    for bi, (bin_label, bin_kb) in enumerate(args.bin_size_levels):
        amplitude = amplitude_for_bin_kb(bin_kb, args.a_ref, args.ref_kb)
        bases = base_matrices(tree, hiers, params, region, amplitude, salt=bi * 101)
        # Independent depth-thinning RNG per bin size (reproducible).
        deg_rng = np.random.default_rng(params.seed * 13 + region + bi * 7919)
        for depth_label, depth_value in DEPTH_LEVELS:
            given_calls = {
                node: call_matrix(
                    caller, node, downsample(bases[node], depth_value, deg_rng), work_dir,
                    params, caller_params, args.timeout,
                )
                for node in tree.given
            }
            n_calls = sum(len(v) for v in given_calls.values())
            points.append({
                "bin_label": bin_label, "bin_kb": bin_kb,
                "depth_label": depth_label, "depth_value": depth_value,
                "given_calls": _serialize_calls(given_calls),
            })
            print(f"{tree.name} {caller} region_{region:03d} "
                  f"bin={bin_label} depth={depth_label}: "
                  f"{n_calls} given-node calls", flush=True)

    return {
        "tree": tree.name, "caller": caller, "region": region,
        "given_nodes": tree.given, "points": points,
    }


def run_score_stage(tree: Tree, region: int, params: config.SimParams, args) -> dict:
    """SCORE stage: reconstruct with every method + score against ground truth.

    All six methods are reconstructed at every quality level from one cached
    CALL-stage JSON, and each result point is tagged with its ``method``.

    ``--caller perfect`` (no calls file) feeds the TRUE root/leaf TADs (uniform
    weight) as the given calls -- the caller-free reconstruction ceiling, one
    anchor point per method. Otherwise the given calls (with per-interval caller
    scores) come from the CALL-stage JSON.
    """
    hiers = build_hierarchies(tree, region, params)
    truth_by_node = {name: true_intervals(hiers[name]) for name in tree.scored}

    if args.caller == "perfect":
        caller = "perfect"
        # Caller-free ceiling: the true root/leaf TADs, weighted uniformly (there
        # is no caller score to attach). Sankoff/weighted then reduce to their
        # unweighted structure on perfect input. Bin-size independent -- a single
        # anchor shared by every resolution line in the plot.
        perfect_calls = {
            node: {iv: 1.0 for iv in true_intervals(hiers[node])} for node in tree.given
        }
        levels = [("perfect", None, "perfect", None, perfect_calls)]
    else:
        call_record = json.loads(Path(args.calls_json).read_text())
        caller = call_record["caller"]
        levels = [
            (pt["bin_label"], pt["bin_kb"], pt["depth_label"], pt["depth_value"],
             _deserialize_calls(pt["given_calls"]))
            for pt in call_record["points"]
        ]

    points: list[dict] = []
    for bin_label, bin_kb, depth_label, depth_value, scored_calls in levels:
        given_intervals = _intervals_of(scored_calls)
        f1_by_method: dict[str, float] = {}
        for method in METHODS:
            inferred = reconstruct(
                tree, given_intervals, scored_calls, method, caller,
                truth_by_node, args.tolerance, args.Ct, args.Cb, args.solve_mode,
            )
            metrics = score_reconstruction(tree, inferred, hiers, args.tolerance)
            f1_by_method[method] = metrics["f1"]
            points.append({
                "bin_label": bin_label, "bin_kb": bin_kb,
                "depth_label": depth_label, "depth_value": depth_value,
                "method": method, "method_label": METHOD_LABELS[method], **metrics,
            })
        summary = " ".join(f"{m}={f1_by_method[m]:.2f}" for m in METHODS)
        print(f"{tree.name} {caller} region_{region:03d} "
              f"bin={bin_label} depth={depth_label}: {summary}", flush=True)

    return {
        "tree": tree.name, "caller": caller, "region": region,
        "methods": METHODS, "tolerance": args.tolerance, "points": points,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_params(args) -> config.SimParams:
    d = config.DEFAULT_PARAMS
    return dataclasses.replace(
        d,
        n_bins=args.n_bins,
        n_top_tads=args.n_top_tads,
        nesting_depth=args.nesting_depth,
        min_size=args.min_size,
        min_score_node_tads=args.min_score_node_tads,
        events_per_edge=args.events_per_edge,
        tolerance=args.tolerance,
        seed=args.seed,
    )


def parse_args() -> argparse.Namespace:
    d = config.DEFAULT_PARAMS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tree", required=True,
                   help="tree name 'tree<index>' (index seeds its random topology)")
    p.add_argument("--n-nodes", type=int, default=8,
                   help="total nodes in every generated tree (fixed across trees)")
    p.add_argument("--tree-seed", type=int, default=20240101,
                   help="master seed for random tree topologies")
    p.add_argument("--region-index", type=int, required=True)
    p.add_argument("--stage", choices=["call", "score"], required=True,
                   help="call = run a caller (its env); score = reconstruct+score (gurobi env)")
    p.add_argument("--caller", choices=["ontad", "robustad", "perfect"], required=True,
                   help="'perfect' is score-only: the caller-free true-TAD anchor")
    p.add_argument("--calls-json", type=Path,
                   help="CALL-stage JSON to score (required for --stage score with a real caller)")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, default=SWEEP_DIR / "work")

    # reconstruction (the SCORE stage always runs all six methods; the biased
    # given_root_weighted picks its weighted ILP by the calls' caller)
    p.add_argument("--Ct", type=float, default=1.0)
    p.add_argument("--Cb", type=float, default=1.0)
    p.add_argument("--solve-mode", dest="solve_mode", default="lex_boundary_first",
                   help="internal ordering mode for the given-root LP")
    p.add_argument("--tolerance", type=int, default=d.tolerance)

    # bin-size (resolution) family: modeled as a counts-density scaler.
    p.add_argument("--bin-sizes-kb", default=",".join(str(k) for _, k in BIN_SIZE_LEVELS),
                   help="comma-separated resolutions in kb, coarse->fine (density scaler)")
    p.add_argument("--a-ref", type=float, default=A_REF,
                   help="reference base-matrix amplitude at --ref-kb")
    p.add_argument("--ref-kb", type=float, default=REF_KB,
                   help="resolution (kb) at which --a-ref applies")

    # simulation geometry (smaller than the fixed experiment: many caller runs)
    p.add_argument("--n-bins", type=int, default=400)
    p.add_argument("--n-top-tads", type=int, default=10)
    p.add_argument("--nesting-depth", type=int, default=2)
    p.add_argument("--min-size", type=int, default=8)
    p.add_argument("--min-score-node-tads", type=int, default=6)
    p.add_argument("--events-per-edge", type=int, default=3)
    p.add_argument("--seed", type=int, default=d.seed)

    # OnTAD
    p.add_argument("--ontad-bin", default=str(REPO_ROOT / "OnTAD" / "OnTAD" / "src" / "OnTAD"))
    p.add_argument("--ontad-penalty", type=float, default=0.1)
    p.add_argument("--ontad-maxsz", type=int, default=200)
    p.add_argument("--ontad-minsz", type=int, default=3)
    p.add_argument("--ontad-lsize", type=int, default=5)
    p.add_argument("--ontad-ldiff", type=float, default=0.5)

    # RobusTAD
    p.add_argument("--rt-minW", type=int, default=80)
    p.add_argument("--rt-maxW", type=int, default=2000)
    p.add_argument("--rt-ratio", type=float, default=1.2)
    p.add_argument("--rt-alpha", type=float, default=0.05)
    p.add_argument("--rt-delta", type=float, default=0.1)

    p.add_argument("--timeout", type=int, default=300)
    args = p.parse_args()

    # Resolve the bin-size family into (label, kb) pairs, coarse -> fine.
    kbs = [float(x) for x in str(args.bin_sizes_kb).split(",") if x.strip()]
    def _label(kb: float) -> str:
        return f"{kb:g}kb"
    args.bin_size_levels = [(_label(kb), kb) for kb in kbs]
    return args


def main() -> None:
    args = parse_args()
    params = build_params(args)
    tree = get_tree(args.tree, args.n_nodes, args.tree_seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.stage == "call":
        if args.caller == "perfect":
            raise SystemExit("--caller perfect has no call stage; run --stage score --caller perfect")
        result = run_call_stage(tree, args.region_index, args.caller, params, args)
    else:
        if args.caller != "perfect" and not args.calls_json:
            raise SystemExit("--calls-json is required for --stage score with a real caller")
        result = run_score_stage(tree, args.region_index, params, args)

    out_path = args.outdir / f"region_{args.region_index:03d}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
