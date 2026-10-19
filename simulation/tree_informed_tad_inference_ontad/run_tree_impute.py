#!/usr/bin/env python3
"""Tree-informed TAD inference at the LEAVES of a phylogeny.

The question this experiment answers
------------------------------------
A TAD caller run on one leaf's noisy Hi-C misses real TADs. But the leaf has
relatives: if its siblings and cousins all call a domain there, the tree itself
is evidence that the leaf has it too and the caller simply missed it. Does
imputing those TADs back into the leaf improve precision/recall/F1 against the
leaf's true TADs, and does the improvement grow as the Hi-C gets noisier?

Unlike ``../data_quality_sweep`` (which reconstructs the *hidden internal* nodes
and treats leaves as fixed inputs), here **every leaf is both observed and
scored**. No node is given for free: the root and internal nodes are latent,
inferred only as a means of sharing information between leaves.

Pipeline for one (tree, region)
-------------------------------
1. Evolve hidden ground-truth hierarchies top-down from the root, one per node.
2. Simulate a Hi-C matrix for each LEAF, degrade it (binomial down-sampling to a
   fraction of its counts -- 1x, 0.1x, 0.03x ...), and run OnTAD on it.
3. Score the raw OnTAD calls against that leaf's TRUE TADs -> the no-tree
   baseline.
4. Snap the leaves' jittered boundaries onto consensus positions, then re-solve
   with the tree: ``check_lp_constrained.solve_lp_from_edges_and_leaves`` with
   ``impute=True``, which lets a leaf gain intervals it did not call, and read
   the leaf assignments back out.
5. Score the imputed leaf sets against the same truth.

Why the snap is load-bearing
----------------------------
The ILP counts sibling support by EXACT interval equality, but scoring matches
within +/- tolerance. Under noise the boundaries jitter, so relatives that called
the same domain a bin apart looked like unrelated intervals to the solver while
the metric would have called them identical -- it was discarding about a third of
the evidence, and of the reachable ceiling. Snapping to consensus boundaries
first turns a headline gain of +0.028 into +0.075 at 0.03x, and stops the method
from actively HURTING on clean data (-0.046 at 1.0x). See `method_impute_snapped`
and the README.

Why ``impute_cost`` must be > 0
------------------------------
The parsimony objective charges only for *differences* along edges, so putting
the same interval on a parent and on all of its children costs exactly zero:
free imputation is degenerate and floods every leaf (measured: leaf precision
0.37 -> 0.07, F1 0.48 -> 0.13). ``impute_cost`` prices each gained interval, so a
leaf only gains a TAD when the parsimony saving on its parent edge outweighs the
price. It reads as a sibling-support threshold -- see the parameter docs in
``check_lp_constrained.py``.

Methods scored at every noise level (all evaluated on the LEAVES)
-----------------------------------------------------------------
  ontad          raw OnTAD calls -- the no-tree baseline
  snap_c<K>      imputation at ``impute_cost=K`` on consensus boundaries, with
                 the candidate universe restricted to intervals a leaf called

Stages
------
--stage call  : run OnTAD on every leaf at every depth, cache the calls.
--stage score : solve the ILP from the cached calls and score against truth.
Split so the caller and the Gurobi solve can run in their own interpreters, and
so the caller runs are cached across solver retuning. The SCORE stage rebuilds
the ground truth deterministically, so both stages MUST get identical simulation
parameters (the Snakefile's COMMON string).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
SIM_ROOT = EXP_DIR.parent                    # simulation/ (shared generation)
REPO_ROOT = SIM_ROOT.parent                  # repo root (shared solver)
SWEEP_DIR = SIM_ROOT / "data_quality_sweep"  # reused: trees.py, callers.py
for p in (str(REPO_ROOT), str(SIM_ROOT), str(SWEEP_DIR), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from evolve import evolve
from hierarchy import all_domains, generate_root, iter_domains
from simulate_matrix import simulate_matrix
from tad_utilities import match_tads_within_tolerance

from trees import Tree, get_tree

# The ILP pulls in gurobipy; the CALL stage never solves, so import it lazily.


# ---------------------------------------------------------------------------
# Noise ladder.
#
# "Noise" is sequencing depth: the leaf's clean matrix is thinned binomially to
# a fraction of its counts. Ordered cleanest -> noisiest.
#
# The absolute count density is set by AMPLITUDE, following the bin-size scaler
# the data-quality sweep established: A = A_REF * (bin_kb / REF_KB)^2 with
# A_REF = 2000 at REF_KB = 40. The default 7.8 is the 2.5 kb regime -- count-poor
# enough that down-sampling genuinely degrades OnTAD. (At 40 kb the matrices are
# so count-rich that OnTAD's F1 barely moves across the whole ladder, so there is
# nothing for imputation to repair.)
# ---------------------------------------------------------------------------
# New rungs are APPENDED, never inserted: the degradation RNG is consumed in
# ladder order, so appending leaves every existing rung's draws bit-identical and
# the cached results stay comparable.
DEPTH_LEVELS: list[tuple[str, float]] = [
    ("1.0x", 1.0),
    ("0.3x", 0.3),
    ("0.1x", 0.1),
    ("0.03x", 0.03),
    ("0.01x", 0.01),
    ("0.003x", 0.003),
]
DEFAULT_AMPLITUDE = 7.8      # 2.5 kb count density, per the sweep's convention

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
def _stable_hash(text: str) -> int:
    """Process-stable hash (Python's built-in hash() is salted per process)."""
    return zlib.crc32(text.encode())


def build_hierarchies(tree: Tree, region: int, params: config.SimParams) -> dict[str, list]:
    """Evolve every node's hidden hierarchy top-down from the root.

    Regenerates until every LEAF carries at least ``min_score_node_tads``
    domains -- the leaves are what gets scored here, so a leaf with almost no
    TADs would make its F1 meaningless.
    """
    base_seed = params.seed * 1_000_003 + _stable_hash(tree.name) % 100_000 + region
    counts: dict[str, int] = {}
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

        counts = {n: len(all_domains(hiers[n])) for n in tree.leaves}
        if not counts or min(counts.values()) >= params.min_score_node_tads:
            return hiers

    raise RuntimeError(
        f"{tree.name}/region_{region:03d}: could not evolve >= "
        f"{params.min_score_node_tads} TADs at every leaf in "
        f"{params.max_generation_attempts} attempts (last: {counts})"
    )


def true_intervals(hier: list) -> set[tuple[int, int]]:
    """True TAD set in the truth convention: 0-indexed [s, e) -> (s + 1, e)."""
    return {(d.start + 1, d.end) for d in iter_domains(hier)}


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------
def downsample(mat: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Thin counts to a fraction of coverage (binomial per contact)."""
    if frac >= 1.0:
        return mat.copy()
    n = mat.shape[0]
    iu = np.triu_indices(n)
    thinned = rng.binomial(mat[iu].astype(np.int64), frac)
    out = np.zeros((n, n), dtype=np.int64)
    out[iu] = thinned
    out = np.triu(out)
    return out + out.T - np.diag(np.diag(out))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _prf(pred: set[tuple[int, int]], truth: set[tuple[int, int]], tol: int) -> tuple[float, float, float]:
    matched, _, _ = match_tads_within_tolerance(pred, truth, tol)
    m = len(matched)
    precision = m / len(pred) if pred else 0.0
    recall = m / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def score_leaves(
    tree: Tree,
    pred: dict[str, set[tuple[int, int]]],
    truth_by_leaf: dict[str, set[tuple[int, int]]],
    tol: int,
) -> dict[str, float]:
    """Average precision/recall/F1 over the tree's leaves."""
    ps, rs, fs = [], [], []
    for leaf in tree.leaves:
        p, r, f = _prf(pred.get(leaf, set()), truth_by_leaf[leaf], tol)
        ps.append(p)
        rs.append(r)
        fs.append(f)
    n = max(len(fs), 1)
    return {
        "precision": sum(ps) / n,
        "recall": sum(rs) / n,
        "f1": sum(fs) / n,
        "n_leaves": len(fs),
        "n_pred": sum(len(pred.get(leaf, set())) for leaf in tree.leaves) / n,
    }


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------
def _within(iv: tuple[int, int], pool, tol: int) -> bool:
    return any(abs(iv[0] - c[0]) <= tol and abs(iv[1] - c[1]) <= tol for c in pool)


def method_raw(tree: Tree, calls: dict[str, set]) -> dict[str, set]:
    """The caller's own leaf output, untouched: the no-tree baseline."""
    return {leaf: set(calls[leaf]) for leaf in tree.leaves}


# ---------------------------------------------------------------------------
# Boundary snapping
#
# The ILP counts sibling support by EXACT interval equality: `dx[u,v,a,b]`
# compares (a,b) literally, so leaf A's (100,150) and leaf B's (101,151) are two
# unrelated intervals with no shared support. Scoring, by contrast, matches
# within +/- tolerance and calls those the same TAD. Under noise the boundaries
# jitter, so the solver throws away evidence the metric would credit. Measured
# over all 200 regions at 0.03x: a leaf's calls that some other leaf corroborates
# is 0.55 by exact match but 0.74 within tolerance, and the share of a leaf's
# MISSED truth TADs that another leaf called is 0.32 exact vs 0.48 within
# tolerance -- i.e. exact matching hides a third of the evidence and a third of
# the reachable ceiling.
#
# Snapping puts jittered copies of the same boundary onto one consensus
# coordinate before the ILP runs, which restores that evidence and, as a side
# effect, shrinks the endpoint union that makes the model explode.
# ---------------------------------------------------------------------------
def snap_map(positions: dict[int, int], tol: int) -> dict[int, int]:
    """Map jittered boundary positions onto consensus positions.

    ``positions`` is position -> support (how many leaves use it). Greedy
    mode-seeking, NOT single-linkage: repeatedly take the best-supported
    unassigned position as a centre and absorb everything within ``tol`` of it.
    Single-linkage would chain (100-102-104 merging into one cluster whose mean
    is 2 away from both ends), and a boundary displaced by more than the scoring
    tolerance would break a match that the raw call had. Mode-seeking bounds
    every displacement by ``tol`` and snaps to a real, well-supported position
    rather than an invented mean.
    """
    remaining = dict(positions)
    out: dict[int, int] = {}
    while remaining:
        centre = max(remaining, key=lambda p: (remaining[p], -p))
        for p in [q for q in remaining if abs(q - centre) <= tol]:
            out[p] = centre
            del remaining[p]
    return out


def snap_calls(calls: dict[str, set], leaves: list[str], tol: int):
    """Rewrite every leaf's calls onto consensus boundaries.

    Starts and ends are snapped in separate coordinate spaces -- they are
    different roles, and pooling them would let a domain start absorb a
    neighbouring domain end.

    Returns ``(snapped, back)`` where ``back[leaf][snapped_iv]`` is an original
    interval the leaf actually called. The back-map is what keeps the method
    strictly additive over the baseline: a leaf's own calls are reported in their
    ORIGINAL coordinates, so snapping can never degrade a match the raw caller
    already had, and only genuinely new (imputed) intervals are reported at their
    consensus position.
    """
    start_sup: dict[int, int] = {}
    end_sup: dict[int, int] = {}
    for leaf in leaves:
        for (a, b) in calls[leaf]:
            start_sup[a] = start_sup.get(a, 0) + 1
            end_sup[b] = end_sup.get(b, 0) + 1
    smap = snap_map(start_sup, tol)
    emap = snap_map(end_sup, tol)

    snapped: dict[str, set] = {}
    back: dict[str, dict] = {}
    for leaf in leaves:
        s, bk = set(), {}
        for (a, b) in sorted(calls[leaf]):
            iv = (smap[a], emap[b])
            if iv[0] >= iv[1]:
                continue          # snapping collapsed it; keep the original only
            s.add(iv)
            bk.setdefault(iv, (a, b))
        snapped[leaf] = s
        back[leaf] = bk
    return snapped, back


def method_impute_snapped(tree: Tree, calls: dict[str, set], impute_cost: float,
                          ct: float, cb: float, mode: str, tol: int):
    """Imputation on snapped coordinates with a restricted candidate universe.

    Two changes over ``method_impute``:

    1. **Snap** leaf boundaries to consensus positions first, so relatives that
       called the same domain a bin apart now agree exactly and the parsimony
       objective can see it.
    2. **Restrict candidates** to intervals some leaf actually called, instead of
       every (start, end) pair over the boundary union. The all-pairs universe is
       what makes the model grow as (endpoints)^2 and forces the ladder to stop
       at 0.03x; an interval no leaf called can only ever cost the objective, so
       dropping it is close to free and makes the model linear in distinct calls.

    The result is strictly additive over the raw calls: every original call is
    returned unchanged, plus the imputed intervals.
    """
    from check_lp_constrained import solve_lp_from_edges_and_leaves

    nid = tree.node_id
    snapped, back = snap_calls(calls, tree.leaves, tol)
    candidates = sorted(set().union(*(snapped[l] for l in tree.leaves))) if tree.leaves else []
    leaf_intervals = {nid[l]: {f"[{a},{b}]" for (a, b) in snapped[l]} for l in tree.leaves}

    lp_cost, _, assignments, _, _, universe = solve_lp_from_edges_and_leaves(
        tree.edge_ids, leaf_intervals, Ct=ct, Cb=cb, mode=mode,
        require_laminar_internal=True, use_fitch_hartigan=False, verbose=False,
        impute=True, impute_cost=impute_cost, root_intervals=None,
        candidate_intervals=candidates,
    )
    if lp_cost is None:
        return None, len(universe)

    out = {}
    for leaf in tree.leaves:
        keep = set(calls[leaf])                     # originals, coordinates intact
        for iv in assignments[nid[leaf]]:
            iv = tuple(iv)
            if iv not in back[leaf]:                # genuinely new -> consensus coords
                keep.add(iv)
        out[leaf] = keep
    return out, len(universe)


def snap_method_name(cost: float) -> str:
    return f"snap_c{cost:g}"


def method_label(name: str, headline_cost: float) -> str:
    if name == "ontad":
        return "OnTAD (no tree)"
    cost = name.removeprefix("snap_c")
    tag = " [headline]" if float(cost) == headline_cost else ""
    return f"tree-informed imputation (cost={cost}){tag}"


# ---------------------------------------------------------------------------
# CALL stage
# ---------------------------------------------------------------------------
def run_call_stage(tree: Tree, region: int, params: config.SimParams, args) -> dict:
    """Run OnTAD on every leaf at every depth level and cache the calls."""
    import callers

    hiers = build_hierarchies(tree, region, params)
    caller_params = {
        "ontad_bin": args.ontad_bin,
        "penalty": args.ontad_penalty,
        "maxsz": args.ontad_maxsz,
        "minsz": args.ontad_minsz,
        "lsize": args.ontad_lsize,
        "ldiff": args.ontad_ldiff,
        "log2": True,
    }
    work_dir = Path(args.work_dir) / tree.name / f"region_{region:03d}"
    work_dir.mkdir(parents=True, exist_ok=True)

    base_params = dataclasses.replace(params, A=args.amplitude, noise=0.0, depth=None)
    sim_rng = np.random.default_rng(
        params.seed * 7 + _stable_hash(tree.name) % 100_000 + region
    )
    bases = {
        leaf: simulate_matrix(hiers[leaf], params.n_bins, base_params, sim_rng)
        for leaf in tree.leaves
    }
    deg_rng = np.random.default_rng(params.seed * 13 + region + 7919)

    points = []
    for depth_label, depth_value in DEPTH_LEVELS:
        leaf_calls = {}
        for leaf in tree.leaves:
            mat = downsample(bases[leaf], depth_value, deg_rng)
            matrix_path = work_dir / f"{leaf}.matrix"
            callers.write_matrix(mat, matrix_path)
            try:
                leaf_calls[leaf] = callers.call_ontad(
                    matrix_path, work_dir / leaf, Path(args.ontad_bin),
                    caller_params, args.timeout,
                )
            finally:
                matrix_path.unlink(missing_ok=True)
        n_calls = sum(len(v) for v in leaf_calls.values())
        points.append({
            "depth_label": depth_label,
            "depth_value": depth_value,
            "leaf_calls": {
                leaf: sorted([a, b, float(sc[(a, b)])] for (a, b) in sc)
                for leaf, sc in leaf_calls.items()
            },
        })
        print(f"{tree.name} region_{region:03d} depth={depth_label}: "
              f"{n_calls} leaf calls over {len(tree.leaves)} leaves", flush=True)

    return {
        "tree": tree.name, "region": region, "caller": "ontad",
        "leaves": tree.leaves, "amplitude": args.amplitude, "points": points,
    }


# ---------------------------------------------------------------------------
# SCORE stage
# ---------------------------------------------------------------------------
def run_score_stage(tree: Tree, region: int, params: config.SimParams, args) -> dict:
    """Solve + score every method at every noise level from the cached calls."""
    hiers = build_hierarchies(tree, region, params)
    truth_by_leaf = {leaf: true_intervals(hiers[leaf]) for leaf in tree.leaves}

    record = json.loads(Path(args.calls_json).read_text())
    methods = ["ontad"] + [snap_method_name(c) for c in args.snap_costs]

    points = []
    for pt in record["points"]:
        calls = {
            leaf: {(int(r[0]), int(r[1])) for r in rows}
            for leaf, rows in pt["leaf_calls"].items()
        }
        summary = []
        for name in methods:
            t0 = time.time()
            n_candidates = None
            status = "ok"
            if name == "ontad":
                pred = method_raw(tree, calls)
            else:
                cost = float(name.removeprefix("snap_c"))
                pred, n_candidates = method_impute_snapped(
                    tree, calls, cost, args.Ct, args.Cb, args.solve_mode,
                    args.tolerance)
                if pred is None:
                    # Recorded, not silently dropped: a point that fell back to
                    # raw would be indistinguishable from "imputation found
                    # nothing to add".
                    print(f"  WARNING {name} @ {pt['depth_label']}: LP found no "
                          f"solution; falling back to raw calls", flush=True)
                    status = "no-solution"
                    pred = method_raw(tree, calls)

            metrics = score_leaves(tree, pred, truth_by_leaf, args.tolerance)
            added = sum(len(pred[l] - calls[l]) for l in tree.leaves) / max(len(tree.leaves), 1)
            points.append({
                "depth_label": pt["depth_label"], "depth_value": pt["depth_value"],
                "method": name, "method_label": method_label(name, args.headline_cost),
                "imputed_per_leaf": added, "n_candidates": n_candidates,
                "status": status,
                "solve_seconds": round(time.time() - t0, 2),
                **metrics,
            })
            summary.append(f"{name}={metrics['f1']:.3f}")
        print(f"{tree.name} region_{region:03d} depth={pt['depth_label']}: "
              + " ".join(summary), flush=True)

    return {
        "tree": tree.name, "region": region, "caller": "ontad",
        "methods": methods, "headline_cost": args.headline_cost,
        "tolerance": args.tolerance, "n_leaves": len(tree.leaves),
        "points": points,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_params(args) -> config.SimParams:
    return dataclasses.replace(
        config.DEFAULT_PARAMS,
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
    p.add_argument("--tree", required=True, help="tree name 'tree<index>'")
    p.add_argument("--n-nodes", type=int, default=16,
                   help="total nodes per generated tree (more nodes => more leaves => more sibling support)")
    p.add_argument("--tree-seed", type=int, default=20240101)
    p.add_argument("--region-index", type=int, required=True)
    p.add_argument("--stage", choices=["call", "score"], required=True)
    p.add_argument("--calls-json", type=Path,
                   help="CALL-stage JSON to score (required for --stage score)")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, default=EXP_DIR / "work")

    # imputation
    p.add_argument("--snap-costs", default="0.5,1.0,1.5,3.0",
                   help="comma-separated impute_cost values to score")
    p.add_argument("--headline-cost", type=float, default=1.0,
                   help="which impute_cost the headline figure plots")
    p.add_argument("--Ct", type=float, default=1.0)
    p.add_argument("--Cb", type=float, default=1.0)
    p.add_argument("--solve-mode", dest="solve_mode", default="convex")
    p.add_argument("--tolerance", type=int, default=d.tolerance)

    # simulation
    p.add_argument("--amplitude", type=float, default=DEFAULT_AMPLITUDE,
                   help="base contact amplitude (count density); 7.8 ~ the 2.5 kb regime")
    p.add_argument("--n-bins", type=int, default=500)
    p.add_argument("--n-top-tads", type=int, default=10)
    p.add_argument("--nesting-depth", type=int, default=2)
    p.add_argument("--min-size", type=int, default=8)
    p.add_argument("--min-score-node-tads", type=int, default=6)
    p.add_argument("--events-per-edge", type=int, default=1,
                   help="evolution budget per edge; small => relatives share TADs => imputable")
    p.add_argument("--seed", type=int, default=d.seed)

    # OnTAD
    p.add_argument("--ontad-bin", default=str(REPO_ROOT / "OnTAD" / "OnTAD" / "src" / "OnTAD"))
    p.add_argument("--ontad-penalty", type=float, default=0.1)
    p.add_argument("--ontad-maxsz", type=int, default=200)
    p.add_argument("--ontad-minsz", type=int, default=3)
    p.add_argument("--ontad-lsize", type=int, default=5)
    p.add_argument("--ontad-ldiff", type=float, default=0.5)
    p.add_argument("--timeout", type=int, default=300)

    args = p.parse_args()
    args.snap_costs = [float(x) for x in str(args.snap_costs).split(",") if x.strip()]
    return args


def main() -> None:
    args = parse_args()
    params = build_params(args)
    tree = get_tree(args.tree, args.n_nodes, args.tree_seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.stage == "call":
        result = run_call_stage(tree, args.region_index, params, args)
    else:
        if not args.calls_json:
            raise SystemExit("--calls-json is required for --stage score")
        result = run_score_stage(tree, args.region_index, params, args)

    out_path = args.outdir / f"region_{args.region_index:03d}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
