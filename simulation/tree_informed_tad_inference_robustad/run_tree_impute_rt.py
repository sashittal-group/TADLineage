#!/usr/bin/env python3
"""Tree-informed TAD inference at the LEAVES, with RobusTAD and the WEIGHTED ILP.

This is the RobusTAD arm of ``../tree_informed_tad_inference_ontad`` -- the "next step"
that experiment's README asks for. The claim under test is identical:

    A TAD caller run on one leaf's noisy Hi-C misses real domains. The leaf has
    relatives; if they call a domain in the same place, the phylogeny is evidence
    the leaf has it too. Imputing those TADs back should raise the leaf's F1, and
    the gain should grow as the Hi-C degrades.

What changes is *how much the solver knows about each call*. OnTAD gives one
number per called TAD, so the OnTAD arm can only count siblings: a flat
``impute_cost`` that reads as "impute iff at least this fraction of relatives
called it". RobusTAD gives two things OnTAD does not:

  1. a per-TAD score for every call, and
  2. a per-BIN left/right boundary score for the WHOLE region -- including the
     bins where it did not assemble a TAD.

(2) is the interesting one. It is a graded read-out of a leaf's own evidence at
a position it never called, which is exactly the situation imputation is in. So
the weighted method uses all three signals, via
``check_lp_constrained_weighted_updated.solve_lp_from_edges_and_leaves``:

  * ``leaf_importance = {x: TADscore, s: leftScore[a], e: rightScore[b]}``
    -- a confident leaf call pulls its parent toward carrying the domain, so
    strong calls propagate to relatives and weak ones do not.
  * ``impute_cost`` as a per-``(leaf, interval)`` dict -- the price a leaf pays
    to gain (a, b) is discounted by ITS OWN boundary scores at a and b. A leaf
    whose data already hints at a boundary needs less corroboration from its
    relatives; a leaf with nothing there still has to be outvoted.
  * consensus boundary snapping weighted by boundary score rather than by a raw
    count of leaves.

Methods scored at every noise level (all evaluated on the LEAVES)
-----------------------------------------------------------------
  robustad        raw RobusTAD calls -- the no-tree baseline
  snap_c<K>       flat-cost imputation, no scores: the OnTAD-arm method run on
                  RobusTAD calls. The control that isolates what weighting buys.
  wsnap_c<K>      the weighted method above, at ``impute_cost=K``
  wimp_c<K>       ablation: leaf_importance only (flat price, unweighted snap)
  wpri_c<K>       ablation: discounted price only (no leaf_importance)

Stages
------
--stage call  : run RobusTAD on every leaf at every depth, cache calls + scores.
--stage score : solve the ILP from the cached calls and score against truth.
Split because RobusTAD needs a cooler-capable interpreter and the solve needs
gurobipy. The SCORE stage rebuilds the ground truth deterministically, so both
stages MUST get identical simulation parameters (the Snakefile's COMMON string).

The hierarchies, base matrices and down-sampling draws are produced by the exact
same code and seeds as the OnTAD arm, so a leaf's truth is bit-identical across
the two experiments and the baselines are directly comparable.
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
ONTAD_ARM = SIM_ROOT / "tree_informed_tad_inference_ontad"   # reused: the OnTAD arm
RT_DIR = SIM_ROOT / "using_robustad"         # reused: matrix->cooler + RobusTAD calls
for p in (str(REPO_ROOT), str(SIM_ROOT), str(SWEEP_DIR), str(ONTAD_ARM), str(RT_DIR), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from evolve import evolve
from hierarchy import all_domains, generate_root, iter_domains
from simulate_matrix import simulate_matrix
from tad_utilities import match_tads_within_tolerance

from trees import Tree, get_tree

# The ILP pulls in gurobipy and the caller pulls in cooler; neither stage needs
# the other's dependency, so both are imported lazily.


# ---------------------------------------------------------------------------
# Noise ladder. Identical to the OnTAD arm, and consumed in the same order, so
# a leaf's degraded matrix at a given rung is the same matrix in both arms.
# New rungs are APPENDED, never inserted.
# ---------------------------------------------------------------------------
DEPTH_LEVELS: list[tuple[str, float]] = [
    ("1.0x", 1.0),
    ("0.3x", 0.3),
    ("0.1x", 0.1),
    ("0.03x", 0.03),
    ("0.01x", 0.01),
    ("0.003x", 0.003),
]
DEFAULT_AMPLITUDE = 7.8      # 2.5 kb count density, per the sweep's convention

RAW_METHOD = "robustad"

# Boundary scores below this are dropped from the cache. RobusTAD assigns a
# non-zero score to ~95% of bins, and the vast majority are floor noise that
# would only bloat the JSON: the discount they buy is under 1% of the price.
BOUNDARY_SCORE_FLOOR = 0.02


# ---------------------------------------------------------------------------
# Ground truth (identical to the OnTAD arm)
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
# RobusTAD score bookkeeping
#
# Coordinates. RobusTAD's boundary table is keyed by 0-indexed bin; a TAD it
# assembles from left bin i to right bin j is reported by ``callers.call_robustad``
# as the interval (i + 1, j + 1), the same 1-indexed-inclusive convention the
# truth uses. Both score maps are therefore stored SHIFTED by +1 at cache time,
# so an interval (a, b) reads its own boundary evidence as left[a] and right[b]
# with no further arithmetic anywhere downstream.
# ---------------------------------------------------------------------------
class BoundaryScores:
    """One leaf's per-position left/right boundary scores, tolerant of jitter."""

    def __init__(self, left: dict[int, float], right: dict[int, float], tol: int):
        self.left = left
        self.right = right
        self.tol = tol

    @classmethod
    def from_record(cls, rec: dict, tol: int) -> "BoundaryScores":
        return cls({int(k): float(v) for k, v in rec.get("left", {}).items()},
                   {int(k): float(v) for k, v in rec.get("right", {}).items()},
                   tol)

    def _near(self, table: dict[int, float], pos: int) -> float:
        """Best score within +/- tol of ``pos``.

        Snapping moves boundaries by up to ``tol``, and the scoring metric calls
        two TADs the same within ``tol`` anyway, so an exact lookup would report
        "no evidence" for a leaf whose own boundary sits one bin over -- the very
        jitter this experiment exists to absorb.
        """
        best = 0.0
        for d in range(-self.tol, self.tol + 1):
            v = table.get(pos + d)
            if v is not None and v > best:
                best = v
        return best

    def start(self, pos: int) -> float:
        return self._near(self.left, pos)

    def end(self, pos: int) -> float:
        return self._near(self.right, pos)

    def interval_support(self, a: int, b: int) -> float:
        """A leaf's own evidence for a domain at (a, b), in [0, 1].

        The mean of the two boundary scores, not the product: a leaf that sees
        one of the two boundaries clearly has real (if partial) evidence, and a
        product would zero it out whenever either end is unseen.
        """
        return 0.5 * (self.start(a) + self.end(b))


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------
def method_raw(tree: Tree, calls: dict[str, set]) -> dict[str, set]:
    """The caller's own leaf output, untouched: the no-tree baseline."""
    return {leaf: set(calls[leaf]) for leaf in tree.leaves}


def snap_map(weights: dict[int, float], tol: int) -> dict[int, int]:
    """Map jittered boundary positions onto consensus positions.

    ``weights`` is position -> support. Greedy mode-seeking, NOT single-linkage:
    repeatedly take the best-supported unassigned position as a centre and absorb
    everything within ``tol`` of it. Single-linkage would chain 100-102-104 into
    one cluster whose centre is 2 bins from either end, and a boundary displaced
    by more than the scoring tolerance breaks a match the raw call already had.
    Mode-seeking bounds every displacement by ``tol`` and snaps to a real,
    well-supported position rather than an invented mean.
    """
    remaining = dict(weights)
    out: dict[int, int] = {}
    while remaining:
        centre = max(remaining, key=lambda p: (remaining[p], -p))
        for p in [q for q in remaining if abs(q - centre) <= tol]:
            out[p] = centre
            del remaining[p]
    return out


def snap_calls(calls: dict[str, set], leaves: list[str], tol: int,
               boundary: dict[str, BoundaryScores] | None = None):
    """Rewrite every leaf's calls onto consensus boundaries.

    Starts and ends are snapped in separate coordinate spaces -- they are
    different roles, and pooling them would let a domain start absorb a
    neighbouring domain end.

    With ``boundary`` given, a position's support is ``sum(1 + score)`` over the
    leaves that called it instead of a plain count, so among positions with equal
    sibling support the consensus lands on the one RobusTAD is most confident
    about. The ``1 +`` keeps sibling count dominant: no single confident leaf can
    outvote two agreeing ones.

    Returns ``(snapped, back)`` where ``back[leaf][snapped_iv]`` is an original
    interval the leaf actually called. The back-map is what keeps the method
    strictly additive over the baseline: a leaf's own calls are reported in their
    ORIGINAL coordinates, so snapping can never degrade a match the raw caller
    already had, and only genuinely new (imputed) intervals get consensus
    coordinates.
    """
    start_sup: dict[int, float] = {}
    end_sup: dict[int, float] = {}
    for leaf in leaves:
        bs = boundary.get(leaf) if boundary else None
        for (a, b) in calls[leaf]:
            start_sup[a] = start_sup.get(a, 0.0) + 1.0 + (bs.start(a) if bs else 0.0)
            end_sup[b] = end_sup.get(b, 0.0) + 1.0 + (bs.end(b) if bs else 0.0)
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


def build_leaf_importance(
    tree: Tree,
    snapped: dict[str, set],
    back: dict[str, dict],
    tad_scores: dict[str, dict[tuple[int, int], float]],
    boundary: dict[str, BoundaryScores],
    calibrate: bool = True,
) -> dict[int, dict[tuple[int, int], dict[str, float]]]:
    """RobusTAD confidence for every leaf call, in the weighted solver's format.

    ``leaf_importance[leaf_id][(a, b)] = {"x": ..., "s": ..., "e": ...}``, keyed
    by the SNAPPED interval the solver sees but scored at the leaf's ORIGINAL
    coordinates -- the score belongs to the call RobusTAD actually made, and
    snapping only renames it.

    The solver turns each score ``c`` into the term ``3c + var*(1 - 6c)`` on the
    leaf's PARENT, i.e. ``|var - 3c|``. Selecting the variable is *penalised*
    while ``3c < 0.5`` and *rewarded* above it, so **c = 1/6 is the indifference
    point** -- and that is the problem with feeding raw scores in.

    Measured on the pilot cache, RobusTAD's TAD score averages 0.84 at a leaf's
    true positives and 0.33 at its false positives (AUC 0.95, so the signal is
    excellent). But *both* are far above 1/6, so raw scores reward the parent for
    carrying true and false domains alike -- differing only in how much. That is
    what the tuning showed: raw importance was neutral at a small weight and
    actively harmful at a large one, because it inflates the parent rather than
    discriminating on it.

    ``calibrate=True`` fixes this by passing the **percentile rank** of each
    score within that leaf's own calls, scaled to put the median call exactly at
    the indifference point (``x = r / 3`` gives ``3c = r``, neutral at r = 0.5).
    The parent is then pulled toward a leaf's better-than-median calls and pushed
    away from its worse-than-median ones, which is the discrimination the raw
    scores could not express. It is also self-calibrating: no constant here has
    to know what RobusTAD's score distribution looks like at a given depth, and
    the ranks come out of the same machinery the imputation prices use.
    """
    out: dict[int, dict[tuple[int, int], dict[str, float]]] = {}
    nid = tree.node_id
    for leaf in tree.leaves:
        bs = boundary.get(leaf)
        scores = tad_scores.get(leaf, {})
        ivs = sorted(snapped[leaf])
        origs = [back[leaf].get(iv, iv) for iv in ivs]
        raw_x = [float(scores.get(o, 0.0)) for o in origs]
        raw_s = [bs.start(o[0]) if bs else None for o in origs]
        raw_e = [bs.end(o[1]) if bs else None for o in origs]

        if calibrate:
            val_x = [r / 3.0 for r in percentile_ranks(raw_x)]
            val_s = ([r / 3.0 for r in percentile_ranks([v for v in raw_s])]
                     if bs else raw_s)
            val_e = ([r / 3.0 for r in percentile_ranks([v for v in raw_e])]
                     if bs else raw_e)
        else:
            val_x, val_s, val_e = raw_x, raw_s, raw_e

        entries: dict[tuple[int, int], dict[str, float]] = {}
        for i, iv in enumerate(ivs):
            entry = {"x": val_x[i]}
            if bs is not None:
                entry["s"] = val_s[i]
                entry["e"] = val_e[i]
            entries[iv] = entry
        out[nid[leaf]] = entries
    return out


def percentile_ranks(values: list[float]) -> list[float]:
    """Mid-rank percentile of each value within its own list, in [0, 1].

    All-equal input maps to 0.5 everywhere, which is the right degenerate
    behaviour here: a leaf whose boundary scores say nothing gets the flat price.
    """
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: values[i])
    out = [0.5] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        mid = (i + j) / 2.0
        r = mid / (n - 1) if n > 1 else 0.5
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def build_impute_prices(
    tree: Tree,
    snapped: dict[str, set],
    candidates: list[tuple[int, int]],
    boundary: dict[str, BoundaryScores],
    impute_cost: float,
    discount: float,
    floor_frac: float,
) -> dict:
    """Per-(leaf, interval) price for gaining a TAD the leaf did not call.

    ``price = impute_cost * (1 + discount * (1 - 2 * r))`` where ``r`` is the
    percentile rank of the leaf's OWN boundary support for ``(a, b)`` among all
    the candidates that leaf did not call. This is the whole point of using
    RobusTAD: it reports boundary evidence at bins where it assembled nothing, so
    a leaf that half-sees a domain can be talked into it by fewer relatives than
    a leaf that sees nothing there.

    Two design points, both load-bearing:

    * **Percentile, not the raw score.** RobusTAD's boundary field is dense and
      sits high: measured over the pilot, the mean support at intervals that ARE
      true for a leaf is 0.95 against 0.76 at intervals that are not (AUC ~0.78).
      The signal is real but the raw *levels* are nearly equal, so a raw-score
      discount would price true and false candidates within 20% of each other.
      Ranking within the leaf's own candidate pool restores the spread and makes
      the discount adaptive to how confident that leaf's whole field is.

    * **Two-sided around the flat price.** ``r = 0.5`` costs exactly
      ``impute_cost``, so the MEAN price across candidates is the flat price and
      the weighted method spends the same imputation budget as the unweighted
      control -- it only spends it in different places. A one-sided discount
      would just impute more, and any F1 difference would be confounded with the
      extra volume.

    The price is floored at ``floor_frac * impute_cost`` and never reaches zero:
    the objective charges only for differences along edges, so a free interval on
    a parent and all its children costs nothing and the LP floods every leaf.
    """
    prices = {None: float(impute_cost)}
    nid = tree.node_id
    for leaf in tree.leaves:
        bs = boundary.get(leaf)
        if bs is None:
            continue
        have = snapped[leaf]
        uncalled = [iv for iv in candidates if iv not in have]
        if not uncalled:
            continue
        ranks = percentile_ranks([bs.interval_support(a, b) for (a, b) in uncalled])
        for iv, r in zip(uncalled, ranks):
            price = impute_cost * (1.0 + discount * (1.0 - 2.0 * r))
            prices[(nid[leaf], iv)] = max(price, floor_frac * impute_cost)
    return prices


def method_impute(
    tree: Tree,
    calls: dict[str, set],
    tad_scores: dict[str, dict[tuple[int, int], float]],
    boundary: dict[str, BoundaryScores],
    impute_cost: float,
    ct: float,
    cb: float,
    mode: str,
    tol: int,
    use_importance: bool,
    use_price_discount: bool,
    weighted_snap: bool,
    importance_weight: float,
    price_discount: float,
    price_floor: float,
    calibrate_importance: bool = True,
):
    """Imputation on snapped coordinates with a restricted candidate universe.

    Three switches select the method:

    ``use_importance``      feed RobusTAD's per-call confidence as
                            ``leaf_importance`` (TAD score + both boundary
                            scores), which biases the leaf's PARENT toward
                            carrying domains the leaf is confident about.
    ``use_price_discount``  price each leaf's imputation by its OWN boundary
                            evidence instead of charging everyone the same.
    ``weighted_snap``       weight consensus boundary positions by RobusTAD's
                            boundary score as well as by sibling count.

    All three off reproduces the unweighted OnTAD-arm method exactly, which is
    the control this experiment measures the weighting against.

    The candidate universe is restricted to intervals some leaf actually called.
    The all-pairs universe grows as (endpoints)^2 and dominates the model size;
    an interval no leaf called can only ever cost the objective, so dropping it
    is close to free and makes the model linear in the number of distinct calls.

    The result is strictly additive over the raw calls: every original call is
    returned unchanged, plus whatever was imputed.
    """
    from check_lp_constrained_weighted_updated import solve_lp_from_edges_and_leaves

    nid = tree.node_id
    snapped, back = snap_calls(calls, tree.leaves, tol,
                               boundary if weighted_snap else None)
    candidates = sorted(set().union(*(snapped[l] for l in tree.leaves))) if tree.leaves else []
    leaf_intervals = {nid[l]: {f"[{a},{b}]" for (a, b) in snapped[l]} for l in tree.leaves}

    leaf_importance = None
    if use_importance:
        leaf_importance = build_leaf_importance(tree, snapped, back, tad_scores,
                                                boundary, calibrate_importance)

    cost: float | dict = impute_cost
    if use_price_discount:
        cost = build_impute_prices(tree, snapped, candidates, boundary,
                                   impute_cost, price_discount, price_floor)

    lp_cost, _, assignments, _, _, universe = solve_lp_from_edges_and_leaves(
        tree.edge_ids, leaf_intervals, Ct=ct, Cb=cb, mode=mode,
        leaf_importance=leaf_importance, leaf_importance_weight=importance_weight,
        require_laminar_internal=True, use_fitch_hartigan=False, verbose=False,
        impute=True, impute_cost=cost, root_intervals=None,
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


# ---------------------------------------------------------------------------
# Method registry
#
# Every priced method is "<prefix>_c<cost>"; the prefix picks which of the three
# weighting switches are on, so the plot can group them and the ablations are
# named the same way as the headline.
# ---------------------------------------------------------------------------
METHOD_SWITCHES = {
    # prefix:  (importance, price_discount, weighted_snap)
    "snap":  (False, False, False),   # control: the unweighted OnTAD-arm method
    "wsnap": (True,  True,  True),    # headline: everything RobusTAD's scores buy
    "wimp":  (True,  False, False),   # ablation: leaf_importance alone
    "wpri":  (False, True,  False),   # ablation: discounted price alone
}

METHOD_DESCRIPTION = {
    "snap": "tree-informed imputation, unweighted",
    "wsnap": "tree-informed imputation, RobusTAD-weighted",
    "wimp": "weighted: leaf importance only",
    "wpri": "weighted: boundary-discounted price only",
}


def method_name(prefix: str, cost: float) -> str:
    return f"{prefix}_c{cost:g}"


def split_method(name: str) -> tuple[str, float]:
    prefix, _, cost = name.rpartition("_c")
    return prefix, float(cost)


def method_label(name: str, headline_cost: float) -> str:
    if name == RAW_METHOD:
        return "RobusTAD (no tree)"
    prefix, cost = split_method(name)
    tag = " [headline]" if cost == headline_cost else ""
    return f"{METHOD_DESCRIPTION.get(prefix, prefix)} (cost={cost:g}){tag}"


# ---------------------------------------------------------------------------
# CALL stage
# ---------------------------------------------------------------------------
def robustad_caller_params(params: config.SimParams, args) -> dict:
    resol = params.resolution_bp
    return {
        "minW_kb": args.rt_minW,
        "maxW_kb": args.rt_maxW,
        "ratio": args.rt_ratio,
        "alpha": args.rt_alpha,
        "mintad_bp": args.rt_mintad_bins * resol,
        "maxtad_bp": args.rt_maxtad_bins * resol,
        "delta": args.rt_delta,
    }


def read_boundary_scores(work_dir: Path, resol: int, floor: float) -> dict[str, dict[int, float]]:
    """Read the RobusTAD boundary table left behind by ``callers.call_robustad``.

    Returns ``{"left": {pos: score}, "right": {pos: score}}`` with positions
    SHIFTED to the call convention (bin + 1) and scores below ``floor`` dropped,
    so an interval (a, b) looks up left[a] / right[b] directly.
    """
    import run_robustad as rt
    import pandas as pd

    bed = work_dir / "m_boundary.bed"
    if not bed.exists():
        return {"left": {}, "right": {}}
    with_bins = rt.convert_boundary(bed, resol)
    df = pd.read_csv(with_bins, sep="\t")
    out = {"left": {}, "right": {}}
    if df.empty:
        return out
    for col, key in (("leftScore", "left"), ("rightScore", "right")):
        if col not in df.columns:
            continue
        sub = df[df[col] >= floor]
        out[key] = {int(b) + 1: round(float(s), 4)
                    for b, s in zip(sub["bin"], sub[col])}
    return out


def run_call_stage(tree: Tree, region: int, params: config.SimParams, args) -> dict:
    """Run RobusTAD on every leaf at every depth level and cache calls + scores."""
    import callers

    hiers = build_hierarchies(tree, region, params)
    caller_params = robustad_caller_params(params, args)
    resol = params.resolution_bp
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
        leaf_calls, leaf_boundary = {}, {}
        for leaf in tree.leaves:
            mat = downsample(bases[leaf], depth_value, deg_rng)
            matrix_path = work_dir / f"{leaf}.matrix"
            callers.write_matrix(mat, matrix_path)
            leaf_dir = work_dir / f"{leaf}_rt"
            # The same leaf_dir is reused down the whole depth ladder, and
            # RobusTAD failing on a badly degraded matrix leaves its previous
            # outputs in place -- which would silently attach the PREVIOUS rung's
            # boundary scores to this one's (empty) calls.
            for stale in ("m_boundary.bed", "m_boundary_with_bins.tsv",
                          "m_tads_all.bed", "m_tads_hq.bed",
                          "m_tads_all_with_bins.tsv"):
                (leaf_dir / stale).unlink(missing_ok=True)
            try:
                leaf_calls[leaf] = callers.call_robustad(
                    matrix_path, leaf_dir, resol, caller_params, args.timeout,
                )
                leaf_boundary[leaf] = read_boundary_scores(
                    leaf_dir, resol, args.boundary_floor)
            finally:
                matrix_path.unlink(missing_ok=True)
        n_calls = sum(len(v) for v in leaf_calls.values())
        points.append({
            "depth_label": depth_label,
            "depth_value": depth_value,
            "leaf_calls": {
                leaf: sorted([a, b, round(float(sc[(a, b)]), 4)] for (a, b) in sc)
                for leaf, sc in leaf_calls.items()
            },
            "leaf_boundary": leaf_boundary,
        })
        print(f"{tree.name} region_{region:03d} depth={depth_label}: "
              f"{n_calls} leaf calls over {len(tree.leaves)} leaves", flush=True)

    return {
        "tree": tree.name, "region": region, "caller": "robustad",
        "leaves": tree.leaves, "amplitude": args.amplitude,
        "caller_params": {k: v for k, v in caller_params.items()},
        "points": points,
    }


# ---------------------------------------------------------------------------
# SCORE stage
# ---------------------------------------------------------------------------
def run_score_stage(tree: Tree, region: int, params: config.SimParams, args) -> dict:
    """Solve + score every method at every noise level from the cached calls."""
    hiers = build_hierarchies(tree, region, params)
    truth_by_leaf = {leaf: true_intervals(hiers[leaf]) for leaf in tree.leaves}

    record = json.loads(Path(args.calls_json).read_text())
    methods = [RAW_METHOD] + [method_name(p, c)
                              for p in args.method_prefixes for c in args.impute_costs]

    points = []
    for pt in record["points"]:
        calls, tad_scores = {}, {}
        for leaf, rows in pt["leaf_calls"].items():
            calls[leaf] = {(int(r[0]), int(r[1])) for r in rows}
            tad_scores[leaf] = {(int(r[0]), int(r[1])): float(r[2]) for r in rows}
        boundary = {
            leaf: BoundaryScores.from_record(rec, args.tolerance)
            for leaf, rec in pt.get("leaf_boundary", {}).items()
        }

        summary = []
        for name in methods:
            t0 = time.time()
            n_candidates = None
            status = "ok"
            if name == RAW_METHOD:
                pred = method_raw(tree, calls)
            else:
                prefix, cost = split_method(name)
                use_imp, use_pri, wsnap = METHOD_SWITCHES[prefix]
                pred, n_candidates = method_impute(
                    tree, calls, tad_scores, boundary, cost,
                    args.Ct, args.Cb, args.solve_mode, args.tolerance,
                    use_importance=use_imp, use_price_discount=use_pri,
                    weighted_snap=wsnap,
                    importance_weight=args.importance_weight,
                    price_discount=args.price_discount,
                    price_floor=args.price_floor,
                    calibrate_importance=not args.raw_importance,
                )
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
        "tree": tree.name, "region": region, "caller": "robustad",
        "methods": methods, "headline_cost": args.headline_cost,
        "headline_prefix": args.headline_prefix,
        "importance_weight": args.importance_weight,
        "price_discount": args.price_discount,
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
    p.add_argument("--impute-costs", default="0.5,0.75,1.5,3.0",
                   help="comma-separated impute_cost values to score")
    p.add_argument("--methods", default="snap,wsnap",
                   help="comma-separated method prefixes from "
                        f"{sorted(METHOD_SWITCHES)} (each run at every impute cost)")
    p.add_argument("--headline-cost", type=float, default=0.5,
                   help="which impute_cost the headline figure plots")
    p.add_argument("--headline-prefix", default="wsnap",
                   help="which method the headline figure plots")
    p.add_argument("--importance-weight", type=float, default=0.1,
                   help="weight on the RobusTAD leaf-importance term. The solver "
                        "scales every score by 3, so a score-1.0 call rewards its "
                        "parent by 5*w per term (x, s and e); at w=1 that swamps "
                        "the unit parsimony costs entirely")
    p.add_argument("--price-discount", type=float, default=0.6,
                   help="how far a leaf's own boundary evidence moves the impute "
                        "price around the flat one, in [0,1): the best-supported "
                        "candidate pays (1-d)*cost, the worst (1+d)*cost, and the "
                        "median exactly cost")
    p.add_argument("--raw-importance", action="store_true",
                   help="feed RobusTAD's raw scores to leaf_importance instead of "
                        "their per-leaf percentile ranks. Raw scores sit almost "
                        "entirely above the solver's indifference point (1/6), so "
                        "they reward true and false calls alike -- measured to be "
                        "neutral at best. Kept for reproducing that finding")
    p.add_argument("--price-floor", type=float, default=0.25,
                   help="minimum price as a fraction of impute_cost; imputation "
                        "must never become free (the LP would flood every leaf)")
    p.add_argument("--Ct", type=float, default=1.0)
    p.add_argument("--Cb", type=float, default=1.0)
    p.add_argument("--solve-mode", dest="solve_mode", default="convex")
    p.add_argument("--tolerance", type=int, default=d.tolerance)

    # simulation (must match the OnTAD arm for the two to be comparable)
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

    # RobusTAD
    p.add_argument("--rt-minW", type=int, default=80, help="min window, kb")
    p.add_argument("--rt-maxW", type=int, default=2000, help="max window, kb")
    p.add_argument("--rt-ratio", type=float, default=1.2)
    p.add_argument("--rt-alpha", type=float, default=0.05)
    p.add_argument("--rt-delta", type=float, default=0.1)
    p.add_argument("--rt-mintad-bins", type=int, default=8,
                   help="min assembled TAD size in BINS (x resolution for RobusTAD's bp)")
    p.add_argument("--rt-maxtad-bins", type=int, default=200,
                   help="max assembled TAD size in BINS")
    p.add_argument("--boundary-floor", type=float, default=BOUNDARY_SCORE_FLOOR,
                   help="drop cached boundary scores below this")
    p.add_argument("--timeout", type=int, default=900)

    args = p.parse_args()
    args.impute_costs = [float(x) for x in str(args.impute_costs).split(",") if x.strip()]
    args.method_prefixes = [x.strip() for x in str(args.methods).split(",") if x.strip()]
    unknown = [m for m in args.method_prefixes if m not in METHOD_SWITCHES]
    if unknown:
        raise SystemExit(f"unknown method prefix(es) {unknown}; "
                         f"choose from {sorted(METHOD_SWITCHES)}")
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
