import random
import re
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

# Optional external dependency used by your older discrete LP experiments.
# The updated weighted DP below does not depend on it.
try:
    from check_lp_constrained import solve_lp_from_edges_and_leaves  # noqa: F401
except Exception:
    solve_lp_from_edges_and_leaves = None

try:
    from random_simulation import (
        generate_random_intervals,
        generate_random_laminar_intervals_from_list,
    )
except Exception:
    generate_random_intervals = None
    generate_random_laminar_intervals_from_list = None

Interval = Tuple[int, int]
WeightedIntervalEntry = Union[str, Tuple[Union[str, Sequence[int], Interval], float], List[Union[str, Sequence[int], Interval, float]]]


# -----------------------------------------------------------------------------
# Parsing / normalization helpers
# -----------------------------------------------------------------------------

def parse_interval_from_anything(x) -> Interval:
    """Parse intervals from strings like '[1,10]' or from pairs like (1, 10)."""
    if isinstance(x, str):
        nums = re.findall(r"\d+", x)
        if len(nums) != 2:
            raise ValueError(f"Cannot parse interval from string: {x}")
        return int(nums[0]), int(nums[1])

    if isinstance(x, (tuple, list)) and len(x) == 2:
        a, b = x
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return int(a), int(b)
        if isinstance(a, str):
            return parse_interval_from_anything(a)

    raise ValueError(f"Cannot parse interval from: {x!r}")


def parse_weighted_interval_entry(entry) -> Tuple[Interval, float]:
    """Parse a weighted interval entry.

    Supported forms:
      - '[1,10]'                       -> weight 1.0
      - ([1,10], 0.8)
      - ((1,10), 0.8)
      - ['[1,10]', 0.8]
    """
    if isinstance(entry, str):
        return parse_interval_from_anything(entry), 1.0

    if isinstance(entry, (tuple, list)) and len(entry) == 2:
        interval_part, weight = entry
        interval = parse_interval_from_anything(interval_part)
        return interval, float(weight)

    raise ValueError(f"Cannot parse weighted interval entry: {entry!r}")


def normalize_weighted_leaf_intervals(
    leaf_intervals: Dict[int, Sequence[WeightedIntervalEntry]]
) -> Dict[int, Dict[Interval, float]]:
    """Convert per-child interval lists into dict[interval] = weight.

    If an interval appears multiple times for a child, the maximum weight is kept.
    """
    normalized: Dict[int, Dict[Interval, float]] = {}
    for child, entries in leaf_intervals.items():
        d: Dict[Interval, float] = {}
        for entry in entries:
            interval, weight = parse_weighted_interval_entry(entry)
            d[interval] = max(d.get(interval, 0.0), float(weight))
        normalized[child] = d
    return normalized


def derive_boundary_weights_from_intervals(
    interval_weights: Dict[int, Dict[Interval, float]]
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, float]]]:
    """Derive boundary weights from interval weights by taking max weight at endpoints."""
    start_weights: Dict[int, Dict[int, float]] = {}
    end_weights: Dict[int, Dict[int, float]] = {}
    for child, intervals in interval_weights.items():
        s: Dict[int, float] = {}
        e: Dict[int, float] = {}
        for (a, b), w in intervals.items():
            s[a] = max(s.get(a, 0.0), float(w))
            e[b] = max(e.get(b, 0.0), float(w))
        start_weights[child] = s
        end_weights[child] = e
    return start_weights, end_weights


def normalize_boundary_weights(
    boundary_weights: Optional[Dict[int, Dict[int, float]]],
    children: Iterable[int],
) -> Dict[int, Dict[int, float]]:
    """Normalize boundary-weight dicts; missing children get empty dicts."""
    out: Dict[int, Dict[int, float]] = {}
    children = list(children)
    for c in children:
        out[c] = {}
    if boundary_weights is None:
        return out
    for c, d in boundary_weights.items():
        out[c] = {int(pos): float(w) for pos, w in d.items()}
    return out


def build_leaf_state(
    edges: List[Tuple[int, int]],
    leaf_intervals: Dict[int, Sequence[WeightedIntervalEntry]],
    leaf_start_weights: Optional[Dict[int, Dict[int, float]]] = None,
    leaf_end_weights: Optional[Dict[int, Dict[int, float]]] = None,
) -> Dict[int, Dict[str, Dict]]:
    """Build a per-child state with:
        state[c]['intervals'] = {(a,b): w}
        state[c]['starts']    = {a: w}
        state[c]['ends']      = {b: w}

    If start/end weights are not provided, they are derived from interval weights.
    """
    children = [v for (u, v) in edges if u == 0]
    interval_weights = normalize_weighted_leaf_intervals(leaf_intervals)

    if leaf_start_weights is None or leaf_end_weights is None:
        derived_starts, derived_ends = derive_boundary_weights_from_intervals(interval_weights)
        if leaf_start_weights is None:
            leaf_start_weights = derived_starts
        if leaf_end_weights is None:
            leaf_end_weights = derived_ends

    start_weights = normalize_boundary_weights(leaf_start_weights, children)
    end_weights = normalize_boundary_weights(leaf_end_weights, children)

    state: Dict[int, Dict[str, Dict]] = {}
    for c in children:
        state[c] = {
            "intervals": dict(interval_weights.get(c, {})),
            "starts": dict(start_weights.get(c, {})),
            "ends": dict(end_weights.get(c, {})),
        }
    return state


def clone_state_map(state_map: Dict[int, Dict[str, Dict]]) -> Dict[int, Dict[str, Dict]]:
    return {
        c: {
            "intervals": dict(state_map.get(c, {}).get("intervals", {})),
            "starts": dict(state_map.get(c, {}).get("starts", {})),
            "ends": dict(state_map.get(c, {}).get("ends", {})),
        }
        for c in state_map
    }


def merge_child_maps(
    m1: Dict[int, Dict[str, Dict]],
    m2: Dict[int, Dict[str, Dict]],
) -> Dict[int, Dict[str, Dict]]:
    """Merge two child-state maps by taking the max weight for each item."""
    merged: Dict[int, Dict[str, Dict]] = {}
    keys = set(m1.keys()) | set(m2.keys())
    for c in keys:
        merged[c] = {"intervals": {}, "starts": {}, "ends": {}}
        for category in ("intervals", "starts", "ends"):
            tmp: Dict = {}
            for src in (m1, m2):
                cat = src.get(c, {}).get(category, {})
                for k, v in cat.items():
                    tmp[k] = max(tmp.get(k, 0.0), float(v))
            merged[c][category] = tmp
    return merged


# -----------------------------------------------------------------------------
# Accessors / score helpers
# -----------------------------------------------------------------------------

def get_interval_weight(c: int, interval: Interval, cmap: Dict[int, Dict[str, Dict]]) -> float:
    return float(cmap.get(c, {}).get("intervals", {}).get(interval, 0.0))


def get_start_weight(c: int, pos: int, cmap: Dict[int, Dict[str, Dict]]) -> float:
    return float(cmap.get(c, {}).get("starts", {}).get(pos, 0.0))


def get_end_weight(c: int, pos: int, cmap: Dict[int, Dict[str, Dict]]) -> float:
    return float(cmap.get(c, {}).get("ends", {}).get(pos, 0.0))


def child_has_start_at(c: int, pos: int, cmap: Dict[int, Dict[str, Dict]]) -> bool:
    return get_start_weight(c, pos, cmap) > 0.0


def child_has_end_at(c: int, pos: int, cmap: Dict[int, Dict[str, Dict]]) -> bool:
    return get_end_weight(c, pos, cmap) > 0.0


def count_children_with_interval(a: int, b: int, cmap: Dict[int, Dict[str, Dict]], children: List[int]) -> float:
    return sum(get_interval_weight(c, (a, b), cmap) for c in children)


def count_children_missing_interval(a: int, b: int, cmap: Dict[int, Dict[str, Dict]], children: List[int]) -> float:
    return sum(1.0 - get_interval_weight(c, (a, b), cmap) for c in children)


def count_child_intervals_start_ge_i_end_eq_j(
    i: int,
    j: int,
    cmap: Dict[int, Dict[str, Dict]],
    positions: List[int],
    children: List[int],
) -> float:
    a = positions[i]
    b = positions[j]
    cnt = 0.0
    for c in children:
        for (x, y), w in cmap.get(c, {}).get("intervals", {}).items():
            if (x, y) == (a, b):
                continue
            if x >= a and y == b:
                cnt += float(w)
    return cnt


def count_child_intervals_start_eq_i_end_le_j(
    i: int,
    j: int,
    cmap: Dict[int, Dict[str, Dict]],
    positions: List[int],
    children: List[int],
) -> float:
    a = positions[i]
    b = positions[j]
    cnt = 0.0
    for c in children:
        for (x, y), w in cmap.get(c, {}).get("intervals", {}).items():
            if (x, y) == (a, b):
                continue
            if x == a and y <= b:
                cnt += float(w)
    return cnt


def count_child_intervals_crossing_k(
    i: int,
    k: int,
    j: int,
    cmap: Dict[int, Dict[str, Dict]],
    positions: List[int],
    children: List[int],
) -> float:
    a = positions[i]
    pk = positions[k]
    b = positions[j]
    cnt = 0.0
    for c in children:
        for (x, y), w in cmap.get(c, {}).get("intervals", {}).items():
            if (x, y) == (a, b):
                continue
            if a <= x < pk and pk < y <= b:
                cnt += float(w)
    return cnt


def boundary_cost_include_start(a_pos: int, cmap: Dict[int, Dict[str, Dict]], children: List[int], Cb: float) -> float:
    return Cb * sum(1.0 - get_start_weight(c, a_pos, cmap) for c in children)


def boundary_cost_include_end(b_pos: int, cmap: Dict[int, Dict[str, Dict]], children: List[int], Cb: float) -> float:
    return Cb * sum(1.0 - get_end_weight(c, b_pos, cmap) for c in children)


def boundary_cost_exclude_start(a_pos: int, cmap: Dict[int, Dict[str, Dict]], children: List[int], Cb: float) -> float:
    return Cb * sum(get_start_weight(c, a_pos, cmap) for c in children)


def boundary_cost_exclude_end(b_pos: int, cmap: Dict[int, Dict[str, Dict]], children: List[int], Cb: float) -> float:
    return Cb * sum(get_end_weight(c, b_pos, cmap) for c in children)


def laminar_ok_to_impute_for_child(c: int, interval: Interval, cmap: Dict[int, Dict[str, Dict]]) -> bool:
    """Check whether adding interval to child c would preserve laminarity."""
    a, b = interval
    for (x, y) in cmap.get(c, {}).get("intervals", {}):
        if (x, y) == (a, b):
            continue
        if (x < a < y < b) or (a < x < b < y):
            return False
    return True


def impute_interval_for_child(c: int, interval: Interval, cmap: Dict[int, Dict[str, Dict]]) -> None:
    """Set interval and its boundary weights to 1 for child c."""
    a, b = interval
    cmap.setdefault(c, {"intervals": {}, "starts": {}, "ends": {}})
    if interval not in cmap[c]["intervals"]:
        cmap[c].setdefault("intervals", {})[interval] = 1.0
    if not child_has_start_at(c, a, cmap):
        cmap[c].setdefault("starts", {})[a] = 1.0
    if not child_has_end_at(c, b, cmap):
        cmap[c].setdefault("ends", {})[b] = 1.0


# -----------------------------------------------------------------------------
# Core DP
# -----------------------------------------------------------------------------

def infer_tads_with_dp(
    edges,
    leaf_intervals,
    start_boundaries,
    end_boundaries,
    leaf_start_weights: Optional[Dict[int, Dict[int, float]]] = None,
    leaf_end_weights: Optional[Dict[int, Dict[int, float]]] = None,
    Ct=1.0,
    Cb=1.0,
    trace=False,
):
    """Weighted Nussinov-style DP with explicit child imputation.

    The leaf inputs may be weighted:
      leaf_intervals[c] can contain entries like ([1,10], 0.8)
      leaf_start_weights[c][pos] and leaf_end_weights[c][pos] are optional.

    The DP still chooses parent intervals discretely, but scores are weighted.
    Whenever an interval is imputed into a child, that interval and its boundary
    weights are set to 1.0 for that child.
    """
    children = [v for (u, v) in edges if u == 0]
    initial_children_map = build_leaf_state(
        edges,
        leaf_intervals,
        leaf_start_weights=leaf_start_weights,
        leaf_end_weights=leaf_end_weights,
    )

    positions = sorted(set(start_boundaries) | set(end_boundaries))
    n = len(positions)
    if n < 2:
        if trace:
            return set(), 0.0, initial_children_map, []
        return set(), 0.0, initial_children_map

    # Memo tables.
    inf = float("inf")
    dp0_done = [[False] * n for _ in range(n)]
    dpl_done = [[False] * n for _ in range(n)]
    dpr_done = [[False] * n for _ in range(n)]
    dp0_val = [[None] * n for _ in range(n)]
    dpl_val = [[None] * n for _ in range(n)]
    dpr_val = [[None] * n for _ in range(n)]

    def dp0(i, j):
        if i >= j:
            return frozenset(), inf, None, initial_children_map
        if dp0_done[i][j]:
            return dp0_val[i][j]

        a = positions[i]
        b = positions[j]
        if a not in start_boundaries or b not in end_boundaries:
            dp0_done[i][j] = True
            dp0_val[i][j] = (frozenset(), inf, None, initial_children_map)
            return dp0_val[i][j]

        # Base case: adjacent positions.
        if j == i + 1:
            # INCLUDE: impute interval into children if laminar.
            cmap = clone_state_map(initial_children_map)
            for c in children:
                if (a, b) in cmap[c]["intervals"]:
                    # Existing interval: promote weight to 1 and boundaries to 1.
                    impute_interval_for_child(c, (a, b), cmap)
                    continue
                if laminar_ok_to_impute_for_child(c, (a, b), cmap):
                    impute_interval_for_child(c, (a, b), cmap)
            miss = count_children_missing_interval(a, b, cmap, children)
            bcost = boundary_cost_include_start(a, cmap, children, Cb) + boundary_cost_include_end(b, cmap, children, Cb)
            include_score = Ct * miss + bcost
            include_parent = frozenset({(a, b)})

            # EXCLUDE: keep children unchanged.
            cmap_ex = initial_children_map
            have = count_children_with_interval(a, b, cmap_ex, children)
            bcost_ex = boundary_cost_exclude_start(a, cmap_ex, children, Cb) + boundary_cost_exclude_end(b, cmap_ex, children, Cb)
            exclude_score = Ct * have + bcost_ex
            exclude_parent = frozenset()

            if include_score <= exclude_score:
                res = (include_parent, include_score, ("base", "INCLUDE"), cmap)
            else:
                res = (exclude_parent, exclude_score, ("base", "EXCLUDE"), cmap_ex)

            dp0_done[i][j] = True
            dp0_val[i][j] = res
            return res

        best_score = inf
        best_parent_set = frozenset()
        best_choice = None
        best_cmap = initial_children_map

        # ------------------------------------------------------------------
        # INCLUDE [a,b] branches
        # ------------------------------------------------------------------

        # 1.1 call dpl(i, j-1)
        pset, sc, _, cmap_sub = dpl(i, j - 1)
        cmap_mut = clone_state_map(cmap_sub)
        for c in children:
            if (a, b) in cmap_mut[c]["intervals"]:
                impute_interval_for_child(c, (a, b), cmap_mut)
                continue
            if laminar_ok_to_impute_for_child(c, (a, b), cmap_mut):
                impute_interval_for_child(c, (a, b), cmap_mut)
        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i, j, cmap_mut, positions, children)
        miss = count_children_missing_interval(a, b, cmap_mut, children)
        c = positions[j - 1]
        bcost = (
            boundary_cost_include_start(a, cmap_mut, children, Cb)
            + boundary_cost_include_end(b, cmap_mut, children, Cb)
            + boundary_cost_exclude_start(c, cmap_mut, children, Cb)
        )
        total = sc + Ct * extra_cnt + Ct * miss + bcost
        cand_parent = frozenset({(a, b)}) | pset
        if total <= best_score:
            best_score = total
            best_parent_set = cand_parent
            best_choice = ("call", "dpl", i, j - 1)
            best_cmap = cmap_mut

        # 1.2 call dpr(i+1, j)
        pset, sc, _, cmap_sub = dpr(i + 1, j)
        cmap_mut = clone_state_map(cmap_sub)
        for c in children:
            if (a, b) in cmap_mut[c]["intervals"]:
                impute_interval_for_child(c, (a, b), cmap_mut)
                continue
            if laminar_ok_to_impute_for_child(c, (a, b), cmap_mut):
                impute_interval_for_child(c, (a, b), cmap_mut)
        extra_cnt = count_child_intervals_start_eq_i_end_le_j(i, j, cmap_mut, positions, children)
        miss = count_children_missing_interval(a, b, cmap_mut, children)
        c = positions[i + 1]
        bcost = (
            boundary_cost_include_start(a, cmap_mut, children, Cb)
            + boundary_cost_include_end(b, cmap_mut, children, Cb)
            + boundary_cost_exclude_end(c, cmap_mut, children, Cb)
        )
        total = sc + Ct * extra_cnt + Ct * miss + bcost
        cand_parent = frozenset({(a, b)}) | pset
        if total <= best_score:
            best_score = total
            best_parent_set = cand_parent
            best_choice = ("call", "dpr", i + 1, j)
            best_cmap = cmap_mut

        # 1.3 splits
        for k in range(i + 1, j):
            pk = positions[k]
            if (pk in start_boundaries) and (pk in end_boundaries):
                left_parent, left_score, _, left_cmap = dpl(i, k)
                right_parent, right_score, _, right_cmap = dpr(k, j)
                merged = merge_child_maps(left_cmap, right_cmap)
                cmap_mut = clone_state_map(merged)
                for c in children:
                    if (a, b) in cmap_mut[c]["intervals"]:
                        impute_interval_for_child(c, (a, b), cmap_mut)
                        continue
                    if laminar_ok_to_impute_for_child(c, (a, b), cmap_mut):
                        impute_interval_for_child(c, (a, b), cmap_mut)
                cross_cnt = count_child_intervals_crossing_k(i, k, j, cmap_mut, positions, children)
                miss = count_children_missing_interval(a, b, cmap_mut, children)
                bcost = boundary_cost_include_start(a, cmap_mut, children, Cb) + boundary_cost_include_end(b, cmap_mut, children, Cb)
                total = left_score + right_score + Ct * cross_cnt + Ct * miss + bcost
                cand_parent = frozenset({(a, b)}) | left_parent | right_parent
                if total <= best_score:
                    best_score = total
                    best_parent_set = cand_parent
                    best_choice = ("split", [("dpl", i, k), ("dpr", k, j)])
                    best_cmap = cmap_mut

        # ------------------------------------------------------------------
        # EXCLUDE [a,b] branches (no imputation)
        # ------------------------------------------------------------------

        pset, sc, _, cmap_sub = dp0(i, j - 1)
        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i, j, cmap_sub, positions, children)
        have = count_children_with_interval(a, b, cmap_sub, children)
        c = positions[j - 1]
        bcost = boundary_cost_exclude_end(b, cmap_sub, children, Cb) + boundary_cost_exclude_start(c, cmap_sub, children, Cb)
        total = sc + Ct * extra_cnt + Ct * have + bcost
        if total <= best_score:
            best_score = total
            best_parent_set = pset
            best_choice = ("call", "dp0", i, j - 1)
            best_cmap = cmap_sub

        pset, sc, _, cmap_sub = dp0(i + 1, j)
        extra_cnt = count_child_intervals_start_eq_i_end_le_j(i, j, cmap_sub, positions, children)
        have = count_children_with_interval(a, b, cmap_sub, children)
        c = positions[i + 1]
        bcost = boundary_cost_exclude_start(a, cmap_sub, children, Cb) + boundary_cost_exclude_end(c, cmap_sub, children, Cb)
        total = sc + Ct * extra_cnt + Ct * have + bcost
        if total <= best_score:
            best_score = total
            best_parent_set = pset
            best_choice = ("call", "dp0", i + 1, j)
            best_cmap = cmap_sub

        for k in range(i + 1, j):
            pk = positions[k]
            if (pk in start_boundaries) and (pk in end_boundaries):
                left_parent, left_score, _, left_cmap = dp0(i, k)
                right_parent, right_score, _, right_cmap = dp0(k, j)
                merged = merge_child_maps(left_cmap, right_cmap)
                cross_cnt = count_child_intervals_crossing_k(i, k, j, merged, positions, children)
                have = count_children_with_interval(a, b, merged, children)
                total = left_score + right_score + Ct * cross_cnt + Ct * have
                cand_parent = left_parent | right_parent
                if total < best_score:
                    best_score = total
                    best_parent_set = cand_parent
                    best_choice = ("split", [("dp0", i, k), ("dp0", k, j)])
                    best_cmap = merged

        dp0_done[i][j] = True
        dp0_val[i][j] = (best_parent_set, best_score, best_choice, best_cmap)
        return dp0_val[i][j]

    def dpl(i, j):
        if i >= j:
            return frozenset(), inf, None, initial_children_map
        if dpl_done[i][j]:
            return dpl_val[i][j]

        a = positions[i]
        b = positions[j]
        if a not in start_boundaries or b not in end_boundaries:
            dpl_done[i][j] = True
            dpl_val[i][j] = (frozenset(), inf, None, initial_children_map)
            return dpl_val[i][j]

        if j == i + 1:
            cmap = clone_state_map(initial_children_map)
            for c in children:
                if (a, b) in cmap[c]["intervals"]:
                    impute_interval_for_child(c, (a, b), cmap)
                    continue
                if laminar_ok_to_impute_for_child(c, (a, b), cmap):
                    impute_interval_for_child(c, (a, b), cmap)
            miss = count_children_missing_interval(a, b, cmap, children)
            bcost = boundary_cost_include_end(b, cmap, children, Cb)
            include_score = Ct * miss + bcost
            include_parent = frozenset({(a, b)})

            cmap_ex = initial_children_map
            have = count_children_with_interval(a, b, cmap_ex, children)
            bcost_ex = boundary_cost_exclude_end(b, cmap_ex, children, Cb)
            exclude_score = Ct * have + bcost_ex
            exclude_parent = frozenset()

            if include_score <= exclude_score:
                res = (include_parent, include_score, ("base", "INCLUDE"), cmap)
            else:
                res = (exclude_parent, exclude_score, ("base", "EXCLUDE"), cmap_ex)

            dpl_done[i][j] = True
            dpl_val[i][j] = res
            return res

        best_score = inf
        best_parent_set = frozenset()
        best_choice = None
        best_cmap = initial_children_map

        # INCLUDE via dpl(i, j-1)
        pset, sc, _, cmap_sub = dpl(i, j - 1)
        cmap_mut = clone_state_map(cmap_sub)
        for c in children:
            if (a, b) in cmap_mut[c]["intervals"]:
                impute_interval_for_child(c, (a, b), cmap_mut)
                continue
            if laminar_ok_to_impute_for_child(c, (a, b), cmap_mut):
                impute_interval_for_child(c, (a, b), cmap_mut)
        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i, j, cmap_mut, positions, children)
        miss = count_children_missing_interval(a, b, cmap_mut, children)
        c = positions[j - 1]
        bcost = boundary_cost_include_end(b, cmap_mut, children, Cb) + boundary_cost_exclude_start(c, cmap_mut, children, Cb)
        total = sc + Ct * extra_cnt + Ct * miss + bcost
        cand_parent = frozenset({(a, b)}) | pset
        if total <= best_score:
            best_score = total
            best_parent_set = cand_parent
            best_choice = ("call", "dpl", i, j - 1)
            best_cmap = cmap_mut

        # INCLUDE via dpr(i+1, j)
        pset, sc, _, cmap_sub = dpr(i + 1, j)
        cmap_mut = clone_state_map(cmap_sub)
        for c in children:
            if (a, b) in cmap_mut[c]["intervals"]:
                impute_interval_for_child(c, (a, b), cmap_mut)
                continue
            if laminar_ok_to_impute_for_child(c, (a, b), cmap_mut):
                impute_interval_for_child(c, (a, b), cmap_mut)
        extra_cnt = count_child_intervals_start_eq_i_end_le_j(i, j, cmap_mut, positions, children)
        miss = count_children_missing_interval(a, b, cmap_mut, children)
        c = positions[i + 1]
        bcost = boundary_cost_include_end(b, cmap_mut, children, Cb) + boundary_cost_exclude_end(c, cmap_mut, children, Cb)
        total = sc + Ct * extra_cnt + Ct * miss + bcost
        cand_parent = frozenset({(a, b)}) | pset
        if total <= best_score:
            best_score = total
            best_parent_set = cand_parent
            best_choice = ("call", "dpr", i + 1, j)
            best_cmap = cmap_mut

        # splits
        for k in range(i + 1, j):
            pk = positions[k]
            if (pk in start_boundaries) and (pk in end_boundaries):
                lp, lsc, _, lcmap = dpl(i, k)
                rp, rsc, _, rcmap = dpr(k, j)
                merged = merge_child_maps(lcmap, rcmap)
                cmap_mut = clone_state_map(merged)
                for c in children:
                    if (a, b) in cmap_mut[c]["intervals"]:
                        impute_interval_for_child(c, (a, b), cmap_mut)
                        continue
                    if laminar_ok_to_impute_for_child(c, (a, b), cmap_mut):
                        impute_interval_for_child(c, (a, b), cmap_mut)
                cross_cnt = count_child_intervals_crossing_k(i, k, j, cmap_mut, positions, children)
                miss = count_children_missing_interval(a, b, cmap_mut, children)
                bcost = boundary_cost_include_end(b, cmap_mut, children, Cb)
                total = lsc + rsc + Ct * cross_cnt + Ct * miss + bcost
                cand_parent = frozenset({(a, b)}) | lp | rp
                if total <= best_score:
                    best_score = total
                    best_parent_set = cand_parent
                    best_choice = ("split", [("dpl", i, k), ("dpr", k, j)])
                    best_cmap = cmap_mut

        # EXCLUDE branches
        pset, sc, _, cmap_sub = dpl(i, j - 1)
        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i, j, cmap_sub, positions, children)
        have = count_children_with_interval(a, b, cmap_sub, children)
        c = positions[j - 1]
        bcost = boundary_cost_exclude_end(b, cmap_sub, children, Cb) + boundary_cost_exclude_start(c, cmap_sub, children, Cb)
        total = sc + Ct * extra_cnt + Ct * have + bcost
        if total <= best_score:
            best_score = total
            best_parent_set = pset
            best_choice = ("call", "dpl", i, j - 1)
            best_cmap = cmap_sub

        pset, sc, _, cmap_sub = dp0(i + 1, j)
        extra_cnt = count_child_intervals_start_eq_i_end_le_j(i, j, cmap_sub, positions, children)
        have = count_children_with_interval(a, b, cmap_sub, children)
        c = positions[i + 1]
        bcost = boundary_cost_exclude_end(c, cmap_sub, children, Cb)
        total = sc + Ct * extra_cnt + Ct * have + bcost
        if total <= best_score:
            best_score = total
            best_parent_set = pset
            best_choice = ("call", "dp0", i + 1, j)
            best_cmap = cmap_sub

        for k in range(i + 1, j):
            pk = positions[k]
            if (pk in start_boundaries) and (pk in end_boundaries):
                lp, lsc, _, lcmap = dpl(i, k)
                rp, rsc, _, rcmap = dp0(k, j)
                merged = merge_child_maps(lcmap, rcmap)
                cross_cnt = count_child_intervals_crossing_k(i, k, j, merged, positions, children)
                have = count_children_with_interval(a, b, merged, children)
                total = lsc + rsc + Ct * cross_cnt + Ct * have
                cand_parent = lp | rp
                if total <= best_score:
                    best_score = total
                    best_parent_set = cand_parent
                    best_choice = ("split", [("dpl", i, k), ("dp0", k, j)])
                    best_cmap = merged

        dpl_done[i][j] = True
        dpl_val[i][j] = (best_parent_set, best_score, best_choice, best_cmap)
        return dpl_val[i][j]

    def dpr(i, j):
        if i >= j:
            return frozenset(), inf, None, initial_children_map
        if dpr_done[i][j]:
            return dpr_val[i][j]

        a = positions[i]
        b = positions[j]
        if a not in start_boundaries or b not in end_boundaries:
            dpr_done[i][j] = True
            dpr_val[i][j] = (frozenset(), inf, None, initial_children_map)
            return dpr_val[i][j]

        if j == i + 1:
            cmap = clone_state_map(initial_children_map)
            for c in children:
                if (a, b) in cmap[c]["intervals"]:
                    impute_interval_for_child(c, (a, b), cmap)
                    continue
                if laminar_ok_to_impute_for_child(c, (a, b), cmap):
                    impute_interval_for_child(c, (a, b), cmap)
            miss = count_children_missing_interval(a, b, cmap, children)
            bcost = boundary_cost_include_start(a, cmap, children, Cb)
            include_score = Ct * miss + bcost
            include_parent = frozenset({(a, b)})

            cmap_ex = initial_children_map
            have = count_children_with_interval(a, b, cmap_ex, children)
            bcost_ex = boundary_cost_exclude_start(a, cmap_ex, children, Cb)
            exclude_score = Ct * have + bcost_ex
            exclude_parent = frozenset()

            if include_score <= exclude_score:
                res = (include_parent, include_score, ("base", "INCLUDE"), cmap)
            else:
                res = (exclude_parent, exclude_score, ("base", "EXCLUDE"), cmap_ex)

            dpr_done[i][j] = True
            dpr_val[i][j] = res
            return res

        best_score = inf
        best_parent_set = frozenset()
        best_choice = None
        best_cmap = initial_children_map

        # INCLUDE via dpl(i, j-1)
        pset, sc, _, cmap_sub = dpl(i, j - 1)
        cmap_mut = clone_state_map(cmap_sub)
        for c in children:
            if (a, b) in cmap_mut[c]["intervals"]:
                impute_interval_for_child(c, (a, b), cmap_mut)
                continue
            if laminar_ok_to_impute_for_child(c, (a, b), cmap_mut):
                impute_interval_for_child(c, (a, b), cmap_mut)
        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i, j, cmap_mut, positions, children)
        miss = count_children_missing_interval(a, b, cmap_mut, children)
        c = positions[j - 1]
        bcost = boundary_cost_include_start(a, cmap_mut, children, Cb) + boundary_cost_exclude_start(c, cmap_mut, children, Cb)
        total = sc + Ct * extra_cnt + Ct * miss + bcost
        cand_parent = frozenset({(a, b)}) | pset
        if total <= best_score:
            best_score = total
            best_parent_set = cand_parent
            best_choice = ("call", "dpl", i, j - 1)
            best_cmap = cmap_mut

        # INCLUDE via dpr(i+1, j)
        pset, sc, _, cmap_sub = dpr(i + 1, j)
        cmap_mut = clone_state_map(cmap_sub)
        for c in children:
            if (a, b) in cmap_mut[c]["intervals"]:
                impute_interval_for_child(c, (a, b), cmap_mut)
                continue
            if laminar_ok_to_impute_for_child(c, (a, b), cmap_mut):
                impute_interval_for_child(c, (a, b), cmap_mut)
        extra_cnt = count_child_intervals_start_eq_i_end_le_j(i, j, cmap_mut, positions, children)
        miss = count_children_missing_interval(a, b, cmap_mut, children)
        c = positions[i + 1]
        bcost = boundary_cost_include_start(a, cmap_mut, children, Cb) + boundary_cost_exclude_end(c, cmap_mut, children, Cb)
        total = sc + Ct * extra_cnt + Ct * miss + bcost
        cand_parent = frozenset({(a, b)}) | pset
        if total <= best_score:
            best_score = total
            best_parent_set = cand_parent
            best_choice = ("call", "dpr", i + 1, j)
            best_cmap = cmap_mut

        # splits
        for k in range(i + 1, j):
            pk = positions[k]
            if (pk in start_boundaries) and (pk in end_boundaries):
                lp, lsc, _, lcmap = dpl(i, k)
                rp, rsc, _, rcmap = dpr(k, j)
                merged = merge_child_maps(lcmap, rcmap)
                cmap_mut = clone_state_map(merged)
                for c in children:
                    if (a, b) in cmap_mut[c]["intervals"]:
                        impute_interval_for_child(c, (a, b), cmap_mut)
                        continue
                    if laminar_ok_to_impute_for_child(c, (a, b), cmap_mut):
                        impute_interval_for_child(c, (a, b), cmap_mut)
                cross_cnt = count_child_intervals_crossing_k(i, k, j, cmap_mut, positions, children)
                miss = count_children_missing_interval(a, b, cmap_mut, children)
                bcost = boundary_cost_include_start(a, cmap_mut, children, Cb)
                total = lsc + rsc + Ct * cross_cnt + Ct * miss + bcost
                cand_parent = frozenset({(a, b)}) | lp | rp
                if total <= best_score:
                    best_score = total
                    best_parent_set = cand_parent
                    best_choice = ("split", [("dpl", i, k), ("dpr", k, j)])
                    best_cmap = cmap_mut

        # EXCLUDE branches
        pset, sc, _, cmap_sub = dp0(i, j - 1)
        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i, j, cmap_sub, positions, children)
        have = count_children_with_interval(a, b, cmap_sub, children)
        c = positions[j - 1]
        bcost = boundary_cost_exclude_start(c, cmap_sub, children, Cb)
        total = sc + Ct * extra_cnt + Ct * have + bcost
        if total <= best_score:
            best_score = total
            best_parent_set = pset
            best_choice = ("call", "dp0", i, j - 1)
            best_cmap = cmap_sub

        pset, sc, _, cmap_sub = dp0(i + 1, j)
        extra_cnt = count_child_intervals_start_eq_i_end_le_j(i, j, cmap_sub, positions, children)
        have = count_children_with_interval(a, b, cmap_sub, children)
        c = positions[i + 1]
        bcost = boundary_cost_exclude_start(a, cmap_sub, children, Cb) + boundary_cost_exclude_end(c, cmap_sub, children, Cb)
        total = sc + Ct * extra_cnt + Ct * have + bcost
        if total <= best_score:
            best_score = total
            best_parent_set = pset
            best_choice = ("call", "dp0", i + 1, j)
            best_cmap = cmap_sub

        for k in range(i + 1, j):
            pk = positions[k]
            if (pk in start_boundaries) and (pk in end_boundaries):
                lp, lsc, _, lcmap = dp0(i, k)
                rp, rsc, _, rcmap = dpr(k, j)
                merged = merge_child_maps(lcmap, rcmap)
                cross_cnt = count_child_intervals_crossing_k(i, k, j, merged, positions, children)
                have = count_children_with_interval(a, b, merged, children)
                total = lsc + rsc + Ct * cross_cnt + Ct * have
                cand_parent = lp | rp
                if total <= best_score:
                    best_score = total
                    best_parent_set = cand_parent
                    best_choice = ("split", [("dp0", i, k), ("dpr", k, j)])
                    best_cmap = merged

        dpr_done[i][j] = True
        dpr_val[i][j] = (best_parent_set, best_score, best_choice, best_cmap)
        return dpr_val[i][j]

    def traceback(state, i, j, path):
        if i >= j:
            return
        if state == "dp0":
            if not dp0_done[i][j]:
                return
            parent_set, score, choice, cmap = dp0_val[i][j]
        elif state == "dpl":
            if not dpl_done[i][j]:
                return
            parent_set, score, choice, cmap = dpl_val[i][j]
        elif state == "dpr":
            if not dpr_done[i][j]:
                return
            parent_set, score, choice, cmap = dpr_val[i][j]
        else:
            return

        a = positions[i]
        b = positions[j]
        path.append(
            {
                "state": f"{state}[{i},{j}] ({a},{b})",
                "choice": choice,
                "intervals": set(parent_set),
                "score": score,
                "children_map": deepcopy(cmap),
            }
        )

        if not choice:
            return

        typ = choice[0]
        if typ == "call":
            _, func_name, ni, nj = choice
            traceback(func_name, ni, nj, path)
        elif typ == "split":
            _, list_calls = choice
            for func_name, ni, nj in list_calls:
                traceback(func_name, ni, nj, path)
        elif typ == "base":
            return

    root_parent_set, root_score, _, root_children_map = dp0(0, n - 1)
    if trace:
        optimal_path = []
        traceback("dp0", 0, n - 1, optimal_path)
        return set(root_parent_set), root_score, root_children_map, optimal_path
    return set(root_parent_set), root_score, root_children_map


# -----------------------------------------------------------------------------
# Helper utilities / scoring
# -----------------------------------------------------------------------------

def generate_star_topology(num_children):
    edges = [(0, i) for i in range(1, num_children + 1)]
    leaf_nodes = list(range(1, num_children + 1))
    return edges, leaf_nodes


def extract_boundaries(leaf_intervals):
    start_boundaries = set()
    end_boundaries = set()
    for node_intervals in leaf_intervals.values():
        for entry in node_intervals:
            interval, _ = parse_weighted_interval_entry(entry)
            a, b = interval
            start_boundaries.add(a)
            start_boundaries.add(b)
            end_boundaries.add(a)
            end_boundaries.add(b)
    return sorted(start_boundaries), sorted(end_boundaries)


def calculate_parsimony_score_from_state(
    root_intervals,
    children_state,
    Ct=1.0,
    Cb=1.0,
):
    """Weighted parsimony score using child interval/boundary weights."""
    total_score = 0.0
    root_starts = {a for (a, b) in root_intervals}
    root_ends = {b for (a, b) in root_intervals}

    for child_state in children_state.values():
        child_intervals = child_state.get("intervals", {})
        child_starts = child_state.get("starts", {})
        child_ends = child_state.get("ends", {})

        all_intervals = set(root_intervals) | set(child_intervals)
        for interval in all_intervals:
            p = 1.0 if interval in root_intervals else 0.0
            q = float(child_intervals.get(interval, 0.0))
            total_score += Ct * abs(p - q)

        all_starts = root_starts | set(child_starts)
        for pos in all_starts:
            p = 1.0 if pos in root_starts else 0.0
            q = float(child_starts.get(pos, 0.0))
            total_score += Cb * abs(p - q)

        all_ends = root_ends | set(child_ends)
        for pos in all_ends:
            p = 1.0 if pos in root_ends else 0.0
            q = float(child_ends.get(pos, 0.0))
            total_score += Cb * abs(p - q)

    return total_score


def calculate_parsimony_score(root_intervals, edges, leaf_intervals, leaf_start_weights=None, leaf_end_weights=None, Ct=1.0, Cb=1.0):
    children_state = build_leaf_state(edges, leaf_intervals, leaf_start_weights=leaf_start_weights, leaf_end_weights=leaf_end_weights)
    return calculate_parsimony_score_from_state(root_intervals, children_state, Ct=Ct, Cb=Cb)


# -----------------------------------------------------------------------------
# Lightweight weighted testing helpers
# -----------------------------------------------------------------------------

def make_weighted_from_unweighted_leaf_intervals(leaf_intervals, low=0.25, high=1.0):
    """Convert plain interval lists into weighted interval lists plus derived boundary weights."""
    weighted = {}
    for c, entries in leaf_intervals.items():
        weighted[c] = []
        for entry in entries:
            interval, _ = parse_weighted_interval_entry(entry)
            w = round(random.uniform(low, high), 3)
            weighted[c].append((interval, w))
    return weighted


# -----------------------------------------------------------------------------
# Main example
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Example with weighted intervals. Boundary weights are derived automatically
    # from the weighted intervals unless you pass leaf_start_weights / leaf_end_weights.
    edges = [(0, 1), (0, 2), (0, 3)]
    leaf_intervals = {
        1: [([1,10], 0.8), ([4, 10], 0.8), ([1, 3], 0.8), ([3, 10], 0.8)],
        2: [([1, 5], 0.8), ([2, 5], 0.9)],
        3: [([2, 4], 0.6), ([2, 5], 0.6)],
    }

    # Optional explicit boundary weights if you want to override the defaults.
    # If omitted, they are derived from interval weights.
    leaf_start_weights = {
        1: {1: 0.8, 3: 0.8, 4: 0.8, 10: 0.8},
        2: {1: 0.8, 2: 0.9, 5: 0.9},
        3: {2: 0.8, 4: 0.8, 5: 0.3},
    }
    leaf_end_weights = {
        1: {1: 0.8, 3: 0.8, 4: 0.8, 10: 0.8},
        2: {1: 0.8, 2: 0.9, 5: 0.9},
        3: {2: 0.8, 4: 0.8, 5: 0.3},
    }

    start_boundaries, end_boundaries = extract_boundaries(leaf_intervals)

    dp_intervals, dp_score, dp_children_map = infer_tads_with_dp(
        edges,
        leaf_intervals,
        start_boundaries,
        end_boundaries,
        leaf_start_weights=leaf_start_weights,
        leaf_end_weights=leaf_end_weights,
        Ct=1.0,
        Cb=1.0,
        trace=False,
    )

    print(f"DP intervals: {sorted(dp_intervals)} with score {dp_score:.6f}")
    for child in sorted(dp_children_map):
        print(f"Child {child} state: {dp_children_map[child]}")

    # Threshold the weighted input into an unweighted instance: weights < 0.5 -> 0,
    # weights >= 0.5 -> 1. Boundaries for the unweighted run are derived from these
    # thresholded intervals.
    from nussinov_with_imputation import infer_tads_with_dp as infer_unweighted_dp

    thresholded_leaf_intervals = {}
    for child, entries in leaf_intervals.items():
        thresholded_leaf_intervals[child] = []
        for entry in entries:
            interval, weight = parse_weighted_interval_entry(entry)
            if float(weight) >= 0.5:
                thresholded_leaf_intervals[child].append(f"[{interval[0]},{interval[1]}]")

    unweighted_start_boundaries, unweighted_end_boundaries = extract_boundaries(thresholded_leaf_intervals)
    unweighted_intervals, unweighted_score, unweighted_children_map = infer_unweighted_dp(
        edges,
        thresholded_leaf_intervals,
        unweighted_start_boundaries,
        unweighted_end_boundaries,
        Ct=1.0,
        Cb=1.0,
        trace=False,
    )

    unweighted_dp_intervals = set(unweighted_intervals)
    unweighted_dp_children_map = {child: set(intervals) for child, intervals in unweighted_children_map.items()}
    weighted_dp_children_map = {
        child: set(state.get("intervals", {}).keys())
        for child, state in dp_children_map.items()
    }

    intervals_match = set(dp_intervals) == unweighted_dp_intervals
    children_match = weighted_dp_children_map == unweighted_dp_children_map

    print("\nThresholded comparison against unweighted solver")
    print(f"  intervals_match: {intervals_match}")
    print(f"  children_match:   {children_match}")
    print(f"  weighted_score:   {dp_score:.6f}")
    print(f"  unweighted_score: {unweighted_score:.6f}")
    print(f"Unweighted parent intervals: {sorted(unweighted_dp_intervals)}")
    for child in sorted(unweighted_dp_children_map):
        print(f"Unweighted child {child} intervals: {sorted(unweighted_dp_children_map[child])}")
    if intervals_match and children_match:
        print("\nMatched parent/child intervals:")
        print(f"Weighted parent intervals:   {sorted(dp_intervals)}")
        for child in sorted(weighted_dp_children_map):
            print(f"Weighted child {child} intervals: {sorted(weighted_dp_children_map[child])}")
    else:
        only_weighted = sorted(set(dp_intervals) - unweighted_dp_intervals)
        only_unweighted = sorted(unweighted_dp_intervals - set(dp_intervals))
        print(f"  only_in_weighted:   {only_weighted}")
        print(f"  only_in_unweighted: {only_unweighted}")
