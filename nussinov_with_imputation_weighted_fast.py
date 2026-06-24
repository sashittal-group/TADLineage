import random
import re
from bisect import bisect_left
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

# Optional external dependency used by your older discrete LP experiments.
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
WeightedIntervalEntry = Union[
    str,
    Tuple[Union[str, Sequence[int], Interval], float],
    List[Union[str, Sequence[int], Interval, float]],
]


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


def _insert_sorted_unique(xs: List[int], value: int) -> None:
    """Insert value into sorted list xs if not already present."""
    idx = bisect_left(xs, value)
    if idx >= len(xs) or xs[idx] != value:
        xs.insert(idx, value)


def _copy_child_state(child_state: Dict[str, Dict]) -> Dict[str, Dict]:
    """Copy one child state (used only on write)."""
    return {
        "intervals": dict(child_state.get("intervals", {})),
        "interval_keys": frozenset(child_state.get("interval_keys", child_state.get("intervals", {}).keys())),
        "starts": dict(child_state.get("starts", {})),
        "ends": dict(child_state.get("ends", {})),
        "by_start": {k: list(v) for k, v in child_state.get("by_start", {}).items()},
        "by_end": {k: list(v) for k, v in child_state.get("by_end", {}).items()},
        "start_keys": list(child_state.get("start_keys", [])),
        "end_keys": list(child_state.get("end_keys", [])),
    }


def build_leaf_state(
    edges: List[Tuple[int, int]],
    leaf_intervals: Dict[int, Sequence[WeightedIntervalEntry]],
    leaf_start_weights: Optional[Dict[int, Dict[int, float]]] = None,
    leaf_end_weights: Optional[Dict[int, Dict[int, float]]] = None,
) -> Dict[int, Dict[str, Dict]]:
    """Build a per-child state with:
        state[c]['intervals'] = {(a,b): w}
        state[c]['interval_keys'] = frozenset({(a,b), ...})
        state[c]['starts'] = {a: w}
        state[c]['ends'] = {b: w}
        state[c]['by_start'][a] = list of intervals starting at a
        state[c]['by_end'][b]   = list of intervals ending at b
        state[c]['start_keys']  = sorted list of start positions
        state[c]['end_keys']    = sorted list of end positions

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
        intervals = dict(interval_weights.get(c, {}))
        starts = dict(start_weights.get(c, {}))
        ends = dict(end_weights.get(c, {}))

        by_start: Dict[int, List[Interval]] = {}
        by_end: Dict[int, List[Interval]] = {}
        for interval in intervals:
            a, b = interval
            by_start.setdefault(a, []).append(interval)
            by_end.setdefault(b, []).append(interval)

        state[c] = {
            "intervals": intervals,
            "interval_keys": frozenset(intervals.keys()),
            "starts": starts,
            "ends": ends,
            "by_start": by_start,
            "by_end": by_end,
            "start_keys": sorted(starts.keys()),
            "end_keys": sorted(ends.keys()),
        }
    return state


def clone_state_map(state_map: Dict[int, Dict[str, Dict]]) -> Dict[int, Dict[str, Dict]]:
    """Cheap outer copy; child dicts are copied lazily on write."""
    return dict(state_map)


def snapshot_state_map(state_map: Dict[int, Dict[str, Dict]]) -> Dict[int, Dict[str, Dict]]:
    """Deep-enough snapshot for traceback/output."""
    return {
        c: {
            "intervals": dict(state.get("intervals", {})),
            "interval_keys": frozenset(state.get("interval_keys", state.get("intervals", {}).keys())),
            "starts": dict(state.get("starts", {})),
            "ends": dict(state.get("ends", {})),
            "by_start": {k: list(v) for k, v in state.get("by_start", {}).items()},
            "by_end": {k: list(v) for k, v in state.get("by_end", {}).items()},
            "start_keys": list(state.get("start_keys", [])),
            "end_keys": list(state.get("end_keys", [])),
        }
        for c, state in state_map.items()
    }


def merge_child_maps(
    m1: Dict[int, Dict[str, Dict]],
    m2: Dict[int, Dict[str, Dict]],
) -> Dict[int, Dict[str, Dict]]:
    """Merge two child-state maps by taking the max weight for each item."""
    merged: Dict[int, Dict[str, Dict]] = {}
    keys = set(m1.keys()) | set(m2.keys())
    for c in keys:
        d1 = m1.get(c, {})
        d2 = m2.get(c, {})
        intervals: Dict[Interval, float] = {}
        starts: Dict[int, float] = {}
        ends: Dict[int, float] = {}

        for src in (d1, d2):
            for k, v in src.get("intervals", {}).items():
                vv = float(v)
                if vv > intervals.get(k, 0.0):
                    intervals[k] = vv
            for k, v in src.get("starts", {}).items():
                vv = float(v)
                if vv > starts.get(k, 0.0):
                    starts[k] = vv
            for k, v in src.get("ends", {}).items():
                vv = float(v)
                if vv > ends.get(k, 0.0):
                    ends[k] = vv

        by_start: Dict[int, List[Interval]] = {}
        by_end: Dict[int, List[Interval]] = {}
        for interval in intervals:
            a, b = interval
            by_start.setdefault(a, []).append(interval)
            by_end.setdefault(b, []).append(interval)

        merged[c] = {
            "intervals": intervals,
            "interval_keys": frozenset(intervals.keys()),
            "starts": starts,
            "ends": ends,
            "by_start": by_start,
            "by_end": by_end,
            "start_keys": sorted(starts.keys()),
            "end_keys": sorted(ends.keys()),
        }
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


def sum_children_with_interval(a: int, b: int, cmap: Dict[int, Dict[str, Dict]], children: Sequence[int]) -> float:
    interval = (a, b)
    total = 0.0
    for c in children:
        total += get_interval_weight(c, interval, cmap)
    return total


def count_children_missing_interval(a: int, b: int, cmap: Dict[int, Dict[str, Dict]], children: Sequence[int]) -> float:
    return float(len(children)) - sum_children_with_interval(a, b, cmap, children)


def sum_children_with_start(pos: int, cmap: Dict[int, Dict[str, Dict]], children: Sequence[int]) -> float:
    total = 0.0
    for c in children:
        total += get_start_weight(c, pos, cmap)
    return total


def sum_children_with_end(pos: int, cmap: Dict[int, Dict[str, Dict]], children: Sequence[int]) -> float:
    total = 0.0
    for c in children:
        total += get_end_weight(c, pos, cmap)
    return total


def count_child_intervals_start_ge_i_end_eq_j(
    i: int,
    j: int,
    cmap: Dict[int, Dict[str, Dict]],
    positions: List[int],
    children: Sequence[int],
) -> float:
    """Count weighted intervals with start >= positions[i] and end == positions[j]."""
    a = positions[i]
    b = positions[j]
    cnt = 0.0
    for c in children:
        by_end = cmap.get(c, {}).get("by_end", {})
        for x, y in by_end.get(b, []):
            if (x, y) == (a, b):
                continue
            if x >= a:
                cnt += float(cmap[c]["intervals"][(x, y)])
    return cnt


def count_child_intervals_start_eq_i_end_le_j(
    i: int,
    j: int,
    cmap: Dict[int, Dict[str, Dict]],
    positions: List[int],
    children: Sequence[int],
) -> float:
    """Count weighted intervals with start == positions[i] and end <= positions[j]."""
    a = positions[i]
    b = positions[j]
    cnt = 0.0
    for c in children:
        by_start = cmap.get(c, {}).get("by_start", {})
        for x, y in by_start.get(a, []):
            if (x, y) == (a, b):
                continue
            if y <= b:
                cnt += float(cmap[c]["intervals"][(x, y)])
    return cnt


def count_child_intervals_crossing_k(
    i: int,
    k: int,
    j: int,
    cmap: Dict[int, Dict[str, Dict]],
    positions: List[int],
    children: Sequence[int],
) -> float:
    """Count weighted intervals crossing the split point positions[k]."""
    a = positions[i]
    pk = positions[k]
    b = positions[j]
    cnt = 0.0
    for c in children:
        child = cmap.get(c, {})
        by_start = child.get("by_start", {})
        start_keys = child.get("start_keys", [])
        lo = bisect_left(start_keys, a)
        hi = bisect_left(start_keys, pk)
        for x in start_keys[lo:hi]:
            for xx, yy in by_start.get(x, []):
                if (xx, yy) == (a, b):
                    continue
                if pk < yy <= b:
                    cnt += float(child["intervals"][(xx, yy)])
    return cnt


def boundary_cost_include_start(a_pos: int, cmap: Dict[int, Dict[str, Dict]], children: Sequence[int], Cb: float) -> float:
    return Cb * (float(len(children)) - sum_children_with_start(a_pos, cmap, children))


def boundary_cost_include_end(b_pos: int, cmap: Dict[int, Dict[str, Dict]], children: Sequence[int], Cb: float) -> float:
    return Cb * (float(len(children)) - sum_children_with_end(b_pos, cmap, children))


def boundary_cost_exclude_start(a_pos: int, cmap: Dict[int, Dict[str, Dict]], children: Sequence[int], Cb: float) -> float:
    return Cb * sum_children_with_start(a_pos, cmap, children)


def boundary_cost_exclude_end(b_pos: int, cmap: Dict[int, Dict[str, Dict]], children: Sequence[int], Cb: float) -> float:
    return Cb * sum_children_with_end(b_pos, cmap, children)


def laminar_ok_to_impute_for_child(c: int, interval: Interval, cmap: Dict[int, Dict[str, Dict]]) -> bool:
    """Check whether adding interval to child c would preserve laminarity."""
    a, b = interval
    child_state = cmap.get(c, {})
    interval_keys = child_state.get("interval_keys")
    if interval_keys is None:
        interval_keys = child_state.get("intervals", {}).keys()

    for x, y in interval_keys:
        if (x, y) == (a, b):
            continue
        if (x < a < y < b) or (a < x < b < y):
            return False
    return True


def impute_interval_for_child(c: int, interval: Interval, cmap: Dict[int, Dict[str, Dict]]) -> None:
    """Add missing interval/boundary entries for child c.

    This matches nussinov_with_imputation_weighted.py: existing weighted
    interval or boundary values are preserved rather than promoted to 1.0.
    """
    a, b = interval
    child_state = cmap.get(c)

    if child_state is None:
        child_state = {
            "intervals": {},
            "interval_keys": frozenset(),
            "starts": {},
            "ends": {},
            "by_start": {},
            "by_end": {},
            "start_keys": [],
            "end_keys": [],
        }
        cmap[c] = child_state
    else:
        interval_exists = interval in child_state.get("intervals", {})
        start_exists = child_state.get("starts", {}).get(a, 0.0) > 0.0
        end_exists = child_state.get("ends", {}).get(b, 0.0) > 0.0
        if interval_exists and start_exists and end_exists:
            return

        child_state = _copy_child_state(child_state)
        cmap[c] = child_state

    if interval not in child_state["intervals"]:
        child_state["intervals"][interval] = 1.0
        child_state["interval_keys"] = child_state["interval_keys"] | {interval}
        child_state["by_start"].setdefault(a, []).append(interval)
        child_state["by_end"].setdefault(b, []).append(interval)

    if child_state["starts"].get(a, 0.0) <= 0.0:
        child_state["starts"][a] = 1.0
        _insert_sorted_unique(child_state["start_keys"], a)
    if child_state["ends"].get(b, 0.0) <= 0.0:
        child_state["ends"][b] = 1.0
        _insert_sorted_unique(child_state["end_keys"], b)


def impute_interval_across_children(
    interval: Interval,
    cmap: Dict[int, Dict[str, Dict]],
    children: Sequence[int],
) -> None:
    """Impute an interval into every compatible child."""
    for c in children:
        if interval in cmap.get(c, {}).get("intervals", {}):
            impute_interval_for_child(c, interval, cmap)
        elif laminar_ok_to_impute_for_child(c, interval, cmap):
            impute_interval_for_child(c, interval, cmap)


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
    """Weighted Nussinov-style DP with explicit child imputation."""
    children = tuple(v for (u, v) in edges if u == 0)
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

    start_boundary_set = set(start_boundaries)
    end_boundary_set = set(end_boundaries)
    split_ok_positions = {
        pos for pos in positions if pos in start_boundary_set and pos in end_boundary_set
    }

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
        if a not in start_boundary_set or b not in end_boundary_set:
            dp0_done[i][j] = True
            dp0_val[i][j] = (frozenset(), inf, None, initial_children_map)
            return dp0_val[i][j]

        if j == i + 1:
            cmap = clone_state_map(initial_children_map)
            impute_interval_across_children((a, b), cmap, children)
            have = sum_children_with_interval(a, b, cmap, children)
            miss = float(len(children)) - have
            bcost = (
                boundary_cost_include_start(a, cmap, children, Cb)
                + boundary_cost_include_end(b, cmap, children, Cb)
            )
            include_score = Ct * miss + bcost
            include_parent = frozenset({(a, b)})

            cmap_ex = initial_children_map
            have_ex = sum_children_with_interval(a, b, cmap_ex, children)
            bcost_ex = (
                boundary_cost_exclude_start(a, cmap_ex, children, Cb)
                + boundary_cost_exclude_end(b, cmap_ex, children, Cb)
            )
            exclude_score = Ct * have_ex + bcost_ex
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

        # INCLUDE [a,b] branches
        pset, sc, _, cmap_sub = dpl(i, j - 1)
        cmap_mut = clone_state_map(cmap_sub)
        impute_interval_across_children((a, b), cmap_mut, children)
        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i, j, cmap_mut, positions, children)
        have = sum_children_with_interval(a, b, cmap_mut, children)
        miss = float(len(children)) - have
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

        pset, sc, _, cmap_sub = dpr(i + 1, j)
        cmap_mut = clone_state_map(cmap_sub)
        impute_interval_across_children((a, b), cmap_mut, children)
        extra_cnt = count_child_intervals_start_eq_i_end_le_j(i, j, cmap_mut, positions, children)
        have = sum_children_with_interval(a, b, cmap_mut, children)
        miss = float(len(children)) - have
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

        for k in range(i + 1, j):
            pk = positions[k]
            if pk in split_ok_positions:
                left_parent, left_score, _, left_cmap = dpl(i, k)
                right_parent, right_score, _, right_cmap = dpr(k, j)
                merged = merge_child_maps(left_cmap, right_cmap)
                cmap_mut = clone_state_map(merged)
                impute_interval_across_children((a, b), cmap_mut, children)
                cross_cnt = count_child_intervals_crossing_k(i, k, j, cmap_mut, positions, children)
                have = sum_children_with_interval(a, b, cmap_mut, children)
                miss = float(len(children)) - have
                bcost = boundary_cost_include_start(a, cmap_mut, children, Cb) + boundary_cost_include_end(b, cmap_mut, children, Cb)
                total = left_score + right_score + Ct * cross_cnt + Ct * miss + bcost
                cand_parent = frozenset({(a, b)}) | left_parent | right_parent
                if total <= best_score:
                    best_score = total
                    best_parent_set = cand_parent
                    best_choice = ("split", [("dpl", i, k), ("dpr", k, j)])
                    best_cmap = cmap_mut

        # EXCLUDE [a,b] branches
        pset, sc, _, cmap_sub = dp0(i, j - 1)
        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i, j, cmap_sub, positions, children)
        have = sum_children_with_interval(a, b, cmap_sub, children)
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
        have = sum_children_with_interval(a, b, cmap_sub, children)
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
            if pk in split_ok_positions:
                left_parent, left_score, _, left_cmap = dp0(i, k)
                right_parent, right_score, _, right_cmap = dp0(k, j)
                merged = merge_child_maps(left_cmap, right_cmap)
                cross_cnt = count_child_intervals_crossing_k(i, k, j, merged, positions, children)
                have = sum_children_with_interval(a, b, merged, children)
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
        if a not in start_boundary_set or b not in end_boundary_set:
            dpl_done[i][j] = True
            dpl_val[i][j] = (frozenset(), inf, None, initial_children_map)
            return dpl_val[i][j]

        if j == i + 1:
            cmap = clone_state_map(initial_children_map)
            impute_interval_across_children((a, b), cmap, children)
            have = sum_children_with_interval(a, b, cmap, children)
            miss = float(len(children)) - have
            bcost = boundary_cost_include_end(b, cmap, children, Cb)
            include_score = Ct * miss + bcost
            include_parent = frozenset({(a, b)})

            cmap_ex = initial_children_map
            have_ex = sum_children_with_interval(a, b, cmap_ex, children)
            bcost_ex = boundary_cost_exclude_end(b, cmap_ex, children, Cb)
            exclude_score = Ct * have_ex + bcost_ex
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
        impute_interval_across_children((a, b), cmap_mut, children)
        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i, j, cmap_mut, positions, children)
        have = sum_children_with_interval(a, b, cmap_mut, children)
        miss = float(len(children)) - have
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
        impute_interval_across_children((a, b), cmap_mut, children)
        extra_cnt = count_child_intervals_start_eq_i_end_le_j(i, j, cmap_mut, positions, children)
        have = sum_children_with_interval(a, b, cmap_mut, children)
        miss = float(len(children)) - have
        c = positions[i + 1]
        bcost = boundary_cost_include_end(b, cmap_mut, children, Cb) + boundary_cost_exclude_end(c, cmap_mut, children, Cb)
        total = sc + Ct * extra_cnt + Ct * miss + bcost
        cand_parent = frozenset({(a, b)}) | pset
        if total <= best_score:
            best_score = total
            best_parent_set = cand_parent
            best_choice = ("call", "dpr", i + 1, j)
            best_cmap = cmap_mut

        for k in range(i + 1, j):
            pk = positions[k]
            if pk in split_ok_positions:
                lp, lsc, _, lcmap = dpl(i, k)
                rp, rsc, _, rcmap = dpr(k, j)
                merged = merge_child_maps(lcmap, rcmap)
                cmap_mut = clone_state_map(merged)
                impute_interval_across_children((a, b), cmap_mut, children)
                cross_cnt = count_child_intervals_crossing_k(i, k, j, cmap_mut, positions, children)
                have = sum_children_with_interval(a, b, cmap_mut, children)
                miss = float(len(children)) - have
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
        have = sum_children_with_interval(a, b, cmap_sub, children)
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
        have = sum_children_with_interval(a, b, cmap_sub, children)
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
            if pk in split_ok_positions:
                lp, lsc, _, lcmap = dpl(i, k)
                rp, rsc, _, rcmap = dp0(k, j)
                merged = merge_child_maps(lcmap, rcmap)
                cross_cnt = count_child_intervals_crossing_k(i, k, j, merged, positions, children)
                have = sum_children_with_interval(a, b, merged, children)
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
        if a not in start_boundary_set or b not in end_boundary_set:
            dpr_done[i][j] = True
            dpr_val[i][j] = (frozenset(), inf, None, initial_children_map)
            return dpr_val[i][j]

        if j == i + 1:
            cmap = clone_state_map(initial_children_map)
            impute_interval_across_children((a, b), cmap, children)
            have = sum_children_with_interval(a, b, cmap, children)
            miss = float(len(children)) - have
            bcost = boundary_cost_include_start(a, cmap, children, Cb)
            include_score = Ct * miss + bcost
            include_parent = frozenset({(a, b)})

            cmap_ex = initial_children_map
            have_ex = sum_children_with_interval(a, b, cmap_ex, children)
            bcost_ex = boundary_cost_exclude_start(a, cmap_ex, children, Cb)
            exclude_score = Ct * have_ex + bcost_ex
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
        impute_interval_across_children((a, b), cmap_mut, children)
        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i, j, cmap_mut, positions, children)
        have = sum_children_with_interval(a, b, cmap_mut, children)
        miss = float(len(children)) - have
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
        impute_interval_across_children((a, b), cmap_mut, children)
        extra_cnt = count_child_intervals_start_eq_i_end_le_j(i, j, cmap_mut, positions, children)
        have = sum_children_with_interval(a, b, cmap_mut, children)
        miss = float(len(children)) - have
        c = positions[i + 1]
        bcost = boundary_cost_include_start(a, cmap_mut, children, Cb) + boundary_cost_exclude_end(c, cmap_mut, children, Cb)
        total = sc + Ct * extra_cnt + Ct * miss + bcost
        cand_parent = frozenset({(a, b)}) | pset
        if total <= best_score:
            best_score = total
            best_parent_set = cand_parent
            best_choice = ("call", "dpr", i + 1, j)
            best_cmap = cmap_mut

        for k in range(i + 1, j):
            pk = positions[k]
            if pk in split_ok_positions:
                lp, lsc, _, lcmap = dpl(i, k)
                rp, rsc, _, rcmap = dpr(k, j)
                merged = merge_child_maps(lcmap, rcmap)
                cmap_mut = clone_state_map(merged)
                impute_interval_across_children((a, b), cmap_mut, children)
                cross_cnt = count_child_intervals_crossing_k(i, k, j, cmap_mut, positions, children)
                have = sum_children_with_interval(a, b, cmap_mut, children)
                miss = float(len(children)) - have
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
        have = sum_children_with_interval(a, b, cmap_sub, children)
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
        have = sum_children_with_interval(a, b, cmap_sub, children)
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
            if pk in split_ok_positions:
                lp, lsc, _, lcmap = dp0(i, k)
                rp, rsc, _, rcmap = dpr(k, j)
                merged = merge_child_maps(lcmap, rcmap)
                cross_cnt = count_child_intervals_crossing_k(i, k, j, merged, positions, children)
                have = sum_children_with_interval(a, b, merged, children)
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
                "children_map": snapshot_state_map(cmap),
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
    edges = [(0, 1), (0, 2), (0, 3)]
    leaf_intervals = {
        1: [([1, 10], 0.8), ([4, 10], 0.8), ([1, 3], 0.8), ([3, 10], 0.8)],
        2: [([1, 5], 0.8), ([2, 5], 0.9)],
        3: [([2, 4], 0.6), ([2, 5], 0.6)],
    }

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
