from collections import defaultdict, deque
from typing import List, Tuple, Set, Dict, Optional
import random
from tad_utilities import match_tads_within_tolerance, relaxed_boundary_rf, relaxed_tad_rf

Interval = Tuple[int, int]


def _match_boundary_values_with_tolerance(source_values, target_values, tolerance):
    """Greedily match unique boundary values within tolerance."""
    source_sorted = sorted(set(source_values))
    target_sorted = sorted(set(target_values))

    matched_source = set()
    matched_target = set()
    i = j = 0

    while i < len(source_sorted) and j < len(target_sorted):
        source_value = source_sorted[i]
        target_value = target_sorted[j]

        if abs(source_value - target_value) <= tolerance:
            matched_source.add(source_value)
            matched_target.add(target_value)
            i += 1
            j += 1
        elif source_value < target_value - tolerance:
            i += 1
        else:
            j += 1

    return matched_source, matched_target


def _compatible_split(source_iv: Interval, left: Interval, right: Interval, tolerance: int) -> bool:
    a, b = source_iv
    left_a, left_b = left
    right_a, right_b = right
    return (
        abs(a - left_a) <= tolerance
        and abs(b - right_b) <= tolerance
        and left_b <= right_a
    )


def _compatible_merge(left: Interval, right: Interval, target_iv: Interval, tolerance: int) -> bool:
    a, b = target_iv
    left_a, left_b = left
    right_a, right_b = right
    return (
        abs(a - left_a) <= tolerance
        and abs(b - right_b) <= tolerance
        and left_b <= right_a
    )


class DinicEdge:
    __slots__ = ("to", "rev", "cap")

    def __init__(self, to: int, rev: int, cap: int):
        self.to = to
        self.rev = rev
        self.cap = cap


class Dinic:
    def __init__(self, n: int):
        self.n = n
        self.g = [[] for _ in range(n)]
        self.level = [0] * n
        self.it = [0] * n

    def add_edge(self, fr: int, to: int, cap: int) -> DinicEdge:
        fwd = DinicEdge(to, len(self.g[to]), cap)
        rev = DinicEdge(fr, len(self.g[fr]), 0)
        self.g[fr].append(fwd)
        self.g[to].append(rev)
        return fwd

    def bfs(self, s: int, t: int) -> bool:
        self.level = [-1] * self.n
        q = deque([s])
        self.level[s] = 0
        while q:
            v = q.popleft()
            for e in self.g[v]:
                if e.cap > 0 and self.level[e.to] < 0:
                    self.level[e.to] = self.level[v] + 1
                    q.append(e.to)
        return self.level[t] >= 0

    def dfs(self, v: int, t: int, f: int) -> int:
        if v == t:
            return f
        for i in range(self.it[v], len(self.g[v])):
            self.it[v] = i
            e = self.g[v][i]
            if e.cap > 0 and self.level[v] < self.level[e.to]:
                d = self.dfs(e.to, t, min(f, e.cap))
                if d > 0:
                    e.cap -= d
                    self.g[e.to][e.rev].cap += d
                    return d
        return 0

    def max_flow(self, s: int, t: int) -> int:
        flow = 0
        INF = 10**18
        while self.bfs(s, t):
            self.it = [0] * self.n
            while True:
                f = self.dfs(s, t, INF)
                if f == 0:
                    break
                flow += f
        return flow


def transform_sequence(source_intervals: List[Interval],
                       target_intervals: List[Interval],
                       Ct: float = 1.0,
                       Cb: float = 1.0,
                       tolerance: int = 0):
    """
    Transform `source_intervals` into `target_intervals` while grouping eligible
    operations into split/merge events.

    Flow-based selection step:
    1) Remove common TADs first.
    2) Enumerate all valid split/merge trios.
    3) Build a max-flow network where each candidate event is a path.
    4) Select a maximum set of disjoint events.
    5) Apply the selected split/merge events.
    6) Delete all remaining source-only TADs.
    7) Insert all remaining target-only TADs.

    A split is:
        source [a,b] -> target [a,x] and [y,b], with x <= y
    A merge is:
        source [a,x] and [y,b] -> target [a,b], with x <= y

    Returns
    -------
    operations : list of dict
        Each dict is one of {'split', 'merge', 'delete', 'insert'}.
    total_cost : float
    final_intervals : set of intervals
    """
    tolerance = max(0, int(tolerance))

    source_set = set(source_intervals)
    target_set = set(target_intervals)
    common_pairs, unmatched_source, unmatched_target = match_tads_within_tolerance(source_set, target_set, tolerance)
    common = {src for src, _ in common_pairs}
    matched_target = {tgt for _, tgt in common_pairs}

    # Work on reduced sets after removing common TADs (fixed/shared part)
    source_reduced_set = set(unmatched_source)
    target_reduced_set = set(unmatched_target)

    # Global boundary sets (must be computed on full source/target)
    # so common TADs still protect shared boundaries from being re-charged.
    source_starts_global = {a for (a, _) in source_set}
    source_ends_global = {b for (_, b) in source_set}
    target_starts_global = {a for (a, _) in target_set}
    target_ends_global = {b for (_, b) in target_set}

    # Only these boundaries should contribute to RF boundary cost
    matched_source_starts, matched_target_starts = _match_boundary_values_with_tolerance(
        source_starts_global, target_starts_global, tolerance
    )
    matched_source_ends, matched_target_ends = _match_boundary_values_with_tolerance(
        source_ends_global, target_ends_global, tolerance
    )

    # Current state starts from reduced source (common TADs are treated as fixed)
    current_intervals: Set[Interval] = set(source_reduced_set)
    start_count: Dict[int, int] = defaultdict(int)
    end_count: Dict[int, int] = defaultdict(int)
    for a, b in current_intervals:
        start_count[a] += 1
        end_count[b] += 1

    operations = []
    total_cost = 0.0

    # Remaining pools after removing common TADs
    remaining_source = set(source_reduced_set)
    remaining_target = set(target_reduced_set)

    # Freeze TADs used in split/merge, as requested
    frozen_source_side = set()
    frozen_target_side = set()

    def _delete_tad(iv: Interval, grouped: bool = False):
        nonlocal total_cost
        if iv not in current_intervals:
            return 0.0, []

        a, b = iv
        start_disappears = (start_count[a] == 1)
        end_disappears = (end_count[b] == 1)
        boundary_ops = []

        cost = Ct
        if start_disappears and a not in matched_source_starts:
            cost += Cb
            boundary_ops.append(('delete_start', a))
        if end_disappears and b not in matched_source_ends:
            cost += Cb
            boundary_ops.append(('delete_end', b))

        current_intervals.remove(iv)
        start_count[a] -= 1
        end_count[b] -= 1

        if not grouped:
            operations.append({'op': 'delete', 'interval': iv, 'boundary_ops': boundary_ops, 'cost': cost})
        total_cost += cost
        return cost, boundary_ops

    def _insert_tad(iv: Interval, grouped: bool = False):
        nonlocal total_cost
        if iv in current_intervals:
            return 0.0, []

        a, b = iv
        start_new = (start_count[a] == 0)
        end_new = (end_count[b] == 0)
        boundary_ops = []

        cost = Ct
        if start_new and a not in matched_target_starts:
            cost += Cb
            boundary_ops.append(('create_start', a))
        if end_new and b not in matched_target_ends:
            cost += Cb
            boundary_ops.append(('create_end', b))

        current_intervals.add(iv)
        start_count[a] += 1
        end_count[b] += 1

        if not grouped:
            operations.append({'op': 'insert', 'interval': iv, 'boundary_ops': boundary_ops, 'cost': cost})
        total_cost += cost
        return cost, boundary_ops

    def _enumerate_candidate_events():
        """
        Enumerate all valid split/merge trios after tolerance-aware common TAD removal.

        Split:
            source [a,b] -> target [a,x], [y,b], with x <= y

        Merge:
            source [a,x], [y,b] -> target [a,b], with x <= y
        """
        events = []
        seen = set()

        # Split events
        for src_iv in remaining_source:
            for left in remaining_target:
                for right in remaining_target:
                    if left == right:
                        continue
                    if _compatible_split(src_iv, left, right, tolerance):
                        key = ('split', src_iv, left, right)
                        if key not in seen:
                            seen.add(key)
                            events.append({
                                'kind': 'split',
                                'members': [src_iv, left, right],
                                'anchor': src_iv,
                            })

        # Merge events
        for tgt_iv in remaining_target:
            for left in remaining_source:
                for right in remaining_source:
                    if left == right:
                        continue
                    if _compatible_merge(left, right, tgt_iv, tolerance):
                        key = ('merge', left, right, tgt_iv)
                        if key not in seen:
                            seen.add(key)
                            events.append({
                                'kind': 'merge',
                                'members': [left, right, tgt_iv],
                                'anchor': tgt_iv,
                            })

        # Deterministic order; max-flow will still decide the final selected set.
        def sort_key(ev):
            m = ev['members']
            return (
                0 if ev['kind'] == 'split' else 1,
                m[0][0], m[0][1],
                m[1][0], m[1][1],
                m[2][0], m[2][1],
            )

        events.sort(key=sort_key)
        return events

    def _select_events_via_max_flow(events):
        """
        Build a max-flow network where each candidate event is represented as a path:
            source -> event -> member1 -> member2 -> member3 -> sink

        Each TAD node has capacity 1, so disjointness is enforced by node-capacity splitting.
        """
        if not events:
            return []

        # Build node ids
        # Shared TAD nodes and event nodes are all split into in/out nodes.
        all_tads = set()
        for ev in events:
            for iv in ev['members']:
                all_tads.add(iv)

        tad_list = sorted(all_tads)
        event_list = list(events)

        # Node ids
        # 0 = source, 1 = sink
        next_id = 2
        tad_in = {}
        tad_out = {}
        for iv in tad_list:
            tad_in[iv] = next_id
            tad_out[iv] = next_id + 1
            next_id += 2

        event_in = {}
        event_out = {}
        for idx, ev in enumerate(event_list):
            event_in[idx] = next_id
            event_out[idx] = next_id + 1
            next_id += 2

        S = 0
        T = 1
        dinic = Dinic(next_id)

        # Store source->event edges so we can read selected events back.
        source_event_edges = []

        # TAD capacity edges
        for iv in tad_list:
            dinic.add_edge(tad_in[iv], tad_out[iv], 1)

        # Event capacity edges + path edges
        for idx, ev in enumerate(event_list):
            members = ev['members']
            e_in = event_in[idx]
            e_out = event_out[idx]

            dinic.add_edge(e_in, e_out, 1)
            se = dinic.add_edge(S, e_in, 1)
            source_event_edges.append((idx, se))

            # Event path through its trio
            dinic.add_edge(e_out, tad_in[members[0]], 1)
            dinic.add_edge(tad_out[members[0]], tad_in[members[1]], 1)
            dinic.add_edge(tad_out[members[1]], tad_in[members[2]], 1)
            dinic.add_edge(tad_out[members[2]], T, 1)

        dinic.max_flow(S, T)

        selected = []
        for idx, se in source_event_edges:
            # If the source->event edge is saturated, the event was selected.
            if se.cap == 0:
                selected.append(event_list[idx])

        return selected

    def _apply_split(src_iv: Interval, left: Interval, right: Interval):
        event_cost = 0.0
        event_boundary_ops = []

        c, bops = _delete_tad(src_iv, grouped=True)
        event_cost += c
        event_boundary_ops.extend(bops)

        c, bops = _insert_tad(left, grouped=True)
        event_cost += c
        event_boundary_ops.extend(bops)

        c, bops = _insert_tad(right, grouped=True)
        event_cost += c
        event_boundary_ops.extend(bops)

        remaining_source.remove(src_iv)
        remaining_target.remove(left)
        remaining_target.remove(right)

        frozen_source_side.add(src_iv)
        frozen_target_side.add(left)
        frozen_target_side.add(right)

        operations.append({
            'op': 'split',
            'source': src_iv,
            'targets': [left, right],
            'boundary_ops': event_boundary_ops,
            'cost': event_cost
        })

    def _apply_merge(left: Interval, right: Interval, tgt_iv: Interval):
        event_cost = 0.0
        event_boundary_ops = []

        c, bops = _delete_tad(left, grouped=True)
        event_cost += c
        event_boundary_ops.extend(bops)

        c, bops = _delete_tad(right, grouped=True)
        event_cost += c
        event_boundary_ops.extend(bops)

        c, bops = _insert_tad(tgt_iv, grouped=True)
        event_cost += c
        event_boundary_ops.extend(bops)

        remaining_source.remove(left)
        remaining_source.remove(right)
        remaining_target.remove(tgt_iv)

        frozen_source_side.add(left)
        frozen_source_side.add(right)
        frozen_target_side.add(tgt_iv)

        operations.append({
            'op': 'merge',
            'sources': [left, right],
            'target': tgt_iv,
            'boundary_ops': event_boundary_ops,
            'cost': event_cost
        })

    # Step 1-2: enumerate all candidate split/merge trios
    candidate_events = _enumerate_candidate_events()

    # Step 3: select a maximum set of disjoint events by max flow
    selected_events = _select_events_via_max_flow(candidate_events)

    # Apply selected events. Order does not matter because the selected events are disjoint.
    # Use a deterministic order for reproducibility.
    selected_events.sort(key=lambda ev: (
        0 if ev['kind'] == 'split' else 1,
        ev['members'][0][0], ev['members'][0][1],
        ev['members'][1][0], ev['members'][1][1],
        ev['members'][2][0], ev['members'][2][1],
    ))

    for ev in selected_events:
        if ev['kind'] == 'split':
            src_iv, left, right = ev['members']
            if src_iv in remaining_source and left in remaining_target and right in remaining_target:
                _apply_split(src_iv, left, right)
        else:
            left, right, tgt_iv = ev['members']
            if left in remaining_source and right in remaining_source and tgt_iv in remaining_target:
                _apply_merge(left, right, tgt_iv)

    # Step 4: delete leftover source-only TADs first
    for iv in sorted(list(remaining_source), key=lambda z: (z[0], z[1])):
        _delete_tad(iv)

    # Step 5: then insert leftover target-only TADs
    for iv in sorted(list(remaining_target), key=lambda z: (z[0], z[1])):
        _insert_tad(iv)

    # Add fixed common TADs back to report full final state
    return operations, total_cost, set(current_intervals | common)


def calculate_tad_distance(source_intervals, target_intervals, Ct=1.0, Cb=1.0, tolerance=0):
    """
    Calculate TAD distance = RF distance on TADs + RF distance on boundaries.

    Parameters
    ----------
    source_intervals : set or list of (a, b)
    target_intervals : set or list of (a, b)
    Ct : float
        Cost for TAD differences
    Cb : float
        Cost for boundary differences
    tolerance : int
        Matching tolerance in bins. When positive, use relaxed RF-style matching.

    Returns
    -------
    distance : float
    """
    source_set = set(source_intervals)
    target_set = set(target_intervals)

    if tolerance > 0:
        tad_cost = Ct * relaxed_tad_rf(source_set, target_set, tol=tolerance)
    else:
        # RF on TADs (symmetric difference)
        tad_cost = Ct * (len(source_set - target_set) + len(target_set - source_set))

    # Boundary sets
    source_starts = {a for (a, b) in source_set}
    source_ends = {b for (a, b) in source_set}
    target_starts = {a for (a, b) in target_set}
    target_ends = {b for (a, b) in target_set}

    if tolerance > 0:
        boundary_cost = Cb * relaxed_boundary_rf(
            source_starts,
            target_starts,
            source_ends,
            target_ends,
            tol=tolerance,
        )
        return tad_cost + boundary_cost

    # RF on start boundaries (symmetric difference)
    start_cost = Cb * (len(source_starts - target_starts) + len(target_starts - source_starts))

    # RF on end boundaries (symmetric difference)
    end_cost = Cb * (len(source_ends - target_ends) + len(target_ends - source_ends))

    distance = tad_cost + start_cost + end_cost
    return distance


def test_sequence_cost_vs_tad_distance(num_tests=20, min_val=1, max_val=100,
                                       num_intervals=20, max_per_leaf=8,
                                       Ct=1.0, Cb=1.0, tolerance=0, verbose=True):
    """
    Test that transform_sequence cost equals TAD distance for random laminar intervals.

    Parameters
    ----------
    num_tests : int
        Number of random tests to run
    min_val, max_val : int
        Range for interval boundaries
    num_intervals : int
        Pool of random intervals to generate
    max_per_leaf : int
        Maximum intervals per source/target
    Ct, Cb : float
        Costs for TAD and boundary differences
    tolerance : int
        Matching tolerance in bins used for both transform and distance.
    verbose : bool
        Print detailed output
    """
    import re
    from random_simulation import generate_random_intervals, generate_random_laminar_intervals_from_list

    def parse_interval(s):
        """Parse interval string '[a,b]' to tuple (a,b)"""
        a, b = map(int, re.findall(r"\d+", s))
        return (a, b)

    def _crosses(i1, i2):
        (a, b) = i1
        (c, d) = i2
        return (a < c < b < d) or (c < a < d < b)

    def _can_add(interval, base_set):
        return all(not _crosses(interval, other) for other in base_set)

    def _generate_target_with_more_events(source_set, desired_edits=4):
        """
        Build target from source by applying split/merge-style edits.
        This increases probability that transform_sequence detects grouped events.
        """
        target = set(source_set)
        edits_done = 0

        # Allow extra attempts because some randomly chosen edits can be invalid.
        for _ in range(max(10, desired_edits * 8)):
            if edits_done >= desired_edits:
                break

            # Split candidates: prefer leaf intervals with enough length.
            split_candidates = []
            for iv in target:
                a, b = iv
                if b - a < 3:
                    continue
                has_child = any(a < x and y < b for (x, y) in target if (x, y) != iv)
                if not has_child:
                    split_candidates.append(iv)

            # Merge candidates: any disjoint pair [a,x] and [y,b], x < y, where [a,b] is laminar-safe.
            merge_candidates = []
            tlist = sorted(target)
            for left in tlist:
                for right in tlist:
                    if left == right:
                        continue
                    a, x = left
                    y, b = right
                    if not (a < x < y < b):
                        continue
                    merged = (a, b)
                    if merged in target:
                        continue
                    reduced = target - {left, right}
                    if _can_add(merged, reduced):
                        merge_candidates.append((left, right, merged))

            # Bias toward split slightly to create richer structures.
            do_split = False
            if split_candidates and merge_candidates:
                do_split = (random.random() < 0.6)
            elif split_candidates:
                do_split = True
            elif merge_candidates:
                do_split = False
            else:
                break

            if do_split:
                src_iv = random.choice(split_candidates)
                a, b = src_iv
                reduced = target - {src_iv}

                # Try random boundary choices for split.
                did = False
                for _ in range(25):
                    x = random.randint(a + 1, b - 2)
                    y = random.randint(x + 1, b - 1)
                    left = (a, x)
                    right = (y, b)
                    if _can_add(left, reduced) and _can_add(right, reduced | {left}):
                        target.remove(src_iv)
                        target.add(left)
                        target.add(right)
                        edits_done += 1
                        did = True
                        break
                if not did:
                    continue
            else:
                left, right, merged = random.choice(merge_candidates)
                target.remove(left)
                target.remove(right)
                target.add(merged)
                edits_done += 1

        return target, edits_done

    print(f"\n{'='*70}")
    print("Testing Sequence Cost vs TAD Distance")
    print(f"{'='*70}")
    print(f"Number of tests: {num_tests}")
    print(f"Interval range: [{min_val}, {max_val}]")
    print(f"Intervals pool size: {num_intervals}")
    print(f"Max intervals per set: {max_per_leaf}")
    print(f"Costs: Ct={Ct}, Cb={Cb}\n")

    matches = 0
    mismatches = 0

    for test_num in range(1, num_tests + 1):
        # Generate random available intervals
        available_intervals = generate_random_intervals(min_val, max_val, num_intervals)

        # Generate random laminar source intervals for node 1
        source_intervals_list = generate_random_laminar_intervals_from_list(
            [1], available_intervals, max_per_leaf=max_per_leaf
        )
        # Parse strings to tuples
        source_set = {parse_interval(s) for s in source_intervals_list.get(1, [])}

        # Generate target by editing source with split/merge-biased operations
        target_set, synthetic_edits = _generate_target_with_more_events(
            source_set,
            desired_edits=max(3, max_per_leaf // 2)
        )
        target_list = sorted(target_set)

        # Calculate sequence cost
        ops, seq_cost, final_intervals = transform_sequence(source_set, target_list, Ct=Ct, Cb=Cb, tolerance=tolerance)

        # Calculate TAD distance
        tad_dist = calculate_tad_distance(source_set, target_list, Ct=Ct, Cb=Cb, tolerance=tolerance)

        cost_diff = abs(seq_cost - tad_dist)
        match = cost_diff < 1e-9

        if verbose:
            print(f"\n{'-'*70}")
            print(f"Test {test_num}")
            print(f"{'-'*70}")
            print(f"  Input Source: {sorted(source_set)}")
            print(f"  Input Target: {sorted(target_list)}")
            print(f"  Sequence ({len(ops)} ops):")
            for i, op in enumerate(ops, start=1):
                print(f"    {i}. {op}")
            print(f"  Output Final: {sorted(final_intervals)}")
            print(f"  Synthetic split/merge-biased edits used to build target: {synthetic_edits}")
            print(f"  Sequence cost: {seq_cost:.6f}")
            print(f"  TAD distance:  {tad_dist:.6f}")
            print(f"  Difference:    {cost_diff:.9f}")

        if match:
            matches += 1
            print(f"Result {test_num}: ✓ MATCH")
        else:
            mismatches += 1
            print(f"\nTest {test_num}: ✗ MISMATCH FOUND")
            print(f"  Source: {sorted(source_set)}")
            print(f"  Target: {sorted(target_list)}")
            print(f"  Sequence cost: {seq_cost:.6f}")
            print(f"  TAD distance:  {tad_dist:.6f}")
            print(f"  Difference:    {cost_diff:.9f}")
            print(f"  Operations ({len(ops)}):")
            for op in ops:
                print(f"    {op}")
            print(f"  Final intervals: {sorted(final_intervals)}")
            break

    print(f"\n{'='*70}")
    print("Results Summary")
    print(f"{'='*70}")
    print(f"Total tests: {num_tests}")
    print(f"Matches: {matches}")
    print(f"Mismatches: {mismatches}")
    print(f"Success rate: {100*matches/num_tests:.1f}%")
    print(f"{'='*70}\n")


# -------------------------
# Minimal example / quick test
# -------------------------
if __name__ == "__main__":
    # print("\n" + "="*70)
    # print("Original Minimal Example")
    # print("="*70)
    # problematic case
    # source = [(1,10), (2,7)]
    # target = [(1,5),(6,10), (2,3),(6,7)]
    # ops, total, final = transform_sequence(source, target, Ct=1.0, Cb=1.0)
    # print("Operations:")
    # for o in ops:
    #     print(o)
    # print("Total cost:", total)
    # print("Final intervals:", sorted(final))

    # Run the randomized test harness to compare sequence cost and TAD distance.
    # Keeps the original minimal example commented above as requested.
    test_sequence_cost_vs_tad_distance()