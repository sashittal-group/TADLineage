import random
import math
from collections import defaultdict
from functools import lru_cache

from check_lp_constrained import solve_lp_from_edges_and_leaves
from random_simulation import generate_random_tree, generate_random_intervals, generate_random_laminar_intervals_from_list
from given_boundary import laminar_dp
from typing import Dict, Set, Tuple, Iterable, List
from copy import deepcopy

INF = float("inf")

# -----------------------------
# Sankoff-from-ILP checker (per-node)
# -----------------------------
def build_children_map(edges):
    children = defaultdict(list)
    for u, v in edges:
        children[u].append(v)
    return children


def postorder_nodes(root, children):
    order = []

    def dfs(u):
        for v in children.get(u, []):
            dfs(v)
        order.append(u)

    dfs(root)
    return order  # leaves first, root last


def extract_boundaries_from_node_assignments(node_assignments):
    """
    Given node_assignments: dict node -> iterable of (a,b) tuples (numbers)
    return for each node: starts[node], ends[node] as sets of positions
    """
    starts = defaultdict(set)
    ends = defaultdict(set)
    for v, intervals in node_assignments.items():
        for (a, b) in intervals:
            starts[v].add(a)
            ends[v].add(b)
    return starts, ends


def make_boundary_type_map_for_node(starts_v, ends_v):
    """
    returns dict position->'start'|'end'|'both' for that node
    """
    bmap = {}
    for p in sorted(set(list(starts_v) + list(ends_v))):
        if p in starts_v and p in ends_v:
            bmap[p] = "both"
        elif p in starts_v:
            bmap[p] = "start"
        elif p in ends_v:
            bmap[p] = "end"
    return bmap


def candidate_intervals_from_boundaries(starts_v, ends_v):
    cand = []
    for a in starts_v:
        for b in ends_v:
            if a < b:
                cand.append((a, b))
    return sorted(cand)


def parse_interval_str(s):
    a, b = map(int, s.strip("[]").split(","))
    return (a, b)


def build_node_assignments_from_boundaries(starts_map, ends_map):
    """
    Build a node->list-of-(a,b) mapping where each (a,b) is a candidate
    interval with a in starts_map[node], b in ends_map[node], a<b.
    """
    node_assignments = {}
    for v in set(list(starts_map.keys()) + list(ends_map.keys())):
        starts_v = starts_map.get(v, set())
        ends_v = ends_map.get(v, set())
        cand = []
        for a in starts_v:
            for b in ends_v:
                if a < b:
                    cand.append((a, b))
        node_assignments[v] = cand
    return node_assignments


def normalize_node_assignments(node_assignments):
    """
    Some ILP helpers produce interval strings; normalize to numeric (a,b) tuples.
    Accepts dict node -> iterable of str or (a,b) and returns node -> list of (a,b)
    """
    out = {}
    for v, ints in node_assignments.items():
        out_list = []
        for iv in ints:
            if isinstance(iv, str):
                out_list.append(parse_interval_str(iv))
            else:
                out_list.append(tuple(iv))
        out[v] = out_list
    return out


INF = float("inf")

def infer_internal_assignments_from_ilp_boundaries(
    edges: Iterable[Tuple[int, int]],
    leaf_intervals: Dict[int, Set[str]],
    node_assignments: Dict[int, Iterable[Tuple[int, int]]],
    Ct: float = 1.0,
    laminar_dp_factory = None,
    root: int = 0,
    verbose: bool = False
) -> Tuple[Dict[int, Set[Tuple[int,int]]], Dict[int, Dict[Tuple[int,int], Tuple[float,float]]]]:
    """
    Infer internal-node TAD assignments using a Sankoff-style DP constrained
    by ILP-derived boundary sets and laminar feasibility.

    Parameters
    ----------
    edges
        List/iterable of (parent, child) edges. Root is `root`.
    leaf_intervals
        Mapping leaf_node -> set of interval strings like "[a,b]".
    node_assignments
        Mapping node -> iterable of candidate intervals (a,b) tuples (from ILP or constructed candidates).
    Ct
        Transition cost (flip cost) on edges for interval presence/absence.
    laminar_dp_factory
        Callable: (positions: List[int], boundary_type: Dict[int,str]) -> dp function.
        The returned dp has signature dp(i,j,s_idx,e_idx,status) -> bool as defined earlier.
    root
        Root node id (default 0).
    verbose
        If True, prints helpful debug information.

    Returns
    -------
    internal_node_assignments
        dict node -> set of (a,b) intervals chosen to be present (state=1) after Sankoff reconstruction.
    per_node_interval_costs
        dict node -> { (a,b) : (cost_absent, cost_present) } computed by Sankoff DP.
        Costs are INF if infeasible.

    Notes
    -----
    - Leaves are treated as observed: if a leaf observes (a,b) then cost_present=0 and cost_absent=INF;
      otherwise cost_absent=0 and cost_present=INF. Change this if you want flippable leaves.
    - Tie-breaking at the root: if cost_present == cost_absent, we choose PRESENT (state=1).
    - The laminar feasibility dp must be provided (it depends on the node's positions and boundary types).
    """
    if laminar_dp_factory is None:
        raise ValueError("Provide laminar_dp_factory (your laminar_dp function).")

    # Build children map and postorder traversal
    children = build_children_map(edges)
    postorder = postorder_nodes(root, children)  # leaves first, root last

    # Normalize node_assignments keys -> numeric interval tuples
    # (If they are already numeric, this will do nothing)
    normalized_node_assignments = {}
    for v, ivs in node_assignments.items():
        normalized_node_assignments[v] = []
        for iv in ivs:
            if isinstance(iv, str):
                normalized_node_assignments[v].append(parse_interval_str(iv))
            else:
                # assume an iterable of two ints (a,b)
                normalized_node_assignments[v].append(tuple(iv))

    # Extract starts/ends sets from provided node_assignments
    starts_map, ends_map = extract_boundaries_from_node_assignments(normalized_node_assignments)

    # Parse leaf observed intervals into tuple form for quick membership checks
    leaf_observed = {leaf: {parse_interval_str(s) for s in ints} for leaf, ints in leaf_intervals.items()}

    # per_node_interval_costs[v][(a,b)] = (cost_if_absent, cost_if_present)
    per_node_interval_costs: Dict[int, Dict[Tuple[int,int], Tuple[float,float]]] = defaultdict(dict)

    # ---------
    # Bottom-up Sankoff computation (postorder)
    # ---------
    for v in postorder:
        starts_v = starts_map.get(v, set())
        ends_v = ends_map.get(v, set())

        # Candidate positions for this node (only positions that appear as starts or ends)
        positions_v = sorted(set(starts_v) | set(ends_v))
        if not positions_v:
            # nothing to do at this node
            per_node_interval_costs[v] = {}
            if verbose:
                print(f"[Sankoff] Node {v}: no boundary positions -> skipping")
            continue

        # Build laminar DP for this node (feasibility checker)
        boundary_type_v = make_boundary_type_map_for_node(starts_v, ends_v)
        dp_v = laminar_dp_factory(positions_v, boundary_type_v)
        pos_index = {p: i for i, p in enumerate(positions_v)}

        # Candidate intervals at this node: all (start x end) pairs where start < end
        cand_intervals = candidate_intervals_from_boundaries(starts_v, ends_v)
        if verbose:
            print(f"[Sankoff] Node {v}: positions={positions_v}, #candidates={len(cand_intervals)}")

        node_costs: Dict[Tuple[int,int], Tuple[float,float]] = {}

        # For each candidate interval, compute child costs and parent Sankoff costs
        for (a, b) in cand_intervals:
            # indices for dp calls: entire region is [0, len(positions_v)-1]
            i_idx = 0
            j_idx = len(positions_v) - 1
            s_idx = pos_index[a]
            e_idx = pos_index[b]

            # Laminar feasibility checks for this node when interval is required present/absent
            feasible_present = dp_v(i_idx, j_idx, s_idx, e_idx, "present")
            feasible_absent  = dp_v(i_idx, j_idx, s_idx, e_idx, "absent")

            # Extra sanity: if a or b not in the starts/ends sets, presence is impossible
            if (a not in starts_v) or (b not in ends_v):
                feasible_present = False

            # Build list of (cost0_child, cost1_child) for all children
            child_cost_pairs: List[Tuple[float,float]] = []
            child_list = children.get(v, [])

            for c in child_list:
                # Leaf child: costs come from observed leaf_intervals
                if c in leaf_observed:
                    if (a, b) in leaf_observed[c]:
                        child_cost_pairs.append((INF, 0.0))  # (absent_cost, present_cost)
                    else:
                        child_cost_pairs.append((0.0, INF))
                    continue

                # Internal child: we should have computed its costs already (postorder)
                child_map = per_node_interval_costs.get(c, {})
                if (a, b) in child_map:
                    cost_abs_c, cost_pres_c = child_map[(a,b)]
                    child_cost_pairs.append((cost_abs_c, cost_pres_c))
                else:
                    # Child cannot host this interval (e.g., missing boundaries); treat presence impossible
                    child_cost_pairs.append((0.0, INF))

            # Compute parent's cost when parent state = 1 (present)
            if not feasible_present:
                parent_cost_present = INF
            else:
                ssum = 0.0
                impossible = False
                for cost0_c, cost1_c in child_cost_pairs:
                    # child stays present: cost1_c
                    # child flips from absent -> present: cost0_c + Ct
                    choice = min(cost1_c, cost0_c + Ct)
                    if choice == INF:
                        impossible = True
                        break
                    ssum += choice
                parent_cost_present = INF if impossible else ssum

            # Compute parent's cost when parent state = 0 (absent)
            if not feasible_absent:
                parent_cost_absent = INF
            else:
                ssum = 0.0
                impossible = False
                for cost0_c, cost1_c in child_cost_pairs:
                    # child stays absent: cost0_c
                    # child flips from present -> absent: cost1_c + Ct
                    choice = min(cost0_c, cost1_c + Ct)
                    if choice == INF:
                        impossible = True
                        break
                    ssum += choice
                parent_cost_absent = INF if impossible else ssum

            # If this node is actually a leaf (it might happen the node id falls into leaf_intervals),
            # enforce leaf observation semantics: observed state fixed.
            if v in leaf_observed:
                if (a,b) in leaf_observed[v]:
                    parent_cost_absent = INF
                    parent_cost_present = 0.0
                else:
                    parent_cost_absent = 0.0
                    parent_cost_present = INF

            node_costs[(a,b)] = (parent_cost_absent, parent_cost_present)

            if verbose:
                print(f"   interval {(a,b)} -> feasible P={feasible_present}, A={feasible_absent}, costs (abs,pres)=({parent_cost_absent},{parent_cost_present})")

        per_node_interval_costs[v] = node_costs

    # --------------------------
    # Top-down Sankoff reconstruction (assign states)
    # --------------------------
    # Collect all intervals that appear anywhere
    all_intervals = set()
    for cmap in per_node_interval_costs.values():
        all_intervals.update(cmap.keys())

    # Prepare storage for assigned states per node per interval
    assigned_state: Dict[int, Dict[Tuple[int,int], int]] = defaultdict(dict)

    # Root choice: pick the state minimizing cost; tie break -> prefer PRESENT (1)
    root_costs = per_node_interval_costs.get(root, {})

    for iv in all_intervals:
        c_abs, c_pres = root_costs.get(iv, (INF, INF))
        if c_abs == INF and c_pres == INF:
            # No feasible state for this interval at root; skip it
            assigned_state[root][iv] = 0  # or could skip entirely; here we mark as absent
            continue
        elif c_pres <= c_abs:
            assigned_state[root][iv] = 1
        else:
            assigned_state[root][iv] = 0

    # Propagate down: for each node, choose child state given parent state by local Sankoff choices
    # We'll do a simple DFS stack to traverse nodes top-down
    stack = [root]
    visited = set()
    while stack:
        u = stack.pop()
        visited.add(u)
        # push children for traversal
        for c in children.get(u, []):
            if c not in visited:
                stack.append(c)

        # for each candidate interval, decide the child's state given parent's state
        for iv in all_intervals:
            parent_state = assigned_state[u].get(iv, 0)  # default 0 if missing

            # leaf children are fixed by observation; internal children follow Sankoff choice
            for c in children.get(u, []):
                if c in leaf_observed:
                    
                    assigned_state[c][iv] = 1 if iv in leaf_observed[c] else 0
                    continue

                # child costs from bottom-up computation
                child_costs_map = per_node_interval_costs.get(c, {})
                c_abs, c_pres = child_costs_map.get(iv, (INF, INF))
                if c_abs == INF and c_pres == INF:
                    # No feasible state for this interval at child; assign absent (0)
                    assigned_state[c][iv] = 0
                     # (Alternatively, could skip entirely; here we mark as absent)
                     # Note: if parent requires presence but child can't host, this will create a mismatch; that's just the nature of the data.
                     # In a real implementation, you might want to track these infeasibilities more explicitly.
                     # For now, we just assign absent and move on.
                    continue

                elif parent_state == 1:
                    # choose min between child present (c_pres) or child absent + Ct
                    opt_present = c_pres
                    opt_abs_if_flip = c_abs + Ct if c_abs < INF else INF
                    assigned_state[c][iv] = 1 if opt_present <= opt_abs_if_flip else 0
                else:
                    # parent_state == 0
                    opt_abs = c_abs
                    opt_pres_if_flip = c_pres + Ct if c_pres < INF else INF
                    assigned_state[c][iv] = 0 if opt_abs <= opt_pres_if_flip else 1

    # Build final internal assignments: only keep nodes that are internal (not leaves) and intervals with state=1
    leaf_nodes_set = set(leaf_observed.keys())
    internal_node_assignments: Dict[int, Set[Tuple[int,int]]] = {}
    for v, iv_map in assigned_state.items():
        if v in leaf_nodes_set:
            continue
        chosen = {iv for iv, st in iv_map.items() if st == 1}
        internal_node_assignments[v] = chosen


    return internal_node_assignments, per_node_interval_costs

from collections import defaultdict, deque
from copy import deepcopy
import math

INF = float("inf")

def sankoff_cost_single_interval(interval,
                                 node_candidates,
                                 edges,
                                 leaf_intervals,
                                 forced_present,
                                 Ct=1.0,
                                 Cb=1.0,
                                 laminar_dp_factory=None,
                                 root=0):
    """
    Same DP as before, but final score is GLOBAL edge-parsimony
    computed on full interval sets (forced_present + this interval).
    """

    if laminar_dp_factory is None:
        raise ValueError("laminar_dp_factory required")

    a, b = interval

    children = build_children_map(edges)
    postorder = postorder_nodes(root, children)

    leaf_obs = {leaf: {parse_interval_str(s) for s in ints}
                for leaf, ints in leaf_intervals.items()}

    starts_map, ends_map = extract_boundaries_from_node_assignments(node_candidates)

    dp_cost = {}

    # -------------------------
    # Bottom-up DP (unchanged)
    # -------------------------
    for v in postorder:

        if v in leaf_obs:
            if interval in leaf_obs[v]:
                dp_cost[v] = (INF, 0.0)
            else:
                dp_cost[v] = (0.0, INF)
            continue

        starts_v = starts_map.get(v, set())
        ends_v = ends_map.get(v, set())
        structurally_possible = (a in starts_v and b in ends_v)

        feasible_present = False
        feasible_absent = True

        if structurally_possible:
            positions_v = sorted(starts_v | ends_v)
            if len(positions_v) >= 2:
                boundary_type_v = make_boundary_type_map_for_node(starts_v, ends_v)
                dp_v = laminar_dp_factory(positions_v, boundary_type_v)
                pos_index = {p: i for i, p in enumerate(positions_v)}
                i_idx = 0
                j_idx = len(positions_v) - 1
                s_idx = pos_index[a]
                e_idx = pos_index[b]
                feasible_present = dp_v(i_idx, j_idx, s_idx, e_idx, "present")

        # forced_present constraint
        if v in forced_present and interval in forced_present[v]:
            feasible_absent = False

        child_list = children.get(v, [])

        # present cost
        if not feasible_present:
            cost_present = INF
        else:
            ssum = 0.0
            impossible = False
            for c in child_list:
                c0, c1 = dp_cost.get(c, (0.0, INF))
                opt = min(c1, c0 + Ct)
                if opt == INF:
                    impossible = True
                    break
                ssum += opt
            cost_present = INF if impossible else ssum

        # absent cost
        if not feasible_absent:
            cost_absent = INF
        else:
            ssum = 0.0
            impossible = False
            for c in child_list:
                c0, c1 = dp_cost.get(c, (0.0, INF))
                opt = min(c0, c1 + Ct)
                if opt == INF:
                    impossible = True
                    break
                ssum += opt
            cost_absent = INF if impossible else ssum

        dp_cost[v] = (cost_absent, cost_present)

    if root not in dp_cost:
        return INF, {}

    root0, root1 = dp_cost[root]
    if root0 == INF and root1 == INF:
        return INF, {}

    # -------------------------
    # Top-down traceback
    # -------------------------
    assignment = {}
    root_state = 1 if (root1 < root0) else 0
    assignment[root] = root_state

    stack = [root]
    while stack:
        u = stack.pop()
        parent_state = assignment[u]
        for c in children.get(u, []):

            if c in leaf_obs:
                assignment[c] = 1 if interval in leaf_obs[c] else 0
                continue

            c_abs, c_pres = dp_cost.get(c, (0.0, INF))

            if parent_state == 1:
                opt_pres = c_pres
                opt_abs_flip = c_abs + Ct if c_abs < INF else INF
                child_state = 1 if opt_pres <= opt_abs_flip else 0
            else:
                opt_abs = c_abs
                opt_pres_flip = c_pres + Ct if c_pres < INF else INF
                child_state = 0 if opt_abs <= opt_pres_flip else 1

            assignment[c] = child_state
            stack.append(c)

    # ==========================================================
    # 🔥 NEW PART: compute GLOBAL edge-parsimony score
    # ==========================================================

    # Build full interval sets per node
    full_assignments = {}

    # Start from forced_present
    for v in forced_present:
        full_assignments[v] = set(forced_present[v])

    # Add this interval according to assignment
    for v, state in assignment.items():
        if state == 1:
            full_assignments.setdefault(v, set()).add(interval)
        else:
            full_assignments.setdefault(v, set())

    # Add leaf intervals
    for leaf, ivs in leaf_obs.items():
        full_assignments[leaf] = set(ivs)

    # Compute full edge score
    total_score = calculate_edge_parsimony_score(
        edges,
        full_assignments,
        Ct=Ct,
        Cb=Cb
    )

    return total_score, assignment



def refine_with_local_candidate_search(edges,
                                       leaf_intervals,
                                       ilp_node_assignments,
                                       Ct=1.0,
                                       laminar_dp_factory=None,
                                       root=0,
                                       max_sweeps=20,
                                       verbose=False):
    """
    Updated refinement:
      - BFS order from root
      - For each node and for each uncovered start (or end) evaluate all candidates (s,e)
        pick minimum and then freeze globally using the returned assignment_map.
      - Repeat sweeps until convergence or max_sweeps.
    """
    if laminar_dp_factory is None:
        raise ValueError("laminar_dp_factory required")

    node_candidates = normalize_node_assignments(ilp_node_assignments)
    initial_starts_map, initial_ends_map = extract_boundaries_from_node_assignments(node_candidates)

    # initial inference -> starting frozen set
    initial_assign, _ = infer_internal_assignments_from_ilp_boundaries(
        edges, leaf_intervals, node_candidates, Ct=Ct,
        laminar_dp_factory=laminar_dp_factory, root=root, verbose=verbose
    )

    frozen = {v: set(ivset) for v, ivset in initial_assign.items()}

    # ensure every node has entries
    all_nodes = set(initial_starts_map.keys()) | set(initial_ends_map.keys())
    leaf_nodes = set(leaf_intervals.keys())
    internal_nodes = sorted(all_nodes - leaf_nodes)

    for v in all_nodes:
        frozen.setdefault(v, set())
        initial_starts_map.setdefault(v, set())
        initial_ends_map.setdefault(v, set())

    # build children and BFS order
    children = build_children_map(edges)

    def bfs_order(start):
        q = deque([start])
        order = []
        visited = set([start])
        while q:
            u = q.popleft()
            order.append(u)
            for c in children.get(u, []):
                if c not in visited:
                    visited.add(c)
                    q.append(c)
        return order

    bfs_nodes = bfs_order(root)
    # We will only process internal nodes in BFS order
    bfs_internal = [v for v in bfs_nodes if v in internal_nodes]

    diagnostics = []
    changed = True
    sweep = 0
    tol = 1e-9

    # Precompute list of unique candidate intervals (for final scoring)
    unique_intervals = set()
    for candlist in node_candidates.values():
        for iv in candlist:
            unique_intervals.add(tuple(iv))

    while changed and sweep < max_sweeps:
        sweep += 1
        changed = False
        if verbose:
            total_frozen = sum(len(s) for s in frozen.values())
            print(f"\n[Refine] Sweep {sweep} start. Frozen intervals total = {total_frozen}")

        # Recompute uncovered sets before starting BFS
        uncovered_starts = {}
        uncovered_ends = {}
        for v in internal_nodes:
            covered_starts = {a for (a, b) in frozen[v]}
            covered_ends = {b for (a, b) in frozen[v]}
            uncovered_starts[v] = set(initial_starts_map[v]) - covered_starts
            uncovered_ends[v] = set(initial_ends_map[v]) - covered_ends

        # Process nodes in BFS order
        for v in bfs_internal:
            # recompute uncovered for v in case earlier nodes changed frozen this sweep
            covered_starts = {a for (a, b) in frozen[v]}
            covered_ends = {b for (a, b) in frozen[v]}
            uncovered_starts[v] = set(initial_starts_map[v]) - covered_starts
            uncovered_ends[v] = set(initial_ends_map[v]) - covered_ends

            # Start-phase: for each uncovered start s evaluate all candidate (s,e)
            for s in sorted(list(uncovered_starts[v])):
                best_score = INF
                best_e = None
                best_assign = None
                best_e_was_uncovered = False

                possible_ends = sorted(initial_ends_map[v])
                if not possible_ends:
                    continue

                # existing tads at v for laminar checks
                existing_tads = set(frozen[v])

                for e in possible_ends:
                    if e <= s:
                        continue
                    cand = (s, e)

                    # laminarity check at node v wrt existing frozen tads
                    crosses = False
                    for (fa, fb) in existing_tads:
                        if (fa < s < fb < e) or (s < fa < e < fb):
                            crosses = True
                            break
                    if crosses:
                        continue

                    # Evaluate candidate by forcing (s,e) at v only
                    forced_map = {u: set(ivset) for u, ivset in frozen.items()}
                    forced_map.setdefault(v, set()).add(cand)

                    score, assign_map = sankoff_cost_single_interval(
                        cand,
                        node_candidates,
                        edges,
                        leaf_intervals,
                        forced_map,
                        Ct=Ct,
                        laminar_dp_factory=laminar_dp_factory,
                        root=root
                    )

                    if score == INF:
                        continue

                    e_uncovered = (e in uncovered_ends.get(v, set()))

                    # tie-breaking: prefer uncovered counterpart if tie
                    is_better = (score + tol) < best_score
                    is_tie = abs(score - best_score) <= tol

                    if is_better:
                        print (f"   Node {v}: candidate {(s,e)} better than current best with score {score} vs {best_score}")
                        best_score = score
                        best_e = e
                        best_assign = assign_map
                        best_e_was_uncovered = e_uncovered
                    elif is_tie:
                        if e_uncovered and not best_e_was_uncovered:
                            best_score = score
                            best_e = e
                            best_assign = assign_map
                            best_e_was_uncovered = True

                # After scanning all e for this s, freeze best if any
                if best_e is not None and best_score < INF and best_assign:
                    # Freeze globally where assign_map[node]==1
                    for u, st in best_assign.items():
                        if st == 1:
                            frozen.setdefault(u, set()).add((s, best_e))
                            changed = True
                    if verbose:
                        print(f"[Sweep {sweep}] Node {v}: frozen chosen {(s,best_e)} (score={best_score})")

                    # Update uncovered sets for nodes impacted (we update globally for internal nodes)
                    for u in internal_nodes:
                        covered_starts_u = {a for (a,b) in frozen[u]}
                        covered_ends_u = {b for (a,b) in frozen[u]}
                        uncovered_starts[u] = set(initial_starts_map[u]) - covered_starts_u
                        uncovered_ends[u] = set(initial_ends_map[u]) - covered_ends_u

            # End-phase: for each uncovered end e evaluate all candidate (s,e)
            for e in sorted(list(uncovered_ends[v])):
                best_score = INF
                best_s = None
                best_assign = None
                best_s_was_uncovered = False

                possible_starts = sorted(initial_starts_map[v])
                if not possible_starts:
                    continue

                existing_tads = set(frozen[v])

                for s in possible_starts:
                    if s >= e:
                        continue
                    cand = (s, e)

                    # laminarity check
                    crosses = False
                    for (fa, fb) in existing_tads:
                        if (fa < s < fb < e) or (s < fa < e < fb):
                            crosses = True
                            break
                    if crosses:
                        continue

                    forced_map = {u: set(ivset) for u, ivset in frozen.items()}
                    forced_map.setdefault(v, set()).add(cand)

                    score, assign_map = sankoff_cost_single_interval(
                        cand,
                        node_candidates,
                        edges,
                        leaf_intervals,
                        forced_map,
                        Ct=Ct,
                        laminar_dp_factory=laminar_dp_factory,
                        root=root
                    )

                    if score == INF:
                        continue

                    s_uncovered = (s in uncovered_starts.get(v, set()))
                    is_better = (score + tol) < best_score
                    is_tie = abs(score - best_score) <= tol

                    if is_better:
                        best_score = score
                        best_s = s
                        best_assign = assign_map
                        best_s_was_uncovered = s_uncovered
                    elif is_tie:
                        if s_uncovered and not best_s_was_uncovered:
                            best_score = score
                            best_s = s
                            best_assign = assign_map
                            best_s_was_uncovered = True

                if best_s is not None and best_score < INF and best_assign:
                    for u, st in best_assign.items():
                        if st == 1:
                            frozen.setdefault(u, set()).add((best_s, e))
                            changed = True
                    if verbose:
                        print(f"[Sweep {sweep}] Node {v}: frozen chosen {(best_s,e)} (score={best_score})")

                    # Update uncovered sets globally
                    for u in internal_nodes:
                        covered_starts_u = {a for (a,b) in frozen[u]}
                        covered_ends_u = {b for (a,b) in frozen[u]}
                        uncovered_starts[u] = set(initial_starts_map[u]) - covered_starts_u
                        uncovered_ends[u] = set(initial_ends_map[u]) - covered_ends_u

        diagnostics.append({
            "sweep": sweep,
            "frozen_count": sum(len(x) for x in frozen.values())
        })

        if verbose:
            print(f"[Refine] Sweep {sweep} finished. Frozen intervals now = {sum(len(x) for x in frozen.values())}")

        if not changed:
            break

    # Final Sankoff total: sum per-interval Sankoff costs (respecting frozen forced presence)
    final_total = 0.0
    for iv in unique_intervals:
        # call sankoff with frozen as forced_present to compute final contribution
        cost, _ = sankoff_cost_single_interval(
            iv, node_candidates, edges, leaf_intervals, frozen,
            Ct=Ct, laminar_dp_factory=laminar_dp_factory, root=root
        )
        final_total += cost

    if verbose:
        print("[Refine] Finished.")
        print(f"Final frozen count: {sum(len(x) for x in frozen.values())}")
        print(f"Final Sankoff total: {final_total}")

    return dict(frozen), final_total, diagnostics

def calculate_edge_parsimony_score(edges, node_interval_assignments, Ct=1.0, Cb=1.0):
    """
    Parsimony score over edges where edge distance is:
      RF on TAD intervals + RF on start boundaries + RF on end boundaries.
    """
    total = 0.0
    for u, v in edges:
        iu = set(node_interval_assignments.get(u, set()))
        iv = set(node_interval_assignments.get(v, set()))

        starts_u = {a for (a, b) in iu}
        starts_v = {a for (a, b) in iv}
        ends_u = {b for (a, b) in iu}
        ends_v = {b for (a, b) in iv}

        tad_rf = len(iu - iv) + len(iv - iu)
        start_rf = len(starts_u - starts_v) + len(starts_v - starts_u)
        end_rf = len(ends_u - ends_v) + len(ends_v - ends_u)

        total += Ct * tad_rf + Cb * start_rf + Cb * end_rf
    return total


# -----------------------------
# run_single_simulation (integrated)
# -----------------------------
def run_single_simulation(total_nodes, min_val, max_val, num_intervals,
                          max_per_leaf=5, Ct=1.0, Cb=1.0, mode="convex",
                          use_fitch_hartigan=False, verbose=False):
    """
    Run a single simulation instance with random tree and random intervals.
    mode: 'convex' | 'lex_boundary_first' | 'lex_tad_first'
    Returns (lp_cost, is_integral, edges, leaf_intervals, x_vars, nodes, intervals)
    """
    # Generate random tree
    try:
        edges, leaf_nodes = generate_random_tree(total_nodes)
    except ValueError as exc:
        if verbose:
            print(f"Skipped simulation due to invalid tree parameters: {exc}")
        return None, None, [], {}, None, None, None

    # Generate random intervals (available set)
    available_intervals = generate_random_intervals(min_val, max_val, num_intervals)

    # Assign random laminar intervals to leaves
    leaf_intervals = generate_random_laminar_intervals_from_list(
        leaf_nodes, available_intervals, max_per_leaf=min(max_per_leaf, num_intervals)
    )

    # # ---- NOTE: for quick debugging you had a fixed example; keep or remove as needed ----
    # edges = [(0, 1), (0, 2), (2, 3), (3, 4), (3, 5), (1, 6), (1, 9), (2, 7), (4, 10), (4, 11), (5, 12), (5, 8)]
    # leaf_intervals = {
    #     6: ['[6,9]'],
    #     7: ['[4,7]', '[9,10]'],
    #     8: ['[1,3]', '[4,6]', '[9,10]'],
    #     9: ['[4,6]', '[4,7]', '[4,9]'],
    #     10: ['[6,7]', '[8,10]'],
    #     11: ['[2,7]', '[3,4]', '[5,7]', '[9,10]'],
    #     12: ['[1,2]', '[1,3]', '[4,9]', '[5,7]'],
    # }

    if verbose:
        print(f"\nEdges: {edges}")
        print(f"Leaf nodes: {leaf_nodes}")
        print(f"Available intervals (sampled): {available_intervals}")
        print(f"Leaf intervals: {leaf_intervals}")

    lp_cost, is_integral, node_assignments, x, nodes, intervals = solve_lp_from_edges_and_leaves(
        edges,
        leaf_intervals,
        Ct=Ct,
        Cb=Cb,
        mode=mode,
        require_laminar_internal=True,
        use_fitch_hartigan=use_fitch_hartigan,
        verbose=verbose,
    )

    # print the intervals which are non zero at internal nodes from node_assignments
    print("\nLP solution (non-zero values at internal nodes):")
    for v in nodes:
        if v not in leaf_intervals:
            print(f"  Node {v}:")
            for (a, b) in node_assignments.get(v, []):
                print(f"    Interval ({a},{b})")

    # --- Infer internal assignments from ILP boundaries ---
        # --- Iterative Sankoff refinement starting from ILP boundaries ---
    try:
        final_internal_assignments, final_total, diagnostics = refine_with_local_candidate_search(
            edges,
            leaf_intervals,
            node_assignments,
            Ct=Ct,
            laminar_dp_factory=laminar_dp,
            root=0,
            max_sweeps=20,
            verbose=verbose
        )

        # print final internal assignments after refinement (only internal nodes)
        print("\nFinal internal assignments after iterative Sankoff refinement:")
        for v in sorted(final_internal_assignments):
            # skip leaves (they should be in leaf_intervals, but be safe)
            if v in leaf_intervals:
                continue
            print(f"  Node {v}: {sorted(final_internal_assignments[v])}")

    except Exception as e:
        print("Error while running iterative Sankoff refinement:", e)
        final_internal_assignments = {}
        diagnostics = []
        final_total = None

    # Merge inferred internal assignments with observed leaves
    full_assignments = {}

    # Add internal assignments (may be empty)
    for v, ints in final_internal_assignments.items():
        # ensure storing tuples (a,b)
        full_assignments[v] = set(ints)

    # Add leaves exactly as observed (do not change leafs)
    for leaf, ints in leaf_intervals.items():
        full_assignments[leaf] = set(parse_interval_str(s) for s in ints)

    # Compute edge-parsimony (RF-style) total for these assignments
    parsimony_total = calculate_edge_parsimony_score(
        edges, full_assignments, Ct=Ct, Cb=Cb
    )

    # Print comparison summary
    print("\nComparison:")
    print(f"  ILP objective: {lp_cost}")
    print(f"  Edge parsimony total (after iterative Sankoff refinement): {parsimony_total}")

    print("\nFinal inferred internal assignments (union over iterations):")
    for v in sorted(full_assignments):
        if v not in leaf_intervals:
            print(f"  Node {v}: {sorted(full_assignments[v])}")

    # Print diagnostics (if available)
    if diagnostics:
        print("\nRefinement diagnostics (last sweep summary):")
        last = diagnostics[-1]
        print(f"  Last sweep: {last.get('sweep', 'N/A')}, frozen_count={last.get('frozen_count', 'N/A')}")

    # Check for mismatch between ILP and parsimony_total
    mismatch_found = False

    # numeric tolerance
    if parsimony_total is None or lp_cost is None:
        print("  Could not compare (missing values).")
    else:
        if parsimony_total == float("inf") and lp_cost == float("inf"):
            print("  Both infinite (match).")
        else:
            tol = 1e-6
            if abs(lp_cost - parsimony_total) > tol:
                mismatch_found = True
                print("\n*** MISMATCH between ILP objective and edge parsimony total! ***")
                print(f"  ILP = {lp_cost:.6f}, Parsimony = {parsimony_total:.6f}, diff = {lp_cost - parsimony_total:.6f}")

                # Try to print useful debugging info from diagnostics if present
                if diagnostics:
                    print("\nDiagnostics (recent):")
                    for d in diagnostics[-3:]:
                        print(f"  sweep={d.get('sweep')}, frozen_count={d.get('frozen_count')}")
                else:
                    print("  (no diagnostics available)")
            else:
                print("  ✓ ILP objective matches edge parsimony total (within tolerance).")

    # return values (keeps your original calling signature)
    return lp_cost, is_integral, edges, leaf_intervals, x, nodes, intervals, mismatch_found

# -----------------------------
# Run multiple simulations
# -----------------------------
def run_simulation(num_simulations=100, min_nodes=3, max_nodes=20, min_val=1, max_val=10,
                   num_intervals=8, max_per_leaf=5, Ct=2.0, Cb=1.0, mode="convex",
                   use_fitch_hartigan=False):
    """
    Run multiple simulations using the LP-only verification (no brute force).
    Stops early if a non-integral LP solution is found (you can change this behavior).
    """
    print(f"Running {num_simulations} simulations (mode={mode})...")
    print(f"Tree structure: {min_nodes}-{max_nodes} total nodes")
    print(f"Intervals: range [{min_val}, {max_val}], sampled {num_intervals} total intervals")
    print(f"Max intervals per leaf: {max_per_leaf}")
    print(f"Costs: Ct={Ct}, Cb={Cb}\n")
    for sim in range(1, num_simulations + 1):
        total_nodes = random.randint(min_nodes, max_nodes)
        print(f"Simulation {sim} (TotalNodes={total_nodes})...", end=" ")

        result = run_single_simulation(total_nodes, min_val, max_val, num_intervals,
                                       max_per_leaf=max_per_leaf, Ct=Ct, Cb=Cb, mode=mode,
                                       use_fitch_hartigan=use_fitch_hartigan,
                                       verbose=False)

        if result[0] is None:
            print("Skipped")
            continue

        lp_cost, is_integral, edges, leaf_intervals, x_vars, nodes, intervals, mismatch_found = result

        if lp_cost is None:
            print("Solver failed")
            continue

        if not is_integral:
            print("\n\n*** FOUND NON-INTEGRAL LP SOLUTION! ***")
            print(f"Simulation #{sim}")
            print(f"LP cost: {lp_cost:.4f}")
            print(f"Is integral: {is_integral}")
            print(f"\nTree edges: {edges}")
            print("\nLeaf intervals: ")
            for v in sorted(leaf_intervals.keys()):
                print(f"  Node {v}: {sorted(leaf_intervals[v])}")
            print("\nLP solution (non-integral values at internal nodes):")
            for v in nodes:
                if v not in leaf_intervals:
                    print(f"  Node {v}:")
                    for (a, b) in intervals:
                        val = x_vars[v, a, b].X
                        if val > 1e-6 and val < 1 - 1e-6:
                            print(f"    x[{v},({a},{b})] = {val:.6f}")
            return

        if mismatch_found:
            print("\n\n*** ILP and boundary-given-case parsimony mismatch found. Stopping simulations. ***")
            print(f"Simulation #{sim}")
            print(f"Tree edges: {edges}")
            print("Leaf intervals:")
            for v in sorted(leaf_intervals.keys()):
                print(f"  Node {v}: {sorted(leaf_intervals[v])}")
            return

        print(f"OK (LP={lp_cost:.2f}, Integral={is_integral})")

    print(f"\nCompleted {num_simulations} simulations. No non-integral LP solutions found.")


# -----------------------------
# Main Entry Point
# -----------------------------
if __name__ == "__main__":
    # Simulation parameters
    NUM_SIMULATIONS = 10000
    MIN_NODES = 3        # Minimum total number of nodes
    MAX_NODES = 15       # Maximum total number of nodes
    MIN_VAL = 1
    MAX_VAL = 10
    NUM_INTERVALS = 20
    MAX_PER_LEAF = 5    # Maximum number of intervals per leaf
    Ct = 1.0
    Cb = 1.0
    MODE = "convex"  # 'convex' | 'lex_boundary_first' | 'lex_tad_first'
    USE_FITCH_HARTIGAN = False

    # random.seed(42)  # For reproducibility if desired

    run_simulation(
        num_simulations=NUM_SIMULATIONS,
        min_nodes=MIN_NODES,
        max_nodes=MAX_NODES,
        min_val=MIN_VAL,
        max_val=MAX_VAL,
        num_intervals=NUM_INTERVALS,
        max_per_leaf=MAX_PER_LEAF,
        Ct=Ct,
        Cb=Cb,
        mode=MODE,
        use_fitch_hartigan=USE_FITCH_HARTIGAN
    )