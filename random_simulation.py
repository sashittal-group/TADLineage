import random
import math
from functools import lru_cache

from check_lp_constrained import solve_lp_from_edges_and_leaves

# -----------------------------
# Interval helpers
# -----------------------------

def parse_interval_str(s):
    """
    Parse an interval string like "[1,5]" into a tuple (1, 5).
    """
    parts = s.strip().replace("[", "").replace("]", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid interval string: {s}")
    a, b = map(int, parts)
    return (a, b)


def interval_to_str(interval):
    a, b = interval
    return f"[{a},{b}]"


def crosses(i1, i2):
    """
    Strict crossing test.
    Intervals that share endpoints are not considered crossing here.
    """
    a, b = i1
    c, d = i2
    return (a < c < b < d) or (c < a < d < b)


def is_laminar_interval_set(intervals):
    """
    Return True iff a set of intervals is laminar (no strict crossings).
    """
    intervals = list(intervals)
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            if crosses(intervals[i], intervals[j]):
                return False
    return True


def get_tree_root(edges, nodes):
    """
    Infer the root from directed edges (parent, child).
    """
    child_nodes = {v for _, v in edges}
    roots = [v for v in nodes if v not in child_nodes]
    return roots[0] if roots else None


def build_children_map(edges, nodes):
    children = {v: [] for v in nodes}
    parent = {}
    for u, v in edges:
        children[u].append(v)
        parent[v] = u
    return children, parent


def postorder_from_root(root, children):
    order = []

    def dfs(node):
        for c in children.get(node, []):
            dfs(c)
        order.append(node)

    if root is not None:
        dfs(root)
    return order


def preorder_from_root(root, children):
    order = []

    def dfs(node):
        order.append(node)
        for c in children.get(node, []):
            dfs(c)

    if root is not None:
        dfs(root)
    return order


# -----------------------------
# Random Tree Generation
# -----------------------------

def generate_random_tree(total_nodes):
    """
    Generate a random rooted tree with total_nodes nodes, where each internal
    node has at least 2 children. Returns edges list and leaf node list.
    """
    if total_nodes < 3:
        raise ValueError("total_nodes must be at least 3 to allow 2 children at the root.")

    max_internal = (total_nodes - 1) // 2
    if max_internal < 1:
        raise ValueError("No feasible internal node count for the given total_nodes.")

    # Choose number of internal nodes uniformly from feasible range
    num_internal = random.randint(1, max_internal)
    num_leaves = total_nodes - num_internal

    while True:
        edges = []
        child_counts = {i: 0 for i in range(num_internal)}

        # Assign parents for internal nodes (except root) to ensure acyclicity
        for i in range(1, num_internal):
            parent = random.choice(range(i))
            edges.append((parent, i))
            child_counts[parent] += 1

        # Check if we have enough leaves to satisfy the 2-children constraint
        deficit = sum(max(0, 2 - child_counts[i]) for i in range(num_internal))
        if deficit > num_leaves:
            continue

        leaf_nodes = list(range(num_internal, total_nodes))
        random.shuffle(leaf_nodes)

        # Ensure each internal node has at least 2 children
        for internal_node in range(num_internal):
            needed = max(0, 2 - child_counts[internal_node])
            for _ in range(needed):
                if not leaf_nodes:
                    break
                leaf = leaf_nodes.pop()
                edges.append((internal_node, leaf))
                child_counts[internal_node] += 1

        # Distribute remaining leaves randomly among internal nodes
        for leaf in leaf_nodes:
            parent = random.choice(range(num_internal))
            edges.append((parent, leaf))
            child_counts[parent] += 1

        return edges, list(range(num_internal, total_nodes))

    raise ValueError("Failed to generate a valid tree after multiple attempts.")


# -----------------------------
# Random Interval Generation
# -----------------------------

def generate_random_intervals(min_val, max_val, num_intervals, min_length=1):
    """
    Generate random intervals that could be laminar.
    Returns a list of interval strings like "[1,5]".
    """
    intervals = []
    range_size = max_val - min_val + 1

    for _ in range(num_intervals):
        attempts = 0
        while attempts < 50:
            length = random.randint(min_length, max(min_length, range_size // 2))
            start = random.randint(min_val, max_val - length)
            end = start + length
            interval_str = f"[{start},{end}]"
            if interval_str not in intervals:
                intervals.append(interval_str)
                break
            attempts += 1

    return intervals


def generate_random_laminar_intervals_from_list(leaf_nodes, available_intervals, max_per_leaf=3):
    """
    Assign random laminar intervals to each leaf node from the available intervals.
    """
    def parse_interval(s):
        a, b = map(int, s.strip("[]").split(","))
        return (a, b)

    # parse available intervals to tuples
    available_tuples = [parse_interval(s) for s in available_intervals]

    leaf_intervals = {}
    for leaf in leaf_nodes:
        attempts = 0
        while attempts < 100:
            num_intervals = random.randint(1, min(max_per_leaf, len(available_tuples)))
            candidate = set(random.sample(available_tuples, num_intervals))
            if is_laminar_interval_set(candidate):
                leaf_intervals[leaf] = {f"[{a},{b}]" for (a, b) in candidate}
                break
            attempts += 1
        if leaf not in leaf_intervals:
            a, b = random.choice(available_tuples)
            leaf_intervals[leaf] = {f"[{a},{b}]"}

    return leaf_intervals


def generate_random_root_intervals_from_list(available_intervals, max_per_root=3):
    """
    Generate a laminar interval set for the root from the available intervals.
    """
    root_dict = generate_random_laminar_intervals_from_list([0], available_intervals, max_per_leaf=max_per_root)
    return root_dict[0]


def limit_interval_pool_for_fixed_root(available_intervals, root_intervals, max_candidate_intervals=10):
    """
    Restrict the interval pool so the fixed-root candidate universe stays bounded.

    The LP solver builds candidate intervals from the cross-product of distinct
    starts and ends, so this helper bounds that induced universe rather than
    just the number of sampled intervals.
    """
    def induced_candidate_count(intervals):
        starts = sorted({parse_interval_str(interval)[0] for interval in intervals})
        ends = sorted({parse_interval_str(interval)[1] for interval in intervals})
        return sum(1 for a in starts for b in ends if a < b)

    root_list = sorted(set(root_intervals or []))
    if induced_candidate_count(root_list) > max_candidate_intervals:
        raise ValueError(
            f"Fixed root already induces more than {max_candidate_intervals} "
            f"candidate intervals."
        )

    limited = list(root_list)
    for interval in available_intervals:
        if interval in root_list or interval in limited:
            continue
        proposal = limited + [interval]
        if induced_candidate_count(proposal) <= max_candidate_intervals:
            limited.append(interval)

    return limited


# -----------------------------
# Exact brute-force Sankoff validation
# -----------------------------

def enumerate_laminar_states(candidate_intervals, max_states=None, verbose=False):
    """
    Enumerate all laminar subsets of candidate_intervals.
    Each state is a frozenset of interval tuples.

    WARNING: exponential in the number of candidate intervals.
    """
    intervals = sorted(set(candidate_intervals))
    states = []

    def backtrack(idx, chosen):
        if max_states is not None and len(states) >= max_states:
            return
        if idx == len(intervals):
            states.append(frozenset(chosen))
            return

        # Skip current interval
        backtrack(idx + 1, chosen)

        # Include current interval if laminar with the chosen set
        current = intervals[idx]
        ok = True
        for other in chosen:
            if crosses(current, other):
                ok = False
                break
        if ok:
            chosen.append(current)
            backtrack(idx + 1, chosen)
            chosen.pop()

    backtrack(0, [])

    if verbose:
        print(f"Enumerated {len(states)} laminar states from {len(intervals)} candidate intervals.")

    return states


def state_features(state):
    """
    For a laminar TAD state (frozenset of intervals),
    return interval set, start positions, and end positions.
    """
    intervals = set(state)
    starts = {a for a, _ in intervals}
    ends = {b for _, b in intervals}
    return intervals, starts, ends


def sankoff_bruteforce_laminar(
    edges,
    leaf_intervals,
    candidate_intervals,
    Ct=1.0,
    Cb=1.0,
    root_intervals=None,
    verbose=False,
    max_states=50000,
):
    """
    Exact Sankoff-style solver over all laminar TAD configurations.

    State space:
        all laminar subsets of candidate_intervals

    Edge cost between states S and T:
        Ct * |S xor T| + Cb * (|starts(S) xor starts(T)| + |ends(S) xor ends(T)|)

    Leaves are fixed to their observed TAD sets.
    Root is optionally fixed to root_intervals.
    """
    if not edges:
        return None, None, None, None

    nodes = sorted({u for e in edges for u in e} | {v for e in edges for v in e})
    root = get_tree_root(edges, nodes)
    children, parent = build_children_map(edges, nodes)
    postorder = postorder_from_root(root, children)

    # Candidate states
    states = enumerate_laminar_states(candidate_intervals, max_states=max_states, verbose=verbose)
    if not states:
        return None, None, None, None

    state_index = {s: i for i, s in enumerate(states)}
    nstates = len(states)

    if verbose:
        print(f"Exact solver state count: {nstates}")

    # Precompute features and pairwise costs
    state_ints = []
    state_starts = []
    state_ends = []
    for s in states:
        ints, starts, ends = state_features(s)
        state_ints.append(ints)
        state_starts.append(starts)
        state_ends.append(ends)

    def edge_cost(i, j):
        return (
            Ct * len(state_ints[i] ^ state_ints[j])
            + Cb * (len(state_starts[i] ^ state_starts[j]) + len(state_ends[i] ^ state_ends[j]))
        )

    # Fixed states for leaves and root
    fixed_state_by_node = {}

    for leaf, intset_strs in leaf_intervals.items():
        fixed = frozenset(parse_interval_str(s) for s in intset_strs)
        if not is_laminar_interval_set(fixed):
            raise ValueError(f"Leaf {leaf} has a non-laminar interval set: {sorted(intset_strs)}")
        fixed_state_by_node[leaf] = fixed

    if root is not None and root_intervals is not None:
        root_fixed = frozenset(parse_interval_str(s) for s in root_intervals)
        if not is_laminar_interval_set(root_fixed):
            raise ValueError(f"Root intervals are not laminar: {sorted(root_intervals)}")
        fixed_state_by_node[root] = root_fixed

    for node, fixed_state in fixed_state_by_node.items():
        if fixed_state not in state_index:
            raise ValueError(
                f"Fixed state at node {node} is not in the enumerated laminar state space. "
                f"State={sorted(list(fixed_state))}"
            )

    INF = float("inf")
    dp = {v: [INF] * nstates for v in nodes}
    choice = {v: [None] * nstates for v in nodes}

    # Fixed-node initialization: keep the prescribed state feasible, but still
    # let internal fixed nodes accumulate the optimal child contributions below.
    for v in nodes:
        if v in fixed_state_by_node:
            fixed_idx = state_index[fixed_state_by_node[v]]
            dp[v][fixed_idx] = 0.0

    # DP in postorder
    for v in postorder:
        child_list = children.get(v, [])
        if not child_list:
            # Unfixed leaf not expected in this problem, but keep it safe
            if v not in fixed_state_by_node:
                for s_idx in range(nstates):
                    dp[v][s_idx] = 0.0
            continue

        if v in fixed_state_by_node:
            state_indices = [state_index[fixed_state_by_node[v]]]
        else:
            state_indices = range(nstates)

        for s_idx in state_indices:
            total = 0.0
            child_choices = {}

            for c in child_list:
                best_val = INF
                best_t = None
                for t_idx in range(nstates):
                    val = dp[c][t_idx] + edge_cost(s_idx, t_idx)
                    if val < best_val:
                        best_val = val
                        best_t = t_idx
                total += best_val
                child_choices[c] = best_t

            dp[v][s_idx] = total
            choice[v][s_idx] = child_choices

    # Choose best root state
    if root is None:
        return None, None, None, None

    if root in fixed_state_by_node:
        root_idx = state_index[fixed_state_by_node[root]]
        exact_cost = dp[root][root_idx]
    else:
        root_idx = min(range(nstates), key=lambda i: dp[root][i])
        exact_cost = dp[root][root_idx]

    if math.isinf(exact_cost):
        return None, None, states, dp

    # Reconstruct assignments
    assignment_indices = {}

    def best_child_state(parent_state_idx, child):
        best_val = INF
        best_t = None
        for t_idx in range(nstates):
            val = dp[child][t_idx] + edge_cost(parent_state_idx, t_idx)
            if val < best_val:
                best_val = val
                best_t = t_idx
        return best_t

    def reconstruct(node, s_idx):
        assignment_indices[node] = s_idx
        for c in children.get(node, []):
            node_choice = choice[node][s_idx]
            if node_choice is None:
                t_idx = best_child_state(s_idx, c)
            else:
                t_idx = node_choice[c]
            reconstruct(c, t_idx)

    reconstruct(root, root_idx)

    exact_assignments = {
        v: states[idx] for v, idx in assignment_indices.items()
    }

    return exact_cost, exact_assignments, states, dp


# -----------------------------
# Main LP simulation with optional exact checking
# -----------------------------

def run_single_simulation(
    total_nodes,
    min_val,
    max_val,
    num_intervals,
    max_per_leaf=5,
    Ct=1.0,
    Cb=1.0,
    mode="convex",
    use_fitch_hartigan=False,
    verbose=False,
    root_intervals=None,
    exact_check=False,
    exact_max_candidate_intervals=14,
):
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

    # Optional root intervals: if supplied, pass them to the LP solver;
    # if not supplied, nothing changes for the existing behavior.
    if verbose:
        print(f"\nEdges: {edges}")
        print(f"Leaf nodes: {leaf_nodes}")
        print(f"Available intervals (sampled): {available_intervals}")
        print(f"Leaf intervals: {leaf_intervals}")
        if root_intervals is not None:
            print(f"Root intervals: {sorted(root_intervals)}")

    lp_cost, is_integral, node_assignments, x, nodes, intervals = solve_lp_from_edges_and_leaves(
        edges,
        leaf_intervals,
        Ct=Ct,
        Cb=Cb,
        mode=mode,
        require_laminar_internal=True,
        use_fitch_hartigan=use_fitch_hartigan,
        verbose=verbose,
        root_intervals=root_intervals,
    )

    if lp_cost is None:
        return None, None, edges, leaf_intervals, x, nodes, intervals

    print("\nLP solution (non-zero values at internal nodes):")
    for v in nodes:
        if v not in leaf_intervals:
            print(f"  Node {v}:")
            for (a, b) in node_assignments[v]:
                print(f"    Interval ({a},{b})")

    # Optional exact validation by brute-force Sankoff over laminar states
    if exact_check:
        candidate_intervals = list(intervals)
        if len(candidate_intervals) > exact_max_candidate_intervals:
            print(
                f"\nExact check skipped: candidate interval universe size = {len(candidate_intervals)} "
                f"exceeds threshold {exact_max_candidate_intervals}."
            )
        else:
            try:
                exact_cost, exact_assignments, state_space, dp = sankoff_bruteforce_laminar(
                    edges=edges,
                    leaf_intervals=leaf_intervals,
                    candidate_intervals=candidate_intervals,
                    Ct=Ct,
                    Cb=Cb,
                    root_intervals=root_intervals,
                    verbose=verbose,
                    max_states=50000,
                )

                if exact_cost is None:
                    print("\nExact check failed: no feasible laminar state space solution found.")
                else:
                    matches_exact = abs(lp_cost - exact_cost) <= 1e-6
                    print(f"\nExact Sankoff cost: {exact_cost:.6f}")
                    print(f"LP cost:          {lp_cost:.6f}")
                    print(f"LP matches exact:  {matches_exact}")

                    if not matches_exact:
                        raise SystemExit(
                            "Stopping simulation: LP and exact Sankoff costs do not match "
                            f"(LP={lp_cost:.6f}, exact={exact_cost:.6f})."
                        )

                    if verbose:
                        print("\nExact assignment (internal nodes):")
                        for v in nodes:
                            if v not in leaf_intervals:
                                st = exact_assignments.get(v, frozenset())
                                print(f"  Node {v}: {sorted(st)}")
            except ValueError as exc:
                print(f"\nExact check failed: {exc}")

    return lp_cost, is_integral, edges, leaf_intervals, x, nodes, intervals


def run_single_simulation_fixed_root(
    total_nodes,
    min_val,
    max_val,
    num_intervals,
    max_per_leaf=5,
    Ct=1.0,
    Cb=1.0,
    mode="convex",
    use_fitch_hartigan=False,
    verbose=False,
    root_intervals=None,
    root_max_per=3,
    exact_check=False,
    exact_max_candidate_intervals=10,
):
    """
    Separate convenience wrapper for the fixed-root case.

    If root_intervals is None, a random laminar root interval set is generated
    from the sampled available intervals.
    """
    try:
        edges, leaf_nodes = generate_random_tree(total_nodes)
    except ValueError as exc:
        if verbose:
            print(f"Skipped simulation due to invalid tree parameters: {exc}")
        return None, None, [], {}, None, None, None

    available_intervals = generate_random_intervals(min_val, max_val, num_intervals)

    if root_intervals is None:
        root_intervals = generate_random_root_intervals_from_list(
            available_intervals, max_per_root=min(root_max_per, num_intervals)
        )
    else:
        root_intervals = set(root_intervals)

    try:
        available_intervals = limit_interval_pool_for_fixed_root(
            available_intervals,
            root_intervals,
            max_candidate_intervals=10,
        )
    except ValueError as exc:
        if verbose:
            print(f"Skipped simulation due to fixed-root interval cap: {exc}")
        return None, None, edges, {}, None, None, None

    leaf_intervals = generate_random_laminar_intervals_from_list(
        leaf_nodes, available_intervals, max_per_leaf=min(max_per_leaf, len(available_intervals))
    )

    if verbose:
        print(f"\nEdges: {edges}")
        print(f"Leaf nodes: {leaf_nodes}")
        print(f"Available intervals (sampled): {available_intervals}")
        print(f"Leaf intervals: {leaf_intervals}")
        print(f"Fixed root intervals: {sorted(root_intervals)}")

    lp_cost, is_integral, node_assignments, x, nodes, intervals = solve_lp_from_edges_and_leaves(
        edges,
        leaf_intervals,
        Ct=Ct,
        Cb=Cb,
        mode=mode,
        require_laminar_internal=True,
        use_fitch_hartigan=use_fitch_hartigan,
        verbose=verbose,
        root_intervals=root_intervals,
    )

    if lp_cost is None:
        return None, None, edges, leaf_intervals, x, nodes, intervals

    print("\nLP solution (non-zero values at internal nodes):")
    for v in nodes:
        if v not in leaf_intervals:
            print(f"  Node {v}:")
            for (a, b) in node_assignments[v]:
                print(f"    Interval ({a},{b})")

    if exact_check:
        candidate_intervals = list(intervals)
        if len(candidate_intervals) > exact_max_candidate_intervals:
            print(
                f"\nExact check skipped: candidate interval universe size = {len(candidate_intervals)} "
                f"exceeds threshold {exact_max_candidate_intervals}."
            )
        else:
            try:
                exact_cost, exact_assignments, state_space, dp = sankoff_bruteforce_laminar(
                    edges=edges,
                    leaf_intervals=leaf_intervals,
                    candidate_intervals=candidate_intervals,
                    Ct=Ct,
                    Cb=Cb,
                    root_intervals=root_intervals,
                    verbose=verbose,
                    max_states=50000,
                )

                if exact_cost is None:
                    print("\nExact check failed: no feasible laminar state space solution found.")
                else:
                    matches_exact = abs(lp_cost - exact_cost) <= 1e-6
                    print(f"\nExact Sankoff cost: {exact_cost:.6f}")
                    print(f"LP cost:          {lp_cost:.6f}")
                    print(f"LP matches exact:  {matches_exact}")

                    if not matches_exact:
                        raise SystemExit(
                            "Stopping simulation: LP and exact Sankoff costs do not match "
                            f"(LP={lp_cost:.6f}, exact={exact_cost:.6f})."
                        )

                    if verbose:
                        print("\nExact assignment (internal nodes):")
                        for v in nodes:
                            if v not in leaf_intervals:
                                st = exact_assignments.get(v, frozenset())
                                print(f"  Node {v}: {sorted(st)}")
            except ValueError as exc:
                print(f"\nExact check failed: {exc}")

    return lp_cost, is_integral, edges, leaf_intervals, x, nodes, intervals


def run_given_instance(
    edges,
    leaf_intervals,
    Ct=1.0,
    Cb=1.0,
    mode="convex",
    use_fitch_hartigan=False,
    verbose=True,
    root_intervals=None,
    exact_check=False,
    exact_max_candidate_intervals=14,
):
    """
    Run the LP solver on one manually provided instance.
    """
    normalized_leaf_intervals = {
        node: set(intervals) for node, intervals in leaf_intervals.items()
    }

    if verbose:
        print(f"\nEdges: {edges}")
        print(f"Leaf intervals: {normalized_leaf_intervals}")
        if root_intervals is not None:
            print(f"Root intervals: {sorted(root_intervals)}")

    lp_cost, is_integral, node_assignments, x, nodes, intervals = solve_lp_from_edges_and_leaves(
        edges,
        normalized_leaf_intervals,
        Ct=Ct,
        Cb=Cb,
        mode=mode,
        require_laminar_internal=True,
        use_fitch_hartigan=use_fitch_hartigan,
        verbose=verbose,
        root_intervals=root_intervals,
    )

    if lp_cost is None:
        return None, None, edges, normalized_leaf_intervals, x, nodes, intervals

    print("\nLP solution (non-zero values at internal nodes):")
    for v in nodes:
        if v not in normalized_leaf_intervals:
            print(f"  Node {v}:")
            for (a, b) in node_assignments[v]:
                print(f"    Interval ({a},{b})")

    if exact_check:
        candidate_intervals = list(intervals)
        if len(candidate_intervals) > exact_max_candidate_intervals:
            print(
                f"\nExact check skipped: candidate interval universe size = {len(candidate_intervals)} "
                f"exceeds threshold {exact_max_candidate_intervals}."
            )
        else:
            try:
                exact_cost, exact_assignments, state_space, dp = sankoff_bruteforce_laminar(
                    edges=edges,
                    leaf_intervals=normalized_leaf_intervals,
                    candidate_intervals=candidate_intervals,
                    Ct=Ct,
                    Cb=Cb,
                    root_intervals=root_intervals,
                    verbose=verbose,
                    max_states=50000,
                )

                if exact_cost is None:
                    print("\nExact check failed: no feasible laminar state space solution found.")
                else:
                    matches_exact = abs(lp_cost - exact_cost) <= 1e-6
                    print(f"\nExact Sankoff cost: {exact_cost:.6f}")
                    print(f"LP cost:          {lp_cost:.6f}")
                    print(f"LP matches exact:  {matches_exact}")
            except ValueError as exc:
                print(f"\nExact check failed: {exc}")

    return lp_cost, is_integral, edges, normalized_leaf_intervals, x, nodes, intervals


# -----------------------------
# Run multiple simulations
# -----------------------------

def run_simulation(
    num_simulations=100,
    min_nodes=3,
    max_nodes=20,
    min_val=1,
    max_val=10,
    num_intervals=8,
    max_per_leaf=5,
    Ct=2.0,
    Cb=1.0,
    mode="convex",
    use_fitch_hartigan=False,
    root_intervals=None,
    exact_check=False,
    exact_max_candidate_intervals=14,
):
    """
    Run multiple simulations using the LP-only verification,
    with optional exact Sankoff validation.
    """
    print(f"Running {num_simulations} simulations (mode={mode})...")
    print(f"Tree structure: {min_nodes}-{max_nodes} total nodes")
    print(f"Intervals: range [{min_val}, {max_val}], sampled {num_intervals} total intervals")
    print(f"Max intervals per leaf: {max_per_leaf}")
    print(f"Costs: Ct={Ct}, Cb={Cb}\n")

    for sim in range(1, num_simulations + 1):
        total_nodes = random.randint(min_nodes, max_nodes)
        print(f"Simulation {sim} (TotalNodes={total_nodes})...", end=" ")

        result = run_single_simulation(
            total_nodes,
            min_val,
            max_val,
            num_intervals,
            max_per_leaf=max_per_leaf,
            Ct=Ct,
            Cb=Cb,
            mode=mode,
            use_fitch_hartigan=use_fitch_hartigan,
            verbose=False,
            root_intervals=root_intervals,
            exact_check=exact_check,
            exact_max_candidate_intervals=exact_max_candidate_intervals,
        )

        if result[0] is None:
            print("Skipped")
            continue

        lp_cost, is_integral, edges, leaf_intervals, x_vars, nodes, intervals = result

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
            if root_intervals is not None:
                print(f"\nFixed root intervals: {sorted(root_intervals)}")
            print("\nLP solution (non-integral values at internal nodes):")
            for v in nodes:
                if v not in leaf_intervals:
                    print(f"  Node {v}:")
                    for (a, b) in intervals:
                        val = x_vars[v, a, b].X
                        if val > 1e-6 and val < 1 - 1e-6:
                            print(f"    x[{v},({a},{b})] = {val:.6f}")
            return

        print(f"OK (LP={lp_cost:.2f}, Integral={is_integral})")

    print(f"\nCompleted {num_simulations} simulations. No non-integral LP solutions found.")


def run_simulation_fixed_root(
    num_simulations=100,
    min_nodes=3,
    max_nodes=20,
    min_val=1,
    max_val=10,
    num_intervals=8,
    max_per_leaf=5,
    Ct=2.0,
    Cb=1.0,
    mode="convex",
    use_fitch_hartigan=False,
    root_intervals=None,
    root_max_per=3,
    exact_check=False,
    exact_max_candidate_intervals=10,
):
    """
    Separate multi-simulation wrapper for the fixed-root case.
    If root_intervals is None, a random root set is generated per simulation.
    """
    print(f"Running {num_simulations} simulations with fixed root (mode={mode})...")
    print(f"Tree structure: {min_nodes}-{max_nodes} total nodes")
    print(f"Intervals: range [{min_val}, {max_val}], sampled {num_intervals} total intervals")
    print(f"Max intervals per leaf: {max_per_leaf}")
    print(f"Costs: Ct={Ct}, Cb={Cb}\n")

    for sim in range(1, num_simulations + 1):
        total_nodes = random.randint(min_nodes, max_nodes)
        print(f"Simulation {sim} (TotalNodes={total_nodes})...", end=" ")

        if root_intervals is None:
            # Generate a fresh random root set for each simulation
            # by sampling after the tree/interval pool is created inside the single-simulation function.
            result = run_single_simulation_fixed_root(
                total_nodes,
                min_val,
                max_val,
                num_intervals,
                max_per_leaf=max_per_leaf,
                Ct=Ct,
                Cb=Cb,
                mode=mode,
                use_fitch_hartigan=use_fitch_hartigan,
                verbose=True,
                root_intervals=None,
                root_max_per=root_max_per,
                exact_check=exact_check,
                exact_max_candidate_intervals=exact_max_candidate_intervals,
            )
        else:
            result = run_single_simulation_fixed_root(
                total_nodes,
                min_val,
                max_val,
                num_intervals,
                max_per_leaf=max_per_leaf,
                Ct=Ct,
                Cb=Cb,
                mode=mode,
                use_fitch_hartigan=use_fitch_hartigan,
                verbose=False,
                root_intervals=root_intervals,
                root_max_per=root_max_per,
                exact_check=exact_check,
                exact_max_candidate_intervals=exact_max_candidate_intervals,
            )

        if result[0] is None:
            print("Skipped")
            continue

        lp_cost, is_integral, edges, leaf_intervals, x_vars, nodes, intervals = result

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

        print(f"OK (LP={lp_cost:.2f}, Integral={is_integral})")

    print(f"\nCompleted {num_simulations} simulations. No non-integral LP solutions found.")


# -----------------------------
# Main Entry Point
# -----------------------------
if __name__ == "__main__":
    # Simulation parameters
    NUM_SIMULATIONS = 1
    MIN_NODES = 3
    MAX_NODES = 15
    MIN_VAL = 1
    MAX_VAL = 10
    NUM_INTERVALS = 100
    MAX_PER_LEAF = 5
    Ct = 1.0
    Cb = 1.0
    MODE = "convex"  # 'convex' | 'lex_boundary_first' | 'lex_tad_first'
    USE_FITCH_HARTIGAN = False

    # Exact validation is exponential; keep it off for large instances.
    EXACT_CHECK = True

    # Set this to a list like {"[1,10]", "[12,20]"} to fix the root,
    # or leave as None for the original leaf-only behavior.
    ROOT_INTERVALS = None

    # Example fixed-root simulation:
    # ROOT_INTERVALS = {"[1,10]", "[20,30]"}
    RUN_GIVEN_INSTANCE = True
    GIVEN_INSTANCE_EDGES = [(0, 1), (1, 2), (1, 3), (2, 4), (2, 5)]
    GIVEN_INSTANCE_LEAF_INTERVALS = {
        0: {"[1,10]", "[20,30]"},  # Root node with fixed intervals
        3: {"[1,5]"},
        4: {"[1,10]"},
        5: {"[20,30]"},
    }

    
    if RUN_GIVEN_INSTANCE:
        run_given_instance(
            edges=GIVEN_INSTANCE_EDGES,
            leaf_intervals=GIVEN_INSTANCE_LEAF_INTERVALS,
            Ct=Ct,
            Cb=Cb,
            mode=MODE,
            use_fitch_hartigan=USE_FITCH_HARTIGAN,
            verbose=True,
            root_intervals=ROOT_INTERVALS,
            exact_check=EXACT_CHECK,
            exact_max_candidate_intervals=14,
        )
        raise SystemExit(0)

    # run_simulation(
    #     num_simulations=NUM_SIMULATIONS,
    #     min_nodes=MIN_NODES,
    #     max_nodes=MAX_NODES,
    #     min_val=MIN_VAL,
    #     max_val=MAX_VAL,
    #     num_intervals=NUM_INTERVALS,
    #     max_per_leaf=MAX_PER_LEAF,
    #     Ct=Ct,
    #     Cb=Cb,
    #     mode=MODE,
    #     use_fitch_hartigan=USE_FITCH_HARTIGAN,
    #     root_intervals=ROOT_INTERVALS,
    #     exact_check=EXACT_CHECK,
    # )

    # If you want the separate fixed-root wrapper instead, use this:
    # run_simulation_fixed_root(
    #     num_simulations=NUM_SIMULATIONS,
    #     min_nodes=MIN_NODES,
    #     max_nodes=MAX_NODES,
    #     min_val=MIN_VAL,
    #     max_val=MAX_VAL,
    #     num_intervals=NUM_INTERVALS,
    #     max_per_leaf=MAX_PER_LEAF,
    #     Ct=Ct,
    #     Cb=Cb,
    #     mode=MODE,
    #     use_fitch_hartigan=USE_FITCH_HARTIGAN,
    #     root_intervals=ROOT_INTERVALS,
    #     exact_check=EXACT_CHECK,
    #     exact_max_candidate_intervals=10,
    # )
