import gurobipy as gp
from gurobipy import GRB
import re

# -----------------------------
# Main LP solver (updated)
# -----------------------------
def solve_lp_from_edges_and_leaves(
    edges,
    leaf_intervals,
    Ct=1.0,
    Cb=1.0,
    mode="convex",
    require_laminar_internal=True,
    use_fitch_hartigan=False,
    verbose=False,
    impute=False,
    root_intervals=None,   # NEW
):
    """
    Solve the LP for a given tree and leaf interval assignments.

    Parameters
    ----------
    edges : list[tuple]
        Tree edges as (parent, child).
    leaf_intervals : dict
        Mapping leaf node -> set of interval strings like "[a,b]".
    ...
    impute : bool
        If True, allow missing intervals at fixed nodes to be imputed.
    root_intervals : iterable[str] or None
        If given, fix the root intervals exactly the same way as leaf intervals.
    Ct, Cb : float
        Cost parameters.
    mode : str
        'convex' | 'lex_boundary_first' | 'lex_tad_first'.
    require_laminar_internal : bool
        If True, only return a solution where internal nodes are laminar.
    use_fitch_hartigan : bool
        If True, restrict candidate boundary positions per node using Fitch-Hartigan.
    verbose : bool
        Print diagnostic info if True.

    Returns
    -------
    lp_cost : float or None
    is_integral : bool or None
    node_assignments : dict
        Mapping node -> set of (a,b) intervals (from x vars).
    x_vars : gurobipy tupledict
    nodes : list
    intervals : list
    """
    if not edges:
        return None, None, {}, None, [], []

    def parse_interval(s):
        a, b = map(int, re.findall(r"\d+", s))
        return (a, b)

    nodes = sorted({u for e in edges for u in e} | {v for e in edges for v in e})

    parent_nodes = {v for _, v in edges}
    roots = [v for v in nodes if v not in parent_nodes]
    root = roots[0] if roots else None  # same convention as your FH helper

    root_interval_set = set(root_intervals) if root_intervals is not None else set()

    def parse_interval(s):
        a, b = map(int, re.findall(r"\d+", s))
        return (a, b)

    def fitch_hartigan_majority_sets(edges_list, leaf_sets, all_nodes, all_positions):
        """Assign boundary states with a bottom-up majority pass and top-down resolution."""
        children = {v: [] for v in all_nodes}
        parent = {}
        for u, v in edges_list:
            children[u].append(v)
            parent[v] = u

        roots = [v for v in all_nodes if v not in parent]
        root = roots[0] if roots else None

        postorder = []

        def dfs(node):
            for c in children.get(node, []):
                dfs(c)
            postorder.append(node)

        if root is not None:
            dfs(root)

        node_states = {v: {} for v in all_nodes}
        for v in postorder:
            if v in leaf_sets:
                for p in all_positions:
                    node_states[v][p] = 1 if p in leaf_sets[v] else 0
                continue
            child_list = children.get(v, [])
            for p in all_positions:
                ones = 0
                zeros = 0
                for c in child_list:
                    state = node_states[c].get(p, 0)
                    if state == 1:
                        ones += 1
                    elif state == 0:
                        zeros += 1
                if ones > zeros:
                    node_states[v][p] = 1
                elif zeros > ones:
                    node_states[v][p] = 0
                else:
                    node_states[v][p] = None

        preorder = []

        def dfs_pre(node):
            preorder.append(node)
            for c in children.get(node, []):
                dfs_pre(c)

        if root is not None:
            dfs_pre(root)

        for v in preorder:
            for p in all_positions:
                state = node_states[v].get(p, 0)
                if v == root and state is None:
                    node_states[v][p] = 1
                elif state is None:
                    node_states[v][p] = node_states[parent[v]][p]

        node_sets = {v: set() for v in all_nodes}
        for v in all_nodes:
            for p, state in node_states[v].items():
                if state == 1:
                    node_sets[v].add(p)

        return node_sets

    # Collect start and end positions from leaves
    start_positions = set()
    end_positions = set()

    for v in leaf_intervals:
        for s in leaf_intervals[v]:
            a, b = parse_interval(s)
            start_positions.add(a)
            end_positions.add(b)
            if impute:
                start_positions.add(b)
                end_positions.add(a)

    # NEW: include root intervals in the candidate universe if provided
    if root is not None and root_intervals is not None:
        for s in root_intervals:
            a, b = parse_interval(s)
            start_positions.add(a)
            end_positions.add(b)
            if impute:
                start_positions.add(b)
                end_positions.add(a)

    leaf_start_sets = {v: set() for v in leaf_intervals}
    leaf_end_sets = {v: set() for v in leaf_intervals}
    for v in leaf_intervals:
        for s in leaf_intervals[v]:
            a, b = parse_interval(s)
            leaf_start_sets[v].add(a)
            leaf_end_sets[v].add(b)

    if use_fitch_hartigan:
        node_start_sets = fitch_hartigan_majority_sets(edges, leaf_start_sets, nodes, start_positions)
        node_end_sets = fitch_hartigan_majority_sets(edges, leaf_end_sets, nodes, end_positions)
        if node_start_sets:
            start_positions = set().union(*node_start_sets.values())
        if node_end_sets:
            end_positions = set().union(*node_end_sets.values())
        
                # NEW: ensure root-fixed intervals remain feasible under FH pruning
        if root is not None and root_intervals is not None:
            root_starts = {parse_interval(s)[0] for s in root_intervals}
            root_ends = {parse_interval(s)[1] for s in root_intervals}
            node_start_sets.setdefault(root, set()).update(root_starts)
            node_end_sets.setdefault(root, set()).update(root_ends)

            # also keep these positions in the global candidate universe
            start_positions.update(root_starts)
            end_positions.update(root_ends)
    else:
        node_start_sets = {v: set(start_positions) for v in nodes}
        node_end_sets = {v: set(end_positions) for v in nodes}

    start_positions = sorted(start_positions)
    # reverse end positions to try larger intervals first (heuristic for better LP performance)
    end_positions = sorted(end_positions, reverse=True)

    if not start_positions or not end_positions:
        if verbose:
            print("No start or end positions collected from leaves; skipping.")
        return None, None, {}, None, nodes, []

    # Candidate intervals
    if use_fitch_hartigan:
        node_intervals = {}
        for v in nodes:
            node_intervals[v] = [
                (a, b)
                for a in node_start_sets.get(v, set())
                for b in node_end_sets.get(v, set())
                if a < b
            ]
        interval_universe = sorted(set().union(*node_intervals.values())) if node_intervals else []
        intervals = interval_universe
    else:
        intervals = []
        for a in start_positions:
            for b in end_positions:
                if a < b:
                    intervals.append((a, b))
        intervals = sorted(set(intervals))
        node_intervals = {v: intervals for v in nodes}

    if not intervals:
        if verbose:
            print("No valid candidate intervals (start < end) found; skipping.")
        return None, None, {}, None, nodes, intervals

    internal_nodes = [v for v in nodes if v not in leaf_intervals]
    print("\nInternal node candidate stats:")
    for v in internal_nodes:
        if use_fitch_hartigan:
            start_count = len(node_start_sets.get(v, set()))
            end_count = len(node_end_sets.get(v, set()))
        else:
            start_count = len(start_positions)
            end_count = len(end_positions)
        tad_count = len(node_intervals.get(v, []))
        print(f"  Node {v}: starts={start_count}, ends={end_count}, intervals={tad_count}")

    positions = sorted(set(start_positions) | set(end_positions))

    # Build starts_at and ends_at maps for candidate intervals
    if use_fitch_hartigan:
        starts_at = {v: {p: [] for p in positions} for v in nodes}
        ends_at = {v: {p: [] for p in positions} for v in nodes}
        for v in nodes:
            for (a, b) in node_intervals[v]:
                starts_at[v][a].append((a, b))
                ends_at[v][b].append((a, b))
    else:
        starts_at = {p: [] for p in positions}
        ends_at = {p: [] for p in positions}
        for (a, b) in intervals:
            starts_at[a].append((a, b))
            ends_at[b].append((a, b))
    print("Number of candidate intervals:", len(intervals))
    # print ("Number of crossing interval pairs:", len(cross_pairs))

    # -----------------------------
    # Build LP
    # -----------------------------
    m = gp.Model("TAD_small_parsimony_LP")
    m.setParam("OutputFlag", 0)
    # m.setParam("TimeLimit", 0)

    x = m.addVars([(v, a, b) for v in nodes for (a, b) in node_intervals[v]], lb=0, ub=1, name="x", vtype=GRB.BINARY)
    s = m.addVars([(v, p) for v in nodes for p in positions], lb=0, ub=1, name="s")
    e = m.addVars([(v, p) for v in nodes for p in positions], lb=0, ub=1, name="e")

    dx = m.addVars([(u, v, a, b) for (u, v) in edges for (a, b) in intervals], lb=0, ub=1, name="dx")
    ds = m.addVars([(u, v, p) for (u, v) in edges for p in positions], lb=0, ub=1, name="ds")
    de = m.addVars([(u, v, p) for (u, v) in edges for p in positions], lb=0, ub=1, name="de")

    # Leaf fixing + optional root fixing
    fixed_nodes = dict(leaf_intervals)
    if root is not None and root_intervals is not None:
        fixed_nodes[root] = root_interval_set

    for v in fixed_nodes:
        present = {parse_interval(interval_str) for interval_str in fixed_nodes[v]}
        for (a, b) in node_intervals[v]:
            if (a, b) in present:
                m.addConstr(x[v, a, b] == 1)
            elif impute == False:
                m.addConstr(x[v, a, b] == 0)

    # Interval ↔ boundary constraints
    for v in nodes:
        for (a, b) in node_intervals[v]:
            m.addConstr(x[v, a, b] <= s[v, a])
            m.addConstr(x[v, a, b] <= e[v, b])

        for p in positions:
            if use_fitch_hartigan and p not in node_start_sets.get(v, set()):
                m.addConstr(s[v, p] == 0)
            elif use_fitch_hartigan:
                m.addConstr(s[v, p] <= gp.quicksum(x[v, a, b] for (a, b) in starts_at[v][p]))
            elif starts_at.get(p):
                m.addConstr(s[v, p] <= gp.quicksum(x[v, a, b] for (a, b) in starts_at[p]))
            else:
                m.addConstr(s[v, p] == 0)

            if use_fitch_hartigan and p not in node_end_sets.get(v, set()):
                m.addConstr(e[v, p] == 0)
            elif use_fitch_hartigan:
                m.addConstr(e[v, p] <= gp.quicksum(x[v, a, b] for (a, b) in ends_at[v][p]))
            elif ends_at.get(p):
                m.addConstr(e[v, p] <= gp.quicksum(x[v, a, b] for (a, b) in ends_at[p]))
            else:
                m.addConstr(e[v, p] == 0)

    # Edge difference constraints
    for (u, v) in edges:
        for (a, b) in intervals:
            xu = x.get((u, a, b), 0)
            xv = x.get((v, a, b), 0)
            m.addConstr(dx[u, v, a, b] >= xu - xv)
            m.addConstr(dx[u, v, a, b] >= xv - xu)

        for p in positions:
            m.addConstr(ds[u, v, p] >= s[u, p] - s[v, p])
            m.addConstr(ds[u, v, p] >= s[v, p] - s[u, p])

            m.addConstr(de[u, v, p] >= e[u, p] - e[v, p])
            m.addConstr(de[u, v, p] >= e[v, p] - e[u, p])

    # Objective
    sum_dx = gp.quicksum(dx[u, v, a, b] for (u, v) in edges for (a, b) in intervals)
    sum_ds_de = gp.quicksum(ds[u, v, p] + de[u, v, p] for (u, v) in edges for p in positions)

    if mode == "convex":
        obj = Ct * sum_dx + Cb * sum_ds_de
    elif mode == "lex_boundary_first":
        denom = (len(start_positions) * len(end_positions) * len(edges)) + 1
        obj = Cb * sum_ds_de + (Ct * sum_dx) / denom
    elif mode == "lex_tad_first":
        denom = (len(edges) * (len(start_positions) + len(end_positions))) + 1
        obj = Ct * sum_dx + (Cb * sum_ds_de) / denom
    else:
        raise ValueError("Unknown mode. Use 'convex'|'lex_boundary_first'|'lex_tad_first'.")

    m.setObjective(obj, GRB.MINIMIZE)

    def build_assignments_from_solution():
        assignments = {v: set() for v in nodes}
        for v in nodes:
            for (a, b) in node_intervals[v]:
                if x[v, a, b].X > 0.5:
                    assignments[v].add((a, b))
        return assignments

    def find_laminarity_violations(assignments):
        violations = []
        for v in nodes:
            if v in leaf_intervals:
                continue
            intervals_list = sorted(assignments[v])
            for i in range(len(intervals_list)):
                a, b = intervals_list[i]
                for j in range(i + 1, len(intervals_list)):
                    c, d = intervals_list[j]
                    if (
                        (a < c < b < d)
                        or
                        (c < a < d < b)
                    ):
                        violations.append((v, (a, b), (c, d)))
        return violations

    max_cut_rounds = 50
    cut_round = 0
    last_lp_cost = None
    selected_assignment = {}
    violations = []
    while True:
        m.optimize()
        if m.status != GRB.OPTIMAL:
            if verbose:
                print("Gurobi did not return OPTIMAL. Status:", m.status)
            return None, None, {}, None, nodes, intervals

        last_lp_cost = m.ObjVal
        selected_assignment = build_assignments_from_solution()
        violations = find_laminarity_violations(selected_assignment)

        if not violations:
            break

        for v, (a, b), (c, d) in violations:
            m.addConstr(x[v, a, b] + x[v, c, d] <= 1)
        cut_round += 1
        if verbose:
            print(f"Added {len(violations)} laminarity cuts (round {cut_round}).")

    lp_cost = last_lp_cost
    if violations and require_laminar_internal:
        if verbose:
            print("No laminar solution found within cut limit.")
        return lp_cost, False, {}, x, nodes, intervals

    # ---- Integrality Check ----
    is_integral = True
    tol = 1e-6

    for v in nodes:
        if v not in leaf_intervals:
            for (a, b) in node_intervals[v]:
                val = x[v, a, b].X
                if tol < val < 1 - tol:
                    if verbose:
                        print(f"Non-integral value at node {v}, interval ({a},{b}): {val}")
                    is_integral = False
                    break
            if not is_integral:
                break

    return lp_cost, is_integral, selected_assignment, x, nodes, intervals
