from itertools import product, combinations
from collections import defaultdict
import math
import copy
import random

# ========================================
# USER INPUT: TREE STRUCTURE AND LEAF ASSIGNMENTS
# ========================================

edges = [
    (0, 1),
    (0, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 6),
    (2, 7),
]

available_intervals = ["[1,10]","[1,5]","[6,10]", "[1,8]", "[6,8]", "[9,10]"]

# Identify leaves (nodes that appear as children but never as parents)
parent_nodes = set(parent for parent, _ in edges)
child_nodes = set(child for _, child in edges)
leaf_nodes = child_nodes - parent_nodes

# Example leaf assignments (immutable)
# 3: {'[1,3]', '[4,5]', '[6,10]'},
# 4: {'[1,10]'},
# 5: {'[1,3]', '[1,5]'},
# 6: {'[1,10]', '[1,5]','[1,3]', '[4,5]', '[6,10]'},
# 7: {'[1,10]', '[1,5]', '[6,10]', '[4,5]', '[1,3]'},

def parse_interval(s):
    a,b = s.strip("[]").split(",")
    return int(a), int(b)

def laminar(i, j):
    ai, bi = parse_interval(i)
    aj, bj = parse_interval(j)

    # disjoint
    if bi < aj or bj < ai:
        return True
    # nested
    if ai <= aj and bj <= bi:
        return True
    if aj <= ai and bi <= bj:
        return True
    return False

def laminar_subset(intervals):
    """Random laminar subset from a given interval list."""
    shuffled = intervals[:]
    random.shuffle(shuffled)

    chosen = []
    for I in shuffled:
        if all(laminar(I, J) for J in chosen):
            # random accept to keep subset random-sized
            if random.random() < 0.5 or not chosen:
                chosen.append(I)
    return set(chosen)

# Assign laminar random subsets to leaves
leaf_assignments = {
    leaf: laminar_subset(available_intervals)
    for leaf in leaf_nodes
}

# ========================================
# BUILD TREE
# ========================================

def build_tree_from_edges(edges):
    tree = defaultdict(list)
    nodes = set()
    children = set()
    for p, c in edges:
        tree[p].append(c)
        nodes |= {p, c}
        children.add(c)
    roots = nodes - children
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, found {len(roots)}: {roots}")
    root = roots.pop()
    for n in nodes:
        tree.setdefault(n, [])
    return dict(tree), root

tree, root = build_tree_from_edges(edges)

def postorder_nodes(tree, root):
    order = []
    def dfs(u):
        for v in tree[u]:
            dfs(v)
        order.append(u)
    dfs(root)
    return order

postorder = postorder_nodes(tree, root)

# ========================================
# INTERVAL HELPERS
# ========================================



def is_union(parent, c1, c2):
    a1,b1 = parse_interval(c1)
    a2,b2 = parse_interval(c2)
    ap,bp = parse_interval(parent)
    return min(a1,a2) == ap and max(b1,b2) == bp and (b1+1==a2 or b2+1==a1)

intervals = sorted({I for S in leaf_assignments.values() for I in S})

# ========================================
# TAD DISTANCE (unchanged)
# ========================================

INS_DEL = 1.0
MERGE_SPLIT = 2.1

def tad_distance(t1, t2):
    t1 = set(t1)
    t2 = set(t2)
    cost = 0.0

    common = t1 & t2
    t1 -= common
    t2 -= common

    # handle splits (t1 -> t2)
    changed = True
    while changed:
        changed = False
        for I in list(t1):
            for J in list(t2):
                for K in list(t2):
                    if J != K and is_union(I, J, K):
                        cost += MERGE_SPLIT
                        t1.remove(I)
                        t2.remove(J)
                        t2.remove(K)
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break

    # handle merges (t2 -> t1)
    changed = True
    while changed:
        changed = False
        for I in list(t2):
            for J in list(t1):
                for K in list(t1):
                    if J != K and is_union(I, J, K):
                        cost += MERGE_SPLIT
                        t2.remove(I)
                        t1.remove(J)
                        t1.remove(K)
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break

    cost += INS_DEL * (len(t1) + len(t2))
    return cost

# ========================================
# FIND TRIOS A = B ∪ C
# ========================================

def find_trios(intervals):
    trios = []
    for B, C in combinations(intervals, 2):
        for A in intervals:
            if A != B and A != C and is_union(A, B, C):
                trios.append((A, B, C))
    return trios

trios = find_trios(intervals)

# track intervals appearing in any trio
intervals_in_trios = set()
for A,B,C in trios:
    intervals_in_trios.update([A,B,C])

# ========================================
# SANkOFF FOR A SINGLE TRIO (CORRECT SEMANTICS)
# - For trio (P,B,C) allow bits for P,B,C to vary
# - All other intervals are taken from current_assign (temporary frozen)
# - Permanently frozen intervals in `frozen` are enforced
# - Leaves are immutable (must match observed triple)
# ========================================

def sankoff_trio_enumerate(tree, root, leaves, trio, frozen, current_assign):
    P, B, C = trio
    states = list(product([0,1], repeat=3))
    nodes = postorder

    # observed triple at leaves
    leaf_triple = {
        u: (1 if P in leaves[u] else 0,
            1 if B in leaves[u] else 0,
            1 if C in leaves[u] else 0)
        for u in leaves
    }

    def state_ok(u, s):
        # enforce permanent frozen for any of P,B,C that are permanently frozen at u
        fu = frozen.get(u, {})
        for i, I in enumerate((P, B, C)):
            if I in fu and fu[I] != s[i]:
                return False
        # do NOT enforce current_assign for P,B,C here: they are variables in this DP
        # leaves are handled by DP initialization (their DP entries will be inf except the observed triple)
        return True

    def node_set(u, s):
        # build full concrete set at node u for cost calculation
        if not tree[u]:
            return set(leaves[u])  # leaves immutable
        # start from current_assign (which is temporary freeze for everything except trio)
        base = set(current_assign.get(u, set()))
        # remove trio intervals (they'll be set according to s)
        for I in (P, B, C):
            base.discard(I)
        # add trio bits from s
        if s[0] == 1:
            base.add(P)
        if s[1] == 1:
            base.add(B)
        if s[2] == 1:
            base.add(C)
        # ensure permanently frozen 1s are present (defensive - they should already be in base)
        for I, val in frozen.get(u, {}).items():
            if val == 1:
                base.add(I)
            else:
                base.discard(I)
        return base

    DP = {u: {} for u in nodes}
    back = {}  # back[(u,s,v)] = best child state for v

    # bottom-up DP
    for u in nodes:
        if not tree[u]:
            # leaf: only the observed triple is allowed
            obs = leaf_triple[u]
            for s in states:
                DP[u][s] = 0.0 if s == obs else math.inf
        else:
            for s in states:
                if not state_ok(u, s):
                    DP[u][s] = math.inf
                    continue
                parent_set = node_set(u, s)
                total = 0.0
                valid = True
                for v in tree[u]:
                    best = math.inf
                    best_t = None
                    for t in states:
                        if not state_ok(v, t):
                            continue
                        child_set = node_set(v, t)
                        val = DP[v].get(t, math.inf) + tad_distance(parent_set, child_set)
                        if val < best:
                            best = val
                            best_t = t
                    if best == math.inf:
                        valid = False
                        break
                    total += best
                    back[(u, s, v)] = best_t
                DP[u][s] = total if valid else math.inf

    # choose best root state
    best_root_state = min(DP[root], key=lambda s: DP[root][s])
    # traceback internal nodes
    assign = {}
    def traceback(u, s):
        if tree[u]:
            assign[u] = s
            for v in tree[u]:
                t = back.get((u, s, v))
                traceback(v, t)
    traceback(root, best_root_state)
    return assign

# ========================================
# total cost helper
# ========================================

def total_cost_from_assign(tree, assign):
    cost = 0.0
    for u in tree:
        for v in tree[u]:
            cost += tad_distance(assign[u], assign[v])
    return cost

# ========================================
# INITIAL GLOBAL ASSIGNMENT and SCORE
# ========================================

current_assign = {}
for u in tree:
    if not tree[u]:  # leaves
        current_assign[u] = set(leaf_assignments[u])
    else:
        current_assign[u] = set()  # internals start empty

current_score = total_cost_from_assign(tree, current_assign)

# permanently frozen parents (node -> {interval:0/1})
frozen = defaultdict(dict)

# ========================================
# MAIN LOOP: process parents in admissible order
# ========================================

trios_list = list(trios)

while trios_list:
    # pick an admissible parent A (A not appearing as B or C in any remaining trio)
    usable_parent = None
    for (A,B,C) in trios_list:
        appears_as_child = any((A == x or A == y) for (z,x,y) in trios_list if (z,x,y) != (A,B,C))
        if not appears_as_child:
            usable_parent = A
            break
    if usable_parent is None:
        usable_parent = trios_list[0][0]

    # gather all trios with this parent
    parent_trios = [t for t in trios_list if t[0] == usable_parent]

    # try each split for parent
    for (P,B,C) in parent_trios:
        assign_map = sankoff_trio_enumerate(tree, root, leaf_assignments, (P,B,C), frozen, current_assign)

        # propose candidate assignment by applying bits for P,B,C from assign_map into a copy
        candidate_assign = copy.deepcopy(current_assign)
        for u, s in assign_map.items():
            p_bit, b_bit, c_bit = s
            if p_bit:
                candidate_assign[u].add(P)
            else:
                candidate_assign[u].discard(P)
            if b_bit:
                candidate_assign[u].add(B)
            else:
                candidate_assign[u].discard(B)
            if c_bit:
                candidate_assign[u].add(C)
            else:
                candidate_assign[u].discard(C)

        candidate_score = total_cost_from_assign(tree, candidate_assign)

        if candidate_score <= current_score:
            # accept candidate (update working assignment and score)
            current_assign = candidate_assign
            current_score = candidate_score
            # do NOT freeze P yet; freeze after finishing all parent_trios

    # after exhausting splits for usable_parent, permanently freeze it
    P = usable_parent
    for u in tree:
        if tree[u]:
            frozen[u][P] = 1 if (P in current_assign[u]) else 0

    # remove trios with this parent
    trios_list = [t for t in trios_list if t[0] != P]

# ========================================
# BINARY SANkOFF for intervals that never appeared in any trio
# Respect frozen + current_assign
# ========================================

def binary_sankoff_with_tad(tree, root, leaves, interval, frozen, current_assign):
    nodes = postorder
    DP = {u: {0: math.inf, 1: math.inf} for u in nodes}

    def node_set_with_bit(u, bit):
        if not tree[u]:
            return set(leaves[u])
        S = set(I for I,val in frozen.get(u, {}).items() if val == 1)
        S |= set(current_assign.get(u, set()))
        if bit == 1:
            S.add(interval)
        else:
            S.discard(interval)
        return S

    for u in nodes:
        if not tree[u]:
            if interval in leaves[u]:
                DP[u][1] = 0.0
                DP[u][0] = math.inf
            else:
                DP[u][0] = 0.0
                DP[u][1] = math.inf
        else:
            frozen_val = frozen.get(u, {}).get(interval, None)
            for s in (0,1):
                if frozen_val is not None and s != frozen_val:
                    DP[u][s] = math.inf
                    continue
                total = 0.0
                valid = True
                parent_set = node_set_with_bit(u, s)
                for v in tree[u]:
                    best = math.inf
                    for t in (0,1):
                        child_frozen_val = frozen.get(v, {}).get(interval, None)
                        if child_frozen_val is not None and t != child_frozen_val:
                            continue
                        child_set = node_set_with_bit(v, t)
                        val = DP[v][t] + tad_distance(parent_set, child_set)
                        if val < best:
                            best = val
                    if best == math.inf:
                        valid = False
                        break
                    total += best
                DP[u][s] = total if valid else math.inf
    return DP

no_trio_intervals = [I for I in intervals if I not in intervals_in_trios]

for I in no_trio_intervals:
    DP = binary_sankoff_with_tad(tree, root, leaf_assignments, I, frozen, current_assign)
    best_root_state = 0 if DP[root][0] <= DP[root][1] else 1

    def traceback_bin(u, s):
        if tree[u]:
            frozen[u][I] = s
            if s == 1:
                current_assign[u].add(I)
            else:
                current_assign[u].discard(I)
        for v in tree[u]:
            candidates = []
            for t in (0,1):
                child_frozen_val = frozen.get(v, {}).get(I, None)
                if child_frozen_val is not None and t != child_frozen_val:
                    continue
                parent_set = set(leaf_assignments[u]) if not tree[u] else set(I2 for I2,val in frozen.get(u, {}).items() if val==1) | set(current_assign.get(u, set())) | ({I} if s==1 else set())
                child_set  = set(leaf_assignments[v]) if not tree[v] else set(I2 for I2,val in frozen.get(v, {}).items() if val==1) | set(current_assign.get(v, set())) | ({I} if t==1 else set())
                val = DP[v][t] + tad_distance(parent_set, child_set)
                candidates.append((val, t))
            if not candidates:
                continue
            _, best_t = min(candidates, key=lambda x: x[0])
            traceback_bin(v, best_t)

    traceback_bin(root, best_root_state)

# ========================================
# FINAL RESULT
# ========================================

final_assign = {}
for u in sorted(tree.keys()):
    if not tree[u]:
        final_assign[u] = set(leaf_assignments[u])
    else:
        final_assign[u] = set(current_assign[u])

def total_cost(tree, assign):
    return sum(tad_distance(assign[u], assign[v]) for u in tree for v in tree[u])

print("\n=== HEURISTIC (final) ===")
print("Parsimony cost (heuristic):", total_cost(tree, final_assign))
print("Assignments per node (heuristic):")
for u in sorted(final_assign):
    print(f"{u}: {sorted(final_assign[u])}")

# Optional: compare to exact Sankoff (exponential) for small examples
def sankoff_full(tree, leaves, active_intervals=None):
    if active_intervals is None:
        active_intervals = intervals

    all_states = list(product([0,1], repeat=len(intervals)))
    active_states = list(product([0,1], repeat=len(active_intervals)))

    DP = {}
    choice = {}

    def vec_to_set_all(vec):
        return {intervals[i] for i,b in enumerate(vec) if b}

    def vec_to_set_active(vec):
        return {active_intervals[i] for i,b in enumerate(vec) if b}

    def postorder(u):
        for v in tree[u]:
            postorder(v)

        if not tree[u]:
            DP[u] = {}
            leaf_set = leaves[u]
            for s in all_states:
                s_set = vec_to_set_all(s)
                DP[u][s] = 0 if s_set == leaf_set else math.inf
        else:
            DP[u] = {}
            choice[u] = {}
            for s in active_states:
                total = 0.0
                back = {}
                s_set = vec_to_set_active(s)
                for v in tree[u]:
                    best = math.inf
                    best_t = None
                    for t in (all_states if not tree[v] else active_states):
                        if not tree[v]:
                            t_set = vec_to_set_all(t)
                        else:
                            t_set = vec_to_set_active(t)
                        val = DP[v][t] + tad_distance(s_set, t_set)
                        if val < best:
                            best = val
                            best_t = t
                    total += best
                    back[v] = best_t
                DP[u][s] = total
                choice[u][s] = back

    postorder(root)
    root_state = min(DP[root], key=DP[root].get)
    total_cost_val = DP[root][root_state]

    assign = {root: root_state}
    def backtrack(u):
        if u not in choice:
            return
        for v in tree[u]:
            assign[v] = choice[u][assign[u]][v]
            backtrack(v)
    backtrack(root)

    assign_intervals = {}
    for u in assign:
        if not tree[u]:
            assign_intervals[u] = vec_to_set_all(assign[u])
        else:
            assign_intervals[u] = vec_to_set_active(assign[u])
    return assign_intervals, total_cost_val

final_assign_opt, opt_score = sankoff_full(tree, leaf_assignments)
print("\n=== GLOBAL OPTIMUM (exact) ===")
print("Parsimony cost (exact):", opt_score)
print("Assignments per node (exact):")
for u in sorted(final_assign_opt):
    print(f"{u}: {sorted(final_assign_opt[u])}")
