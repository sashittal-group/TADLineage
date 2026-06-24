from itertools import product, combinations, chain
from collections import defaultdict
import math
import random


# ========================================
# USER INPUT: TREE STRUCTURE AND LEAF ASSIGNMENTS
# ========================================

# Define tree as a list of edges (parent, child)
edges = [
    (0, 1),
    (0, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 6),
    (2, 7),
]

# Leaf assignments: node -> set of intervals
# Available intervals pool
available_intervals = ["[1,10]", "[1,5]", "[6,10]", "[1,3]", "[4,5]"]

# Identify leaves (nodes that appear as children but never as parents)
parent_nodes = set(parent for parent, _ in edges)
child_nodes = set(child for _, child in edges)
leaf_nodes = child_nodes - parent_nodes

# Randomly assign intervals to each leaf
# random.seed(42)  # Set seed for reproducibility; remove this line for true randomness
leaf_assignments = {
    3: {"[1,5]", "[6,10]"},
    4: {"[1,3]", "[4,5]", "[6,10]"},
    5: {"[1,3]", "[1,5]"},
    6: {"[1,3]", "[4,5]", "[6,10]"},
    7: {"[1,10]"},
    # leaf: set(random.sample(available_intervals, k=random.randint(1, len(available_intervals))))
    # for leaf in leaf_nodes
}

# ========================================
# BUILD TREE STRUCTURE FROM EDGES
# ========================================

def build_tree_from_edges(edges):
    """
    Build tree adjacency dict from edge list.
    Returns (tree, root) where tree maps parent -> [children]
    """
    tree = defaultdict(list)
    all_nodes = set()
    children_nodes = set()
    
    for parent, child in edges:
        tree[parent].append(child)
        all_nodes.add(parent)
        all_nodes.add(child)
        children_nodes.add(child)
    
    # Root is the node that appears as parent but never as child
    roots = all_nodes - children_nodes
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, found {len(roots)}: {roots}")
    
    root = roots.pop()
    
    # Ensure all nodes have an entry (even leaves with no children)
    for node in all_nodes:
        if node not in tree:
            tree[node] = []
    
    return dict(tree), root

tree, root = build_tree_from_edges(edges)
leaves = leaf_assignments

# ----- Build intervals from leaves -----
intervals = sorted({I for S in leaves.values() for I in S})

# Optional: index map if needed elsewhere
idx = {I: i for i, I in enumerate(intervals)}

# leaves_bits = {
#     u: tuple(random.randint(0,1) for _ in range(5))
#     for u in ["c","d","e","f","g"]
# }

# Convert bits to interval sets
# leaves = {
#     u: {intervals[i] for i,b in enumerate(bits) if b}
#     for u,bits in leaves_bits.items()
# }

# ----- Cost parameters -----
INS_DEL = 1.0
MERGE_SPLIT = 2.1

# ----- Helper: parse interval -----
def parse_interval(s):
    a,b = s.strip("[]").split(",")
    return int(a), int(b)

def is_union(parent, c1, c2):
    a1,b1 = parse_interval(c1)
    a2,b2 = parse_interval(c2)
    ap,bp = parse_interval(parent)
    return min(a1,a2)==ap and max(b1,b2)==bp and (b1+1==a2 or b2+1==a1)

# ----- TAD distance -----
def tad_distance(t1, t2, INS_DEL=1.0, MERGE_SPLIT=2.1):
    t1 = set(t1)
    t2 = set(t2)
    cost = 0.0

    # Remove common intervals
    common = t1 & t2
    t1 -= common
    t2 -= common

    # Split: t1 -> t2
    changed = True
    while changed:
        changed = False
        for I in list(t1):
            for J in t2:
                for K in t2:
                    if J!=K and is_union(I,J,K):
                        cost += MERGE_SPLIT
                        t1.remove(I)
                        t2.remove(J)
                        t2.remove(K)
                        changed = True
                        break
                if changed: break
            if changed: break

    # Merge: t2 -> t1
    changed = True
    while changed:
        changed = False
        for I in list(t2):
            for J in t1:
                for K in t1:
                    if J!=K and is_union(I,J,K):
                        cost += MERGE_SPLIT
                        t2.remove(I)
                        t1.remove(J)
                        t1.remove(K)
                        changed = True
                        break
                if changed: break
            if changed: break

    # Remaining insertions/deletions
    cost += INS_DEL * (len(t1)+len(t2))
    return cost

# ----- Generalized Sankoff -----
def sankoff(tree, leaves, active_intervals=None):
    """
    tree: adjacency dict
    leaves: dict {leaf: set of intervals}
    active_intervals: list of intervals to consider for internal nodes (trio)
    """
    if active_intervals is None:
        active_intervals = intervals

    # Generate all possible states for ALL intervals (for leaves)
    all_states = list(product([0,1], repeat=len(intervals)))
    
    # Generate states only for active intervals (for internal nodes)
    active_states = list(product([0,1], repeat=len(active_intervals)))

    DP = {}
    choice = {}

    # Convert full vector to interval set using all intervals
    def vec_to_set_all(vec):
        return {intervals[i] for i,b in enumerate(vec) if b}
    
    # Convert active vector to interval set using only active intervals
    def vec_to_set_active(vec):
        return {active_intervals[i] for i,b in enumerate(vec) if b}

    # Postorder DP
    def postorder(u):
        for v in tree[u]:
            postorder(v)

        if not tree[u]:  # leaf
            DP[u] = {}
            leaf_set = leaves[u]  # Use all assigned intervals
            for s in all_states:
                s_set = vec_to_set_all(s)
                DP[u][s] = 0 if s_set == leaf_set else math.inf
        else:
            DP[u] = {}
            choice[u] = {}
            # For internal nodes, only consider active intervals
            for s in active_states:
                total = 0
                back = {}
                s_set = vec_to_set_active(s)
                for v in tree[u]:
                    best = math.inf
                    best_t = None
                    # Children can be leaves (all_states) or internal (active_states)
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
    total_cost = DP[root][root_state]

    # Backtrack assignments
    assign = {root: root_state}
    def backtrack(u):
        if u not in choice:
            return
        for v in tree[u]:
            assign[v] = choice[u][assign[u]][v]
            backtrack(v)
    backtrack(root)

    # Convert assignment vectors to interval sets
    assign_intervals = {}
    for u in assign:
        if not tree[u]:  # leaf
            assign_intervals[u] = vec_to_set_all(assign[u])
        else:  # internal
            assign_intervals[u] = vec_to_set_active(assign[u])
    
    return assign_intervals, total_cost






def total_cost(tree, assign):
    cost = 0.0
    for u in tree:
        for v in tree[u]:
            cost += tad_distance(assign[u], assign[v])
    return cost


# ===============================
# GLOBAL OPTIMUM
# ===============================

final_assign_opt, opt_score = sankoff(tree, leaves)

# ===============================
# OUTPUT
# ===============================

# Get all nodes in tree for display
all_nodes = sorted(tree.keys())

print("\n=== GLOBAL OPTIMUM ===")
print("Parsimony cost:", opt_score)
print("Assignments per node:")
for u in all_nodes:
    print(f"{u}: {sorted(final_assign_opt[u])}")
