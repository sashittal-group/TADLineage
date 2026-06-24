from itertools import product, combinations, chain
from collections import defaultdict
import math
import random


# ----- Tree structure -----
tree = {
    "r": ["a", "b"],
    "a": ["c", "d", "e"],
    "b": ["f", "g"],
    "c": [], "d": [], "e": [], "f": [], "g": []
}



# assign unique intervals from above  
intervals = ["[1,10]", "[1,5]", "[6,10]", "[1,3]","[4,5]"]
idx = {intervals[i]: i for i in range(len(intervals))}



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
    active_intervals: list of intervals to consider (subset or None for all)
    """
    if active_intervals is None:
        active_intervals = intervals

    # Generate all possible states for the active intervals
    states = list(product([0,1], repeat=len(active_intervals)))

    DP = {}
    choice = {}

    # Convert vector to interval set
    def vec_to_set(vec):
        return {active_intervals[i] for i,b in enumerate(vec) if b}

    # Postorder DP
    def postorder(u):
        for v in tree[u]:
            postorder(v)

        if not tree[u]:  # leaf
            DP[u] = {}
            leaf_state = tuple(1 if I in leaves[u] else 0 for I in active_intervals)
            for s in states:
                DP[u][s] = 0 if s==leaf_state else math.inf
        else:
            DP[u] = {}
            choice[u] = {}
            for s in states:
                total = 0
                back = {}
                s_set = vec_to_set(s)
                for v in tree[u]:
                    best = math.inf
                    best_t = None
                    for t in states:
                        t_set = vec_to_set(t)
                        val = DP[v][t] + tad_distance(s_set, t_set)
                        if val < best:
                            best = val
                            best_t = t
                    total += best
                    back[v] = best_t
                DP[u][s] = total
                choice[u][s] = back

    postorder("r")

    root_state = min(DP["r"], key=DP["r"].get)
    total_cost = DP["r"][root_state]

    # Backtrack assignments
    assign = {"r": root_state}
    def backtrack(u):
        if u not in choice:
            return
        for v in tree[u]:
            assign[v] = choice[u][assign[u]][v]
            backtrack(v)
    backtrack("r")

    # Convert assignment vectors to interval sets
    assign_intervals = {u: vec_to_set(assign[u]) for u in assign}
    return assign_intervals, total_cost
def total_cost(tree, assign):
    cost = 0.0
    for u in tree:
        for v in tree[u]:
            cost += tad_distance(assign[u], assign[v])
    return cost

# ===============================
# EXHAUSTIVE SEARCH OVER LEAVES
# ===============================

leaf_names = ["c","d","e","f","g"]
all_leaf_states = list(product([0,1], repeat=len(intervals)))

trios = []
for I in intervals:
    for J,K in combinations(intervals, 2):
        if is_union(I, J, K):
            trios.append(tuple(sorted([I,J,K])))
trios = list(set(trios))

def bits_to_set(bits):
    return {intervals[i] for i,b in enumerate(bits) if b}

found = False
case_count = 0

for leaf_bits_combo in product(all_leaf_states, repeat=len(leaf_names)):
    case_count += 1

    # Build leaves dict
    leaves = {
        leaf_names[i]: bits_to_set(leaf_bits_combo[i])
        for i in range(len(leaf_names))
    }

    # -------------------------------
    # Run TRIO-BASED HEURISTIC
    # -------------------------------
    votes = defaultdict(lambda: defaultdict(int))

    for trio in trios:
        assign, _ = sankoff(tree, leaves)
        for u, interval_set in assign.items():
            for I in interval_set:
                if I in trio:
                    votes[u][I] += 1

    trio_intervals = set(chain(*trios))
    solo_intervals = [I for I in intervals if I not in trio_intervals]

    for I in solo_intervals:
        assign, _ = sankoff(tree, leaves)
        for u, interval_set in assign.items():
            if I in interval_set:
                votes[u][I] += 1

    final_assign_heuristic = {
        u: {I for I in intervals if votes[u][I] > 0}
        for u in tree
    }

    heuristic_score = total_cost(tree, final_assign_heuristic)

    # -------------------------------
    # Run GLOBAL OPTIMUM
    # -------------------------------
    final_assign_opt, opt_score = sankoff(tree, leaves)

    # -------------------------------
    # Compare
    # -------------------------------
    if abs(heuristic_score - opt_score) > 1e-9:
        print("\n🚨 COUNTEREXAMPLE FOUND 🚨")
        print(f"Checked cases: {case_count}")

        print("\nLeaf assignments:")
        for u in leaf_names:
            print(f"{u}: {sorted(leaves[u])}")

        print("\n=== TRIO-BASED HEURISTIC ===")
        print("Parsimony cost:", heuristic_score)
        for u in ["r","a","b","c","d","e","f","g"]:
            print(f"{u}: {sorted(final_assign_heuristic[u])}")

        print("\n=== GLOBAL OPTIMUM ===")
        print("Parsimony cost:", opt_score)
        for u in ["r","a","b","c","d","e","f","g"]:
            print(f"{u}: {sorted(final_assign_opt[u])}")

        found = True
        break

if not found:
    print("\n✅ No counterexample found after checking", case_count, "cases.")