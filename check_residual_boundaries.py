"""
Exact median (star-topology parent) with RESIDUAL BOUNDARIES allowed.

The DP in `nussinov.py` derives the parent's boundary sets from its interval set:

    S_parent = {a : (a,b) in I_parent}      E_parent = {b : (a,b) in I_parent}

so the parent can only own a boundary if it also owns a TAD that uses it.  This
script drops that restriction: the parent is a pair

    ( I_parent ,  S_parent >= starts(I_parent) ,  E_parent >= ends(I_parent) )

i.e. it may carry *residual* boundaries that no parent TAD is anchored to.
Everything else (the objective) is unchanged:

    cost = sum_c [ Ct*|I_p Δ I_c| + Cb*|S_p Δ S_c| + Cb*|E_p Δ E_c| ]

Both models are brute-forced exactly over every laminar subset of a candidate
interval universe, so the two numbers are directly comparable.

Key structural fact used below: once residual boundaries are free, the boundary
choice at every position decouples from the interval choice.  For a position q
not forced by some parent TAD, owning it costs Cb*(#leaves lacking q) and not
owning it costs Cb*(#leaves having q) -- pick the smaller, independently per
position.

Nothing in the repo is imported or modified except `nussinov.infer_tads_with_dp`,
which is only used for the side-by-side report.
"""

import argparse
import re
from itertools import combinations

from nussinov import infer_tads_with_dp


# ---------------------------------------------------------------- parsing


def parse_interval(s):
    a, b = map(int, re.findall(r"\d+", s))
    return (a, b)


def laminar_pair(iv1, iv2):
    """Nested or disjoint (touching at endpoints allowed), matching tad_utilities."""
    a, b = iv1
    c, d = iv2
    if (a < c < b < d) or (c < a < d < b):
        return False
    return True


def is_laminar(intervals):
    ivs = list(intervals)
    for i in range(len(ivs)):
        for j in range(i + 1, len(ivs)):
            if not laminar_pair(ivs[i], ivs[j]):
                return False
    return True


# ---------------------------------------------------------------- costing


class MedianInstance:
    def __init__(self, leaf_intervals, Ct=1.0, Cb=1.0):
        self.Ct = Ct
        self.Cb = Cb
        self.leaves = sorted(leaf_intervals)
        self.leaf_sets = {
            c: set(parse_interval(x) for x in leaf_intervals[c]) for c in self.leaves
        }
        self.N = len(self.leaves)

        # how many leaves own each interval / start boundary / end boundary
        self.iv_count = {}
        self.start_count = {}
        self.end_count = {}
        for c in self.leaves:
            for (a, b) in self.leaf_sets[c]:
                self.iv_count[(a, b)] = self.iv_count.get((a, b), 0) + 1
            for a in {a for (a, b) in self.leaf_sets[c]}:
                self.start_count[a] = self.start_count.get(a, 0) + 1
            for b in {b for (a, b) in self.leaf_sets[c]}:
                self.end_count[b] = self.end_count.get(b, 0) + 1

    # ---- interval term -------------------------------------------------

    def interval_cost(self, I_p):
        """Ct * sum_c |I_p Δ I_c|, computed from ownership counts."""
        total = 0.0
        for iv in I_p:
            total += self.Ct * (self.N - self.iv_count.get(iv, 0))
        for iv, k in self.iv_count.items():
            if iv not in I_p:
                total += self.Ct * k
        return total

    # ---- boundary term -------------------------------------------------

    def _boundary_cost(self, owned, counts, allow_residual, extra_positions):
        """
        `owned` = positions the parent must own (forced by its TADs).
        Returns (cost, residual_positions_chosen).
        """
        cost = 0.0
        residual = set()
        for q in owned:
            cost += self.Cb * (self.N - counts.get(q, 0))
        for q in extra_positions:
            if q in owned:
                continue
            have = counts.get(q, 0)
            if allow_residual and self.Cb * (self.N - have) < self.Cb * have:
                cost += self.Cb * (self.N - have)
                residual.add(q)
            else:
                cost += self.Cb * have
        return cost, residual

    def evaluate(self, I_p, allow_residual):
        """Total cost of parent interval set I_p under the chosen model."""
        forced_starts = {a for (a, b) in I_p}
        forced_ends = {b for (a, b) in I_p}
        cost = self.interval_cost(I_p)
        s_cost, s_res = self._boundary_cost(
            forced_starts, self.start_count, allow_residual, self.start_count.keys()
        )
        e_cost, e_res = self._boundary_cost(
            forced_ends, self.end_count, allow_residual, self.end_count.keys()
        )
        return cost + s_cost + e_cost, s_res, e_res

    def breakdown(self, I_p, S_p, E_p):
        """Per-leaf cost of an explicit (I_p, S_p, E_p) -- independent re-check."""
        rows = []
        total = 0.0
        for c in self.leaves:
            I_c = self.leaf_sets[c]
            S_c = {a for (a, b) in I_c}
            E_c = {b for (a, b) in I_c}
            ti = self.Ct * len(I_p ^ I_c)
            ts = self.Cb * len(S_p ^ S_c)
            te = self.Cb * len(E_p ^ E_c)
            rows.append((c, ti, ts, te, ti + ts + te))
            total += ti + ts + te
        return rows, total


# ---------------------------------------------------------------- search


def enumerate_laminar_subsets(candidates):
    """DFS over include/exclude with laminarity pruning."""
    cands = sorted(candidates)
    n = len(cands)
    out = []

    def rec(i, chosen):
        if i == n:
            out.append(frozenset(chosen))
            return
        rec(i + 1, chosen)  # exclude
        iv = cands[i]
        if all(laminar_pair(iv, other) for other in chosen):
            chosen.append(iv)
            rec(i + 1, chosen)
            chosen.pop()

    rec(0, [])
    return out


def majority_solution(inst):
    """
    Closed form for the residual-allowed median.

    Once the parent may own boundaries without owning TADs, the objective is a
    sum of independent per-feature terms, so every feature is a majority vote:
    own interval iv iff more leaves have it than lack it, and likewise for each
    start / end position.  The resulting interval set is automatically laminar --
    two strict-majority intervals must co-occur in at least one leaf, and that
    leaf's call set is laminar -- so the laminarity constraint never binds.
    """
    N = inst.N
    I_p = frozenset(iv for iv, k in inst.iv_count.items() if 2 * k > N)
    S_res = {q for q, k in inst.start_count.items() if 2 * k > N} - {a for (a, b) in I_p}
    E_res = {q for q, k in inst.end_count.items() if 2 * k > N} - {b for (a, b) in I_p}
    score, _, _ = inst.evaluate(I_p, True)
    return score, I_p, S_res, E_res


def build_universe(inst, start_boundaries, end_boundaries, mode):
    """
    mode='leaf'  : only intervals some leaf actually calls
    mode='full'  : every (a,b), a in start_boundaries, b in end_boundaries, a<b,
                   restricted to boundaries at least one leaf uses on that side
                   (a parent boundary no leaf owns is pure loss on every edge)
    """
    if mode == "leaf":
        return sorted(inst.iv_count)
    starts = [a for a in sorted(set(start_boundaries)) if a in inst.start_count]
    ends = [b for b in sorted(set(end_boundaries)) if b in inst.end_count]
    return sorted((a, b) for a in starts for b in ends if a < b)


def solve(inst, universe, allow_residual, subsets=None, require_nonempty=False):
    best = None
    for subset in subsets if subsets is not None else enumerate_laminar_subsets(universe):
        if require_nonempty and not subset:
            continue
        score, s_res, e_res = inst.evaluate(subset, allow_residual)
        if best is None or score < best[0] - 1e-9:
            best = (score, subset, s_res, e_res)
    return best


def rank(inst, subsets, allow_residual, k=10):
    scored = []
    for subset in subsets:
        score, s_res, e_res = inst.evaluate(subset, allow_residual)
        scored.append((score, subset, s_res, e_res))
    scored.sort(key=lambda t: (t[0], len(t[1]), sorted(t[1])))
    return scored[:k]


# ---------------------------------------------------------------- report


def report(title, inst, best):
    score, I_p, s_res, e_res = best
    S_p = {a for (a, b) in I_p} | s_res
    E_p = {b for (a, b) in I_p} | e_res
    rows, recheck = inst.breakdown(set(I_p), S_p, E_p)

    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    print(f"  parent TADs        : {sorted(I_p)}")
    print(f"  parent starts      : {sorted(S_p)}   (residual: {sorted(s_res)})")
    print(f"  parent ends        : {sorted(E_p)}   (residual: {sorted(e_res)})")
    print(f"  cost               : {score:.6f}")
    print(f"  cost (re-checked)  : {recheck:.6f}")
    print("  per-leaf   Ct*|IΔI|   Cb*|SΔS|   Cb*|EΔE|   total")
    for c, ti, ts, te, tot in rows:
        print(f"    leaf {c:<4}   {ti:>7.1f}   {ts:>8.1f}   {te:>8.1f}   {tot:>6.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--Ct", type=float, default=1.0)
    ap.add_argument("--Cb", type=float, default=1.0)
    ap.add_argument(
        "--universe",
        choices=["full", "leaf"],
        default="full",
        help="candidate interval universe for the brute force",
    )
    ap.add_argument("--top", type=int, default=10, help="how many near-optima to list")
    args = ap.parse_args()

    # ---- the example from nussinov.py __main__ -------------------------
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    leaf_intervals = {
        1: ["[1,50]", "[60,100]", "[2,5]"],
        2: ["[1,52]", "[53,100]", "[2,6]"],
        3: ["[1,54]", "[55,100]", "[2,7]"],
        4: ["[1,56]", "[57,100]", "[2,8]"],
    }
    start_boundaries = [1, 2, 5, 6, 7, 8, 50, 52, 53, 54, 55, 56, 57, 60, 100]
    end_boundaries = list(start_boundaries)

    inst = MedianInstance(leaf_intervals, Ct=args.Ct, Cb=args.Cb)

    print("Leaf TAD sets:")
    for c in inst.leaves:
        print(f"  leaf {c}: {sorted(inst.leaf_sets[c])}")
    print(f"\nCt={args.Ct}  Cb={args.Cb}")

    universe = build_universe(inst, start_boundaries, end_boundaries, args.universe)
    print(f"Candidate universe ({args.universe}): {len(universe)} intervals")
    subsets = enumerate_laminar_subsets(universe)
    print(f"Laminar subsets enumerated: {len(subsets)}")

    # ---- DP (no residual boundaries) ----------------------------------
    dp_intervals, dp_score = infer_tads_with_dp(
        edges, leaf_intervals, start_boundaries, end_boundaries,
        Ct=args.Ct, Cb=args.Cb, trace=False,
    )
    print(f"\nDP (nussinov.py)      : {sorted(dp_intervals)}  cost={dp_score:.6f}")

    # ---- brute force, both models -------------------------------------
    best_no_res = solve(inst, universe, False, subsets=subsets)
    best_res = solve(inst, universe, True, subsets=subsets)
    best_res_nonempty = solve(inst, universe, True, subsets=subsets, require_nonempty=True)

    report("EXACT OPTIMUM -- residual boundaries FORBIDDEN (DP's model)", inst, best_no_res)
    report("EXACT OPTIMUM -- residual boundaries ALLOWED", inst, best_res)
    report("BEST NON-EMPTY parent -- residual boundaries ALLOWED", inst, best_res_nonempty)

    # ---- landscape ----------------------------------------------------
    print(f"\n{'=' * 72}\nTop parent TAD sets, residual boundaries ALLOWED\n{'=' * 72}")
    for score, I_p, s_res, e_res in rank(inst, subsets, True, k=args.top):
        S_p = sorted({a for (a, b) in I_p} | s_res)
        E_p = sorted({b for (a, b) in I_p} | e_res)
        print(f"  {score:6.1f}   TADs={sorted(I_p)}   S={S_p} E={E_p}")

    # the interval the DP picks, re-scored under the residual model
    dp_pick = frozenset(dp_intervals)
    dp_under_res, s_res, e_res = inst.evaluate(dp_pick, True)
    print(f"\n  DP's set {sorted(dp_pick)} re-scored with residuals: {dp_under_res:.1f}"
          f"   (residual starts {sorted(s_res)}, residual ends {sorted(e_res)})")

    # ---- closed form vs brute force -----------------------------------
    maj = majority_solution(inst)
    report("CLOSED FORM (majority vote per interval / per boundary)", inst, maj)
    agree = abs(maj[0] - best_res[0]) < 1e-9
    print(f"  matches brute-force residual optimum: {agree}")
    if not agree:
        raise SystemExit("closed form disagrees with brute force -- investigate")

    print(f"\n{'=' * 72}")
    print(f"  DP cost                        : {dp_score:.6f}")
    print(f"  exact, no residual allowed     : {best_no_res[0]:.6f}")
    print(f"  exact, residual allowed        : {best_res[0]:.6f}")
    print(f"  exact, residual + non-empty    : {best_res_nonempty[0]:.6f}")
    print(f"  improvement from residuals     : {best_no_res[0] - best_res[0]:.6f}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
