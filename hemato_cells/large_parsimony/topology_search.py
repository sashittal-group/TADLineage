#!/usr/bin/env python3
"""
topology_search.py

Exact large parsimony by branch and bound over tree topologies, for when the
single large ILP (large_parsimony_ilp.py) is too slow to prove optimality. It
returns the optimal cost and EVERY unrooted binary tree that attains it.

Fitch lower bound. Treat each observed TAD, each observed start and each observed
end as an independent binary character and take its Fitch parsimony length on the
tree:
    LB(tree) = Ct * sum_TADs Fitch + Cb * sum_starts,ends Fitch.
This is the TAD-LP objective with laminarity and the TAD/boundary coupling
dropped, and with candidate TADs that no leaf has left at 0 (which costs
nothing), so LB <= the exact small parsimony cost of the tree. Fitch length
never decreases when a leaf is added to a tree, so the bound of a partial tree
bounds every tree completed from it.

Search. UB starts at the exact cost of a greedy stepwise-addition tree. Leaves
are then added one at a time, each on every edge of the current tree, which
generates every unrooted binary topology exactly once; a partial tree is pruned
when its Fitch bound exceeds UB (ties are kept, so every co-optimal tree
survives). The surviving complete trees are scored in increasing Fitch bound
until the bound exceeds UB, which only falls along the way. A tree is stored as
the sequence of edges its leaves were inserted on, so millions of tied trees fit
in memory.

Scoring one tree, cheapest step first. A step either proves the tree's exact cost
or rules it out (a lower bound above UB); the small parsimony ILP runs only when
none does.
  1. Fitch labeling: the most parsimonious Fitch assignment of the TAD characters
     (top-down from leaf 0, parent's state preferred), boundaries derived from the
     TADs at each vertex. If it is laminar it is a feasible solution, so its cost H
     is an upper bound on the tree's exact cost; H equal to the Fitch bound proves
     it exact.
  2. Grouped bound: each TAD's cost is split half to its start group and half to
     its end group; a group (the observed TADs sharing that start / end, a phantom
     bit standing for any unobserved TAD there, and the boundary itself, present
     iff the group is non-empty) is solved exactly by Sankoff on the tree. Still a
     relaxation of the objective, but it keeps the TAD/boundary coupling the Fitch
     bound drops. Ruled out if above UB; exact if it reaches H.
  3. LP relaxation without laminarity, on one relaxed all-arcs model built once,
     with the tree imposed through the arc bounds. Ruled out if above UB; exact if
     it reaches H, or if the LP solution is integral and laminar.
  4. The fixed-tree small parsimony ILP (LargeParsimonyILP with the arcs fixed),
     laminarity enforced by re-solving with the crossing pairs of each optimum,
     and a Gurobi cutoff at UB.
All trees are rooted on the pendant edge of leaf 0, which the objective ignores.

Convex objective only.
"""

import math
import time
from collections import defaultdict

import numpy as np

from large_parsimony_ilp import LargeParsimonyILP, crossing_pairs


class TopologySearch:
    def __init__(self, leaf_tads, Ct=1.0, Cb=1.0, threads=None, time_limit=None, verbose=False,
                 tad_weights=None, start_weights=None, end_weights=None, build_groups=True):
        self.leaf_tads = [{tuple(t) for t in tads if t[0] < t[1]} for tads in leaf_tads]
        self.k = len(self.leaf_tads)
        if self.k < 4:
            raise ValueError("Needs at least 4 leaves.")
        self.Ct, self.Cb = Ct, Cb
        self.threads, self.time_limit = threads, time_limit
        self.verbose = verbose
        # weighted TAD distance d_w (missing keys fall back to Ct / Cb, as in the ILP)
        self.tad_weights = tad_weights or {}
        self.start_weights = start_weights or {}
        self.end_weights = end_weights or {}
        self.distance_weighted = bool(tad_weights or start_weights or end_weights)
        self.integral = (float(Ct).is_integer() and float(Cb).is_integer()
                         and not self.distance_weighted)
        self.slack = 0.5 if self.integral else 1e-6     # Gurobi cutoff above ub
        self._lp = None
        self.counts = defaultdict(int)       # which step settled each scored tree
        self.seconds = defaultdict(float)
        self._encode()
        if build_groups:        # genome_tree.GenomeSearch scores trees per chromosome instead
            self._build_groups()

    # -- Fitch lower bound -----------------------------------------------------
    def _encode(self):
        """Per leaf and character group, the bitset of characters it has (state 1)."""
        tads = sorted(set().union(*self.leaf_tads))
        starts = sorted({a for a, _ in tads})
        ends = sorted({b for _, b in tads})
        t_idx = {t: i for i, t in enumerate(tads)}
        s_idx = {a: i for i, a in enumerate(starts)}
        e_idx = {b: len(starts) + i for i, b in enumerate(ends)}
        self.tads = tads
        self.masks = ((1 << len(tads)) - 1, (1 << (len(starts) + len(ends))) - 1)
        # per-character costs: TADs; starts then ends
        self.char_w = (
            [self.tad_weights.get(t, self.Ct) for t in tads],
            [self.start_weights.get(a, self.Cb) for a in starts]
            + [self.end_weights.get(b, self.Cb) for b in ends],
        )
        # Fitch counts changed characters with popcounts. Uniform costs: one plane,
        # exact. Weighted: the costs floored to multiples of 1/1000 and split into
        # binary planes, so the bound stays a (slightly weaker) lower bound.
        self.planes = []
        for g in (0, 1):
            if not self.distance_weighted:
                self.planes.append([(self.masks[g], (self.Ct, self.Cb)[g])])
                continue
            scaled = [int(math.floor(w * 1000 + 1e-9)) for w in self.char_w[g]]
            planes = []
            for j in range(max(scaled, default=0).bit_length()):
                mask = sum(1 << i for i, w in enumerate(scaled) if w >> j & 1)
                if mask:
                    planes.append((mask, (1 << j) / 1000.0))
            self.planes.append(planes)
        self.leaf_states = []
        for leaf in self.leaf_tads:
            one_t = sum(1 << t_idx[t] for t in leaf)
            one_b = sum(1 << s_idx[a] for a in {a for a, _ in leaf}) \
                | sum(1 << e_idx[b] for b in {b for _, b in leaf})
            # per group: (can be 0, can be 1)
            self.leaf_states.append(((self.masks[0] & ~one_t, one_t), (self.masks[1] & ~one_b, one_b)))
        # for the Fitch labeling: boundary bits and crossing TADs of each TAD
        self.tad_bounds = [(1 << s_idx[a]) | (1 << e_idx[b]) for (a, b) in tads]
        self.crossing = [0] * len(tads)
        for t1, t2 in crossing_pairs(tads):
            self.crossing[t_idx[t1]] |= 1 << t_idx[t2]
            self.crossing[t_idx[t2]] |= 1 << t_idx[t1]
        self._bounds_cache = {}

    def _combine(self, sa, sb):
        """Fitch step for both groups: returns (state, weighted cost of the step)."""
        out, cost = [], 0.0
        for g in (0, 1):
            a0, a1 = sa[g]
            b0, b1 = sb[g]
            i0, i1 = a0 & b0, a1 & b1
            inter = i0 | i1
            empty = self.masks[g] & ~inter
            out.append((i0 | ((a0 | b0) & empty), i1 | ((a1 | b1) & empty)))
            for mask, value in self.planes[g]:
                cost += value * (empty & mask).bit_count()
        return tuple(out), cost

    def lower_bound(self, adj):
        """Fitch length of the unrooted tree ``adj`` (node -> neighbours; leaves are
        0..k-1), rooted on the pendant edge of leaf 0."""
        k = self.k

        def rec(node, parent):
            if node < k:
                return self.leaf_states[node], 0.0
            c1, c2 = [c for c in adj[node] if c != parent]
            s1, x1 = rec(c1, node)
            s2, x2 = rec(c2, node)
            s, x = self._combine(s1, s2)
            return s, x + x1 + x2

        (nb,) = adj[0]
        s, x = rec(nb, 0)
        _, x0 = self._combine(s, self.leaf_states[0])
        return x + x0

    # -- step 1: Fitch labeling (an upper bound) ---------------------------------
    def _boundaries(self, tad_bits):
        cached = self._bounds_cache.get(tad_bits)
        if cached is None:
            cached, bits, i = 0, tad_bits, 0
            while bits:
                if bits & 1:
                    cached |= self.tad_bounds[i]
                bits >>= 1
                i += 1
            self._bounds_cache[tad_bits] = cached
        return cached

    def _laminar(self, tad_bits):
        bits, i = tad_bits, 0
        while bits:
            if bits & 1 and self.crossing[i] & tad_bits:
                return False
            bits >>= 1
            i += 1
        return True

    def labeling_cost(self, adj):
        """Cost of the Fitch labeling of the TADs (boundaries derived from them), or
        None if some vertex is not laminar."""
        k, mask = self.k, self.masks[0]
        pre = {}

        def up(node, parent):
            if node < k:
                return self.leaf_states[node][0]
            c1, c2 = [c for c in adj[node] if c != parent]
            a0, a1 = up(c1, node)
            b0, b1 = up(c2, node)
            i0, i1 = a0 & b0, a1 & b1
            empty = mask & ~(i0 | i1)
            pre[node] = (i0 | ((a0 | b0) & empty), i1 | ((a1 | b1) & empty))
            return pre[node]

        (nb,) = adj[0]
        up(nb, 0)
        val = {v: self.leaf_states[v][0][1] for v in range(k)}
        stack = [(nb, 0)]
        while stack:
            node, parent = stack.pop()
            if node < k:
                continue
            p = val[parent]
            can0, can1 = pre[node]
            v = (p & can1) | (~p & ~can0 & mask)
            if not self._laminar(v):
                return None
            val[node] = v
            stack.extend((c, node) for c in adj[node] if c != parent)
        cost = 0.0
        for u in adj:
            for w in adj[u]:
                if u < w:
                    cost += self._charge(0, val[u] ^ val[w])
                    cost += self._charge(1, self._boundaries(val[u]) ^ self._boundaries(val[w]))
        return cost

    def _charge(self, g, bits):
        """Exact cost of the changed characters ``bits`` of group g."""
        if not self.distance_weighted:
            return (self.Ct, self.Cb)[g] * bits.bit_count()
        weights, total = self.char_w[g], 0.0
        while bits:
            low = bits & -bits
            total += weights[low.bit_length() - 1]
            bits ^= low
        return total

    # -- step 2: grouped Sankoff bound --------------------------------------------
    MAX_GROUP_BITS = 14     # a group's Sankoff table has 2^m states per leaf

    def _build_groups(self):
        """Groups of the grouped bound, batched by size m (members + phantom bit).

        A member is a TAD, or ("any", side, pos): "some TAD starts (ends) at pos".
        A group with more than MAX_GROUP_BITS bits (2^25 states at 250 kb, tol 0:
        tens of GB) is split into a boundary group {("any", side, pos)} that
        carries the boundary cost, and chunks of its TADs with no boundary cost.
        The joint cost of a labeling is the sum of these parts' costs on its
        projections, so the parts' separate optima sum to a lower bound: the bound
        stays valid, only weaker for those few groups."""
        groups = defaultdict(list)     # m -> [(members, boundary cost)]
        chunk = self.MAX_GROUP_BITS - 1
        for side, bweights, default in ((0, self.start_weights, self.Cb), (1, self.end_weights, self.Cb)):
            by_pos = defaultdict(set)
            for tads in self.leaf_tads:
                for t in tads:
                    by_pos[t[side]].add(t)
            for pos, members in by_pos.items():
                members, bw = sorted(members), bweights.get(pos, default)
                if len(members) + 1 <= self.MAX_GROUP_BITS:
                    groups[len(members) + 1].append((members, bw))
                    continue
                groups[2].append(([("any", side, pos)], bw))
                for i in range(0, len(members), chunk):
                    part = members[i:i + chunk]
                    groups[len(part) + 1].append((part, 0.0))

        def leaf_has(tads, member):
            if member[0] == "any":
                return any(t[member[1]] == member[2] for t in tads)
            return member in tads

        self.group_batches = []
        for m, gs in sorted(groups.items()):
            n_states = 1 << m
            leafcost = np.full((self.k, len(gs), n_states), np.inf)
            for li, tads in enumerate(self.leaf_tads):
                for gi, (members, _) in enumerate(gs):
                    s = sum(1 << j for j, t in enumerate(members) if leaf_has(tads, t))
                    leafcost[li, gi, s] = 0.0
            # half of each TAD's cost (a boundary member costs nothing itself); the
            # phantom (last bit) stands for unobserved TADs, which all cost Ct
            bit_w = np.array([[0.0 if t[0] == "any" else self.tad_weights.get(t, self.Ct) / 2.0
                               for t in members] + [self.Ct / 2.0]
                              for members, _ in gs])
            bound_w = np.array([bw for _, bw in gs])
            bits = np.array([[s >> j & 1 for j in range(m)] for s in range(n_states)], dtype=float)
            state_w = bit_w @ bits.T                  # (groups, states): cost of all bits of s
            flips = [np.arange(n_states) ^ (1 << j) for j in range(m)]
            self.group_batches.append((leafcost, bit_w, bound_w, state_w, flips))

    @staticmethod
    def _group_step(C, bit_w, bound_w, state_w, flips):
        """Sankoff edge step: min_t c(s, t) + C(t), with
        c(s, t) = sum of the half TAD costs of s xor t + boundary cost * [s empty != t empty]."""
        D = C.copy()
        D[:, 0] = np.inf
        for j, f in enumerate(flips):           # weighted Hamming distance transform
            D = np.minimum(D, D[:, f] + bit_w[:, [j]])
        from_empty = C[:, [0]] + state_w
        out = np.empty_like(C)
        out[:, 0] = np.minimum(from_empty[:, 0], D[:, 0] + bound_w)
        out[:, 1:] = np.minimum(D[:, 1:], from_empty[:, 1:] + bound_w[:, None])
        return out

    def group_bound(self, adj):
        k = self.k
        (nb,) = adj[0]
        total = 0.0
        for leafcost, *step in self.group_batches:
            def rec(node, parent):
                if node < k:
                    return leafcost[node]
                acc = None
                for c in adj[node]:
                    if c != parent:
                        t = self._group_step(rec(c, node), *step)
                        acc = t if acc is None else acc + t
                return acc

            v = self._group_step(rec(nb, 0), *step) + leafcost[0]
            total += float(v.min(axis=1).sum())
        return total

    # -- step 3: LP relaxation -------------------------------------------------------
    def _lp_model(self):
        """The LP relaxation, without laminarity, of the ILP over ALL candidate arcs,
        built once. A tree is imposed by fixing the arc bounds, so each bound is a
        warm-started re-solve rather than a new model (7x faster at 500 kb)."""
        if self._lp is None:
            big = LargeParsimonyILP(
                [sorted(t) for t in self.leaf_tads], Ct=self.Ct, Cb=self.Cb, mode="convex",
                threads=self.threads, laminarity="off", root_leaf=1, **self._weight_kwargs(),
            )
            relaxed = big.m.relax()
            relaxed.setParam("OutputFlag", 0)
            relaxed.setParam("LazyConstraints", 0)
            if self.threads:
                relaxed.setParam("Threads", int(self.threads))
            evars = {a: relaxed.getVarByName(big.e[a].VarName) for a in big.arcs}
            xvars = {v: [relaxed.getVarByName(big.x[v, t].VarName) for t in big.candidates]
                     for v in big.internal}
            self._lp = (relaxed, evars, xvars, big.candidates)
        return self._lp

    def lp_bound(self, edges):
        """(LP value, whether its solution is integral and laminar)."""
        relaxed, evars, xvars, cands = self._lp_model()
        arcs = set(self.rooted_arcs(edges))
        for a, var in evars.items():
            b = 1.0 if a in arcs else 0.0
            var.LB = b
            var.UB = b
        relaxed.optimize()
        if relaxed.Status != 2:     # GRB.OPTIMAL
            raise RuntimeError(f"LP relaxation status {relaxed.Status}")
        exact = True
        for v, xs in xvars.items():
            vals = relaxed.getAttr("X", xs)
            if any(1e-6 < val < 1 - 1e-6 for val in vals):
                exact = False
                break
            if crossing_pairs([t for t, val in zip(cands, vals) if val > 0.5]):
                exact = False
                break
        return relaxed.ObjVal, exact

    def _weight_kwargs(self):
        return {"tad_weights": self.tad_weights, "start_weights": self.start_weights,
                "end_weights": self.end_weights} if self.distance_weighted else {}

    # -- scoring one tree ------------------------------------------------------------
    def _above(self, value, ub):
        """value > ub, knowing costs are whole numbers when Ct and Cb are."""
        if self.integral:
            return math.ceil(value - 1e-6) > ub + 1e-9
        return value > ub + 1e-6

    def _reaches(self, bound, cost):
        """A lower bound that proves ``cost`` (a feasible cost) optimal."""
        if self.integral:
            return math.ceil(bound - 1e-6) >= cost - 1e-9
        return bound >= cost - 1e-6

    def evaluate(self, edges, ub, fitch=None):
        """Exact cost of the tree if it can be <= ub, else None (steps 1-4 above)."""
        adj = self._adj(edges)
        if fitch is None:
            fitch = self.lower_bound(adj)

        t0 = time.time()
        h = self.labeling_cost(adj)
        self.seconds["labeling"] += time.time() - t0
        if h is not None and self._reaches(fitch, h):
            self.counts["fitch_labeling"] += 1
            return h

        t0 = time.time()
        g = self.group_bound(adj)
        self.seconds["group_bound"] += time.time() - t0
        if self._above(g, ub):
            self.counts["ruled_out_group"] += 1
            return None
        if h is not None and self._reaches(g, h):
            self.counts["group_bound"] += 1
            return h

        t0 = time.time()
        lp, lp_exact = self.lp_bound(edges)
        self.seconds["lp"] += time.time() - t0
        if self._above(lp, ub):
            self.counts["ruled_out_lp"] += 1
            return None
        if h is not None and self._reaches(lp, h):
            self.counts["lp_reaches_labeling"] += 1
            return h
        if lp_exact:
            self.counts["lp_integral"] += 1
            return lp

        return self.ilp_cost(edges, ub)

    def ilp_cost(self, edges, ub):
        """Step 4: the fixed-tree small parsimony ILP; its exact cost, or None if it
        is above ub."""
        t0 = time.time()
        ilp = LargeParsimonyILP(
            [sorted(t) for t in self.leaf_tads], Ct=self.Ct, Cb=self.Cb, mode="convex",
            threads=self.threads, time_limit=self.time_limit,
            fixed_arcs=self.rooted_arcs(edges), laminarity="loop", **self._weight_kwargs(),
        )
        res = ilp.solve(cutoff=None if ub == float("inf") else ub + self.slack)
        self.seconds["ilp"] += time.time() - t0
        if res["status"] == "cutoff":
            self.counts["ruled_out_ilp"] += 1
            return None
        if res["status"] != "optimal":
            raise RuntimeError(f"small parsimony solve failed: {res['status']}")
        self.counts["ilp"] += 1
        return res["cost"]

    def _progress(self):
        return dict(self.counts)

    # -- trees -------------------------------------------------------------------
    @staticmethod
    def _adj(edges):
        adj = {}
        for a, b in edges:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        return adj

    def rooted_arcs(self, edges):
        """1-based arcs of the tree rooted on leaf 0's pendant edge, numbered as
        LargeParsimonyILP expects: leaves 1..k, internal vertices in postorder
        k+1 .. 2k-2, root 2k-1."""
        k = self.k
        adj = self._adj(edges)
        arcs = []
        counter = [k + 1]

        def rec(node, parent):
            if node < k:
                return node + 1
            kids = [rec(c, node) for c in adj[node] if c != parent]
            me = counter[0]
            counter[0] += 1
            arcs.extend((me, c) for c in kids)
            return me

        (nb,) = adj[0]
        top = rec(nb, 0)
        root = 2 * k - 1
        arcs += [(root, 1), (root, top)]
        return arcs

    def split_masks(self, edges):
        """For each internal edge, the bitmask (bit i = leaf i) of the leaves on one side."""
        k = self.k
        adj = self._adj(edges)
        out = []
        for a, b in edges:
            if a < k or b < k:
                continue
            side, stack, seen = 0, [b], {a, b}
            while stack:
                w = stack.pop()
                if w < k:
                    side |= 1 << w
                for n in adj[w]:
                    if n not in seen:
                        seen.add(n)
                        stack.append(n)
            out.append(side)
        return out

    def splits(self, edges):
        """Leaf sets (0-based) on one side of each internal edge."""
        return [frozenset(i for i in range(self.k) if m >> i & 1) for m in self.split_masks(edges)]

    # -- search ------------------------------------------------------------------
    def _start(self):
        k = self.k
        return [(0, k), (1, k), (2, k)]

    def _insert(self, edges, edge, leaf, new):
        a, b = edge
        out = [e for e in edges if e != edge]
        out += [(a, new), (new, b), (new, leaf)]
        return out

    def edges_of(self, path):
        """Rebuild a tree from the edges its leaves 3..k-1 were inserted on."""
        edges = self._start()
        for depth, idx in enumerate(path, start=3):
            edges = self._insert(edges, edges[idx], depth, self.k + depth - 2)
        return edges

    def _greedy(self):
        """Stepwise addition, each leaf on the edge of least Fitch bound."""
        edges = self._start()
        for depth in range(3, self.k):
            best = None
            for e in edges:
                cand = self._insert(edges, e, depth, self.k + depth - 2)
                lb = self.lower_bound(self._adj(cand))
                if best is None or lb < best[0]:
                    best = (lb, cand)
            edges = best[1]
        return edges

    def run(self):
        t_start = time.time()
        k = self.k

        # initial upper bound: greedy stepwise addition, scored exactly
        greedy = self._greedy()
        ub = self.evaluate(greedy, float("inf"))
        ub_initial = ub
        if self.verbose:
            print(f"  greedy tree: Fitch LB={self.lower_bound(self._adj(greedy)):g}, exact={ub:g}", flush=True)

        # every complete tree whose Fitch bound is <= ub, as its insertion path
        survivors = []
        n_nodes = 0

        def dfs(edges, depth, path):
            nonlocal n_nodes
            new = k + depth - 2
            for idx, e in enumerate(list(edges)):
                cand = self._insert(edges, e, depth, new)
                n_nodes += 1
                lb = self.lower_bound(self._adj(cand))
                if self._above(lb, ub):
                    continue
                if depth + 1 < k:
                    dfs(cand, depth + 1, path + (idx,))
                else:
                    survivors.append((lb, path + (idx,)))

        dfs(self._start(), 3, ())
        t_enum = time.time() - t_start
        if self.verbose:
            print(f"  {n_nodes} partial trees, {len(survivors)} complete trees with Fitch LB <= {ub:g} "
                  f"({t_enum:.0f}s)", flush=True)

        # score them in increasing Fitch bound; ub only falls, so stop once LB > ub
        survivors.sort(key=lambda s: s[0])
        costs = []          # (cost, path) of every tree that came in at <= ub
        last_report = time.time()
        for n_done, (lb, path) in enumerate(survivors):
            if self._above(lb, ub):
                break
            cost = self.evaluate(self.edges_of(path), ub, fitch=lb)
            if self.verbose and time.time() - last_report > 60:
                last_report = time.time()
                print(f"  scored {n_done + 1} of {len(survivors)} trees (stops early once the "
                      f"Fitch bound passes ub={ub:g}), {self._progress()}", flush=True)
            if cost is None:
                continue
            costs.append((cost, path))
            ub = min(ub, cost)

        tie = 1e-9 if self.integral else 1e-6 * max(1.0, abs(ub))
        optimal = [path for cost, path in costs if cost <= ub + tie]
        return {
            "status": "optimal",
            "cost": ub,
            "optimal_paths": optimal,
            "ub_initial": ub_initial,
            "n_bnb_nodes": n_nodes,
            "n_complete_survivors": len(survivors),
            "settled_by": dict(self.counts),
            "step_seconds": {key: round(v, 2) for key, v in self.seconds.items()},
            "enumeration_seconds": t_enum,
            "seconds": time.time() - t_start,
        }
