#!/usr/bin/env python3
"""
large_parsimony_ilp.py

The large parsimony integer linear program (TAD-LP) of Supp. Sec. "The large
parsimony integer linear program": given k observed TAD hierarchies and no tree,
choose the edges of a rooted binary tree AND the TAD hierarchies of its k-1
internal vertices so that the total TAD distance over the chosen edges is minimal.

Vertex numbering follows the paper:
    1 .. k          leaves (leaf i carries leaf_tads[i-1])
    k+1 .. 2k-1     internal vertices, 2k-1 is the root
Candidate arcs are (u, v) with u internal and v < u; e[u,v] = 1 makes u the parent
of v. "Exactly one parent" for every non-root vertex plus "exactly two children"
for every internal vertex already make the chosen arcs a rooted binary tree, since
every arc points to a smaller index (no cycles).

The difference variables of an arc are released when the arc is not chosen:
    delta >= x_u - x_v - (1 - e_uv),   delta >= x_v - x_u - (1 - e_uv)
so an unchosen arc contributes 0 and a chosen one contributes |x_u - x_v|.

Candidate TADs are every [a, b] with a an observed start and b an observed end
(Eq. ilp_candidates), laminarity at internal vertices is separated on demand, and
leaves are constants rather than fixed variables (smaller model, same optimum).

For k = 4 there are only three unrooted topologies (quartets). Vertex k+1 = 5 can
only take leaves as children, so its two children ARE the cherry that defines the
quartet, and ``solve(pin_partner=c)`` restricts the search to the quartet that
pairs leaf 1 with leaf c. That lets a caller score all three quartets with the
same model and the same laminarity cuts.

``root_leaf=r`` adds e[2k-1, r] = 1: the root sits on the pendant edge of leaf r.
The objective does not depend on the root (it scores an arc by the two
hierarchies it joins, and every unrooted tree has a pendant edge at r), so this
excludes no unrooted topology and only removes root-placement symmetry. For
k = 4 it leaves exactly one rooted tree per quartet (18 -> 3), which is what makes
the free solve as fast as a pinned one.

``fixed_arcs`` turns the model into small parsimony on one given rooted tree
(numbered as above), and ``initial_cut_pairs`` seeds it with laminarity cuts found
elsewhere (valid for every tree, as they only depend on the candidates).
``laminarity`` picks how laminarity is enforced ("lazy", "loop", or "off" for a
lower bound), ``tad_weights`` / ``start_weights`` / ``end_weights`` give the
weighted TAD distance d_w of the paper (per-TAD and per-boundary costs in place of
Ct and Cb; convex only), ``lp_bound()`` gives the LP relaxation, and ``solve(cutoff=c)`` stops
early when no solution costs at most c. Used by topology_search.py.

Two optional importance terms reproduce the weighted small parsimony solvers
(convex mode only). Both are added to the parsimony cost Ct*T + Cb*B:

``vertex_importance = {u: imp}`` (check_lp_constrained_weighted.py): for an
internal vertex u and every candidate TAD t, with imp(a, b) in [0, 1],
    - sum_t [ x[u,t] * imp(t) + (1 - x[u,t]) * (1 - imp(t)) ].
Which biological node a vertex stands for is the caller's business (with k = 4 and
root_leaf = 1, vertex 6 is the one next to leaf 1 and vertex 5 is the cherry parent).

``leaf_importance = {v: {(a, b): {"x": ., "s": ., "e": .}}}``
(check_lp_constrained_weighted_updated.py): for a leaf v, each of its TADs [a, b]
with scores, and c = leaf_importance_scale * score, the PARENT p of v pays
    leaf_importance_weight * [c + q * (1 - 2c)]
with q = x[p,(a,b)] for "x", the start indicator y[p,a] for "s" and the end
indicator z[p,b] for "e". The parent is chosen by the ILP, so each candidate parent
u gets the product e[u,v] * q, linearised with one auxiliary variable (only the
side of the linearisation the sign of the coefficient needs).
"""

import time
from collections import defaultdict

MODES = ("convex", "lex_boundary_first", "lex_tad_first")


def candidate_universe(leaf_tads):
    """Observed starts, observed ends, and every [a, b] they delimit (a < b)."""
    starts = sorted({a for tads in leaf_tads for (a, _) in tads})
    ends = sorted({b for tads in leaf_tads for (_, b) in tads})
    candidates = [(a, b) for a in starts for b in ends if a < b]
    return starts, ends, candidates


def candidate_arcs(k):
    return [(u, v) for u in range(k + 1, 2 * k) for v in range(1, 2 * k - 1) if v < u]


def crossing_pairs(tads):
    """Pairs of TADs that partially overlap (a < c < b < d). Touching is allowed."""
    ordered = sorted(tads)
    pairs = []
    for i in range(len(ordered)):
        a, b = ordered[i]
        for j in range(i + 1, len(ordered)):
            c, d = ordered[j]
            if c >= b:
                break
            if a < c < b < d:
                pairs.append(((a, b), (c, d)))
    return pairs


def tad_distance_terms(tads_u, tads_v):
    """(TAD changes, boundary changes) between two hierarchies: |T symdiff| and
    |L symdiff| + |R symdiff|, i.e. the two sums the ILP objective weights."""
    starts_u = {a for a, _ in tads_u}
    starts_v = {a for a, _ in tads_v}
    ends_u = {b for _, b in tads_u}
    ends_v = {b for _, b in tads_v}
    t_changes = len(tads_u ^ tads_v)
    b_changes = len(starts_u ^ starts_v) + len(ends_u ^ ends_v)
    return t_changes, b_changes


def score_key(mode, Ct, Cb, t_changes, b_changes, importance_term=0.0):
    """Value the solver minimises, as something comparable with < and ==.
    Lexicographic modes compare as tuples, so ties are exact. With an importance
    term the cost is fractional; rounding to 6 decimals absorbs summation-order noise."""
    if mode == "convex":
        return (round(Ct * t_changes + Cb * b_changes + importance_term, 6),)
    if mode == "lex_boundary_first":
        return (round(Cb * b_changes, 9), round(Ct * t_changes, 9))
    if mode == "lex_tad_first":
        return (round(Ct * t_changes, 9), round(Cb * b_changes, 9))
    raise ValueError(f"Unknown mode {mode!r}; use one of {MODES}")


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------
def unrooted_adjacency(arcs):
    """Undirected adjacency of the chosen arcs with every degree-2 vertex (the
    root of a binary tree) suppressed, i.e. the unrooted tree the objective sees."""
    adj = defaultdict(set)
    for u, v in arcs:
        adj[u].add(v)
        adj[v].add(u)
    changed = True
    while changed:
        changed = False
        for w in list(adj):
            if len(adj[w]) == 2:
                a, b = adj[w]
                adj[a].discard(w)
                adj[b].discard(w)
                adj[a].add(b)
                adj[b].add(a)
                del adj[w]
                changed = True
                break
    return dict(adj)


def nontrivial_splits(arcs, k):
    """Leaf bipartitions induced by the internal edges of the unrooted tree.
    Each split is returned as the frozenset of leaves on the side WITHOUT leaf 1."""
    adj = unrooted_adjacency(arcs)
    leaves = set(range(1, k + 1))
    splits = set()
    for u in adj:
        for v in adj[u]:
            if u > v or u in leaves or v in leaves:
                continue
            # leaves reachable from v without crossing the edge (u, v)
            side, stack, seen = set(), [v], {u, v}
            while stack:
                w = stack.pop()
                if w in leaves:
                    side.add(w)
                for nxt in adj[w]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            if 1 in side:
                side = leaves - side
            splits.add(frozenset(side))
    return splits


def quartet_partner(arcs):
    """For k = 4: the leaf that the tree pairs with leaf 1."""
    splits = nontrivial_splits(arcs, 4)
    if len(splits) != 1:
        raise RuntimeError(f"Expected one nontrivial split on 4 leaves, got {splits}")
    (far_side,) = splits
    (partner,) = {2, 3, 4} - far_side
    return partner


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class LargeParsimonyILP:
    def __init__(
        self,
        leaf_tads,
        Ct=1.0,
        Cb=1.0,
        mode="convex",
        threads=None,
        time_limit=None,
        root_leaf=None,
        lp_method=None,
        solver_log=False,
        max_cut_rounds=200,
        verbose=False,
        vertex_importance=None,
        leaf_importance=None,
        leaf_importance_weight=1.0,
        leaf_importance_scale=3.0,
        fixed_arcs=None,
        initial_cut_pairs=None,
        laminarity=None,
        tad_weights=None,
        start_weights=None,
        end_weights=None,
    ):
        import gurobipy as gp
        from gurobipy import GRB

        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; use one of {MODES}")
        if len(leaf_tads) < 3:
            raise ValueError("Large parsimony needs at least 3 leaves.")

        self.gp, self.GRB = gp, GRB
        self.Ct, self.Cb, self.mode = Ct, Cb, mode
        self.max_cut_rounds = max_cut_rounds
        self.verbose = verbose
        self.solver_log = solver_log or verbose
        self.lp_method = lp_method
        self.vertex_importance = vertex_importance or {}
        self.leaf_importance = leaf_importance or {}
        self.leaf_importance_weight = float(leaf_importance_weight)
        self.leaf_importance_scale = float(leaf_importance_scale)
        self.weighted = bool(self.vertex_importance or self.leaf_importance)
        if self.weighted and mode != "convex":
            raise ValueError("Importance terms are only implemented for mode='convex'.")

        self.k = k = len(leaf_tads)
        # a TAD with a >= b is not a candidate, so it could not be pinned at its leaf
        self.leaf_tads = {i + 1: {tuple(t) for t in tads if t[0] < t[1]}
                          for i, tads in enumerate(leaf_tads)}
        self.leaf_starts = {v: {a for a, _ in tads} for v, tads in self.leaf_tads.items()}
        self.leaf_ends = {v: {b for _, b in tads} for v, tads in self.leaf_tads.items()}
        self.internal = list(range(k + 1, 2 * k))
        self.root = 2 * k - 1
        if root_leaf is not None and not 1 <= root_leaf <= k:
            raise ValueError(f"root_leaf must be a leaf in 1..{k}")
        self.root_leaf = root_leaf
        if any(u not in self.internal for u in self.vertex_importance):
            raise ValueError("vertex_importance keys must be internal vertices.")
        if any(not 1 <= v <= k for v in self.leaf_importance):
            raise ValueError("leaf_importance keys must be leaves.")

        self.starts, self.ends, self.candidates = candidate_universe(list(self.leaf_tads.values()))
        if not self.candidates:
            raise ValueError("No candidate TADs (no observed start lies left of an observed end).")
        if fixed_arcs is None:
            self.arcs = candidate_arcs(k)
        else:
            # small parsimony on a given rooted tree, numbered as above (children < parent)
            self.arcs = sorted(tuple(a) for a in fixed_arcs)
            if root_leaf is not None:
                raise ValueError("root_leaf does not apply with fixed_arcs.")
            if set(self.arcs) - set(candidate_arcs(k)) or len(self.arcs) != 2 * k - 2:
                raise ValueError("fixed_arcs must be the 2k-2 arcs of a rooted binary tree "
                                 "with children numbered below their parent.")
        self.fixed = fixed_arcs is not None
        if laminarity not in (None, "lazy", "loop", "off"):
            raise ValueError("laminarity must be 'lazy', 'loop' or 'off'.")
        self.laminarity = laminarity
        # weighted TAD distance d_w: per-TAD / per-boundary costs replacing Ct / Cb
        # (missing keys fall back to Ct / Cb)
        self.tad_weights = tad_weights or {}
        self.start_weights = start_weights or {}
        self.end_weights = end_weights or {}
        self.distance_weighted = bool(tad_weights or start_weights or end_weights)
        if self.distance_weighted and mode != "convex":
            raise ValueError("Weighted TAD distance is only implemented for mode='convex'.")
        self.cut_pairs = set()     # laminarity cuts already in the model
        self._build()
        for pair in initial_cut_pairs or ():
            if pair[0] in self.x_index and pair[1] in self.x_index:
                self._add_cut_pair(pair)
        self.set_params(threads=threads, time_limit=time_limit)

    # -- model construction --------------------------------------------------
    def _build(self):
        gp, GRB = self.gp, self.GRB
        k, internal, arcs = self.k, self.internal, self.arcs
        cands, starts, ends = self.candidates, self.starts, self.ends
        t0 = time.time()

        m = gp.Model("TAD_large_parsimony")
        m.setParam("OutputFlag", 1 if self.solver_log else 0)
        m.setParam("MIPGap", 0.0)   # the topology comparison needs true optima
        if (self.mode == "convex" and not self.weighted and not self.distance_weighted
                and float(self.Ct).is_integer() and float(self.Cb).is_integer()):
            # Every solution then has an integer cost, so a gap below 1 already
            # proves optimality: stopping there is exact, and skips closing the last unit.
            m.setParam("MIPGapAbs", 1.0 - 1e-6)
        if self.lp_method is not None:
            # Gurobi Method for the root relaxation: -1 auto, 1 dual simplex, 2 barrier, 3 concurrent
            m.setParam("Method", int(self.lp_method))
        self.m = m

        # Laminarity is separated inside one branch and bound (lazy constraints in a
        # callback) rather than by re-solving after each round: at 100 kb one solve
        # takes minutes and the re-solve loop adds only a few pairs per round.
        # Gurobi's callbacks do not combine with multi-objective, so the lexicographic
        # modes keep the re-solve loop.
        # laminarity: "lazy" (callback), "loop" (re-solve with the crossing pairs of
        # each optimum), or "off" (dropped: the solve is then a lower bound)
        self.laminarity = self.laminarity or ("lazy" if self.mode == "convex" else "loop")
        if self.laminarity == "lazy" and self.mode != "convex":
            raise ValueError("Lazy laminarity cuts need mode='convex'.")
        self.use_lazy = self.laminarity == "lazy"
        if self.use_lazy:
            m.setParam("LazyConstraints", 1)

        e = m.addVars(arcs, vtype=GRB.BINARY, lb=1 if self.fixed else 0, name="e")
        x = m.addVars([(v, t) for v in internal for t in cands], vtype=GRB.BINARY, name="x")
        self.x_lists = {v: [x[v, t] for t in cands] for v in internal}
        self.x_index = set(cands)
        y = m.addVars([(v, p) for v in internal for p in starts], lb=0, ub=1, name="y")
        z = m.addVars([(v, p) for v in internal for p in ends], lb=0, ub=1, name="z")
        self.e, self.x, self.y, self.z = e, x, y, z

        # Rooted binary tree: one parent per non-root vertex, two children per internal vertex.
        for v in range(1, 2 * k - 1):
            m.addConstr(gp.quicksum(e[u, w] for (u, w) in arcs if w == v) == 1, name=f"parent_{v}")
        for u in internal:
            m.addConstr(gp.quicksum(e[w, v] for (w, v) in arcs if w == u) == 2, name=f"children_{u}")
        if self.root_leaf is not None:
            m.addConstr(e[self.root, self.root_leaf] == 1, name="root_on_leaf_edge")

        # A TAD carries its two boundaries; a boundary delimits at least one TAD.
        starts_at, ends_at = defaultdict(list), defaultdict(list)
        for (a, b) in cands:
            starts_at[a].append((a, b))
            ends_at[b].append((a, b))
        for v in internal:
            for (a, b) in cands:
                m.addConstr(x[v, (a, b)] <= y[v, a])
                m.addConstr(x[v, (a, b)] <= z[v, b])
            for p in starts:
                m.addConstr(y[v, p] <= gp.quicksum(x[v, t] for t in starts_at[p]))
            for p in ends:
                m.addConstr(z[v, p] <= gp.quicksum(x[v, t] for t in ends_at[p]))

        # Difference variables, active only on chosen arcs.
        dT = m.addVars([(u, v, t) for (u, v) in arcs for t in cands], lb=0, ub=1, name="dT")
        dL = m.addVars([(u, v, p) for (u, v) in arcs for p in starts], lb=0, ub=1, name="dL")
        dR = m.addVars([(u, v, p) for (u, v) in arcs for p in ends], lb=0, ub=1, name="dR")

        def xval(v, t):
            return (1 if t in self.leaf_tads[v] else 0) if v <= k else x[v, t]

        def yval(v, p):
            return (1 if p in self.leaf_starts[v] else 0) if v <= k else y[v, p]

        def zval(v, p):
            return (1 if p in self.leaf_ends[v] else 0) if v <= k else z[v, p]

        def add_absdiff(d, vu, vv, earc):
            # d >= |vu - vv| - (1 - earc); vu is always a variable (u is internal).
            if isinstance(vv, int):
                if vv == 1:
                    m.addConstr(d >= earc - vu)          # other side is <= 0, redundant
                else:
                    m.addConstr(d >= vu + earc - 1)      # other side is <= 0, redundant
            else:
                m.addConstr(d >= vu - vv + earc - 1)
                m.addConstr(d >= vv - vu + earc - 1)

        for (u, v) in arcs:
            earc = e[u, v]
            for t in cands:
                add_absdiff(dT[u, v, t], xval(u, t), xval(v, t), earc)
            for p in starts:
                add_absdiff(dL[u, v, p], yval(u, p), yval(v, p), earc)
            for p in ends:
                add_absdiff(dR[u, v, p], zval(u, p), zval(v, p), earc)

        sum_T = dT.sum()
        sum_B = dL.sum() + dR.sum()
        m.ModelSense = GRB.MINIMIZE
        if self.mode == "convex" and self.distance_weighted:
            wt, ws, we = self.w_tad, self.w_start, self.w_end
            weighted = (gp.quicksum(wt(t) * dT[u, v, t] for (u, v) in arcs for t in cands)
                        + gp.quicksum(ws(p) * dL[u, v, p] for (u, v) in arcs for p in starts)
                        + gp.quicksum(we(p) * dR[u, v, p] for (u, v) in arcs for p in ends))
            m.setObjective(weighted + self._importance_expr())
        elif self.mode == "convex":
            m.setObjective(self.Ct * sum_T + self.Cb * sum_B + self._importance_expr())
        elif self.mode == "lex_boundary_first":
            m.setObjectiveN(self.Cb * sum_B, index=0, priority=2, name="boundaries")
            m.setObjectiveN(self.Ct * sum_T, index=1, priority=1, name="tads")
        else:
            m.setObjectiveN(self.Ct * sum_T, index=0, priority=2, name="tads")
            m.setObjectiveN(self.Cb * sum_B, index=1, priority=1, name="boundaries")

        m.update()
        self.build_seconds = time.time() - t0
        if self.verbose:
            print(f"Built large parsimony ILP: k={k}, arcs={len(arcs)}, candidates={len(cands)}, "
                  f"vars={m.NumVars}, constrs={m.NumConstrs} in {self.build_seconds:.1f}s")

    def w_tad(self, t):
        return self.tad_weights.get(t, self.Ct)

    def w_start(self, p):
        return self.start_weights.get(p, self.Cb)

    def w_end(self, p):
        return self.end_weights.get(p, self.Cb)

    def distance(self, tads_u, tads_v):
        """The (weighted) TAD distance the objective charges on one arc."""
        if not self.distance_weighted:
            dt, db = tad_distance_terms(tads_u, tads_v)
            return self.Ct * dt + self.Cb * db
        su, sv = {a for a, _ in tads_u}, {a for a, _ in tads_v}
        eu, ev = {b for _, b in tads_u}, {b for _, b in tads_v}
        return (sum(self.w_tad(t) for t in tads_u ^ tads_v)
                + sum(self.w_start(p) for p in su ^ sv)
                + sum(self.w_end(p) for p in eu ^ ev))

    def _importance_expr(self):
        """Linear expression (with constant) of both importance terms; 0 if none."""
        gp, GRB, m = self.gp, self.GRB, self.m
        expr = gp.LinExpr()

        # vertex importance: - sum_t [x*imp + (1-x)*(1-imp)] = - sum(1-imp) - sum x*(2imp-1)
        self.vertex_imp_values = {}
        for u, imp in self.vertex_importance.items():
            vals = [float(imp(a, b)) for (a, b) in self.candidates]
            self.vertex_imp_values[u] = dict(zip(self.candidates, vals))
            expr.addConstant(-sum(1.0 - v for v in vals))
            expr.addTerms([-(2.0 * v - 1.0) for v in vals], self.x_lists[u])

        # leaf importance: w * [c + q_parent * (1 - 2c)], parent chosen by the arcs
        w = self.leaf_importance_weight
        self.leaf_imp_terms = []        # (leaf, kind, key, c) for the recount
        for v, entries in self.leaf_importance.items():
            parents = [u for (u, vv) in self.arcs if vv == v]
            for (a, b), scores in entries.items():
                for kind, key in (("x", (a, b)), ("s", a), ("e", b)):
                    score = scores.get(kind)
                    if score is None:
                        continue
                    if (parents[0], key) not in (self.x if kind == "x" else self.y if kind == "s" else self.z):
                        continue        # not a candidate (cannot happen for a leaf TAD with a < b)
                    c = self.leaf_importance_scale * float(score)
                    self.leaf_imp_terms.append((v, kind, key, c))
                    expr.addConstant(w * c)          # every leaf has exactly one parent
                    coef = w * (1.0 - 2.0 * c)
                    if coef == 0:
                        continue
                    for u in parents:
                        q = (self.x if kind == "x" else self.y if kind == "s" else self.z)[u, key]
                        earc = self.e[u, v]
                        p = m.addVar(lb=0, ub=1)
                        if coef < 0:             # reward: p <= q AND p <= e
                            m.addConstr(p <= q)
                            m.addConstr(p <= earc)
                        else:                    # penalty: p >= q + e - 1
                            m.addConstr(p >= q + earc - 1)
                        expr.addTerms(coef, p)
        return expr

    def importance_value(self, arcs, tads):
        """Both importance terms evaluated on a solution (for the recount)."""
        total = 0.0
        for u, vals in self.vertex_imp_values.items():
            chosen = tads[u]
            total -= sum(v if t in chosen else 1.0 - v for t, v in vals.items())
        if self.leaf_imp_terms:
            parent = {v: u for (u, v) in arcs}
            w = self.leaf_importance_weight
            bounds = {}
            for (v, kind, key, c) in self.leaf_imp_terms:
                u = parent[v]
                if u not in bounds:
                    bounds[u] = ({a for a, _ in tads[u]}, {b for _, b in tads[u]})
                starts_u, ends_u = bounds[u]
                q = (key in tads[u]) if kind == "x" else (key in starts_u) if kind == "s" else (key in ends_u)
                total += w * (c + (1.0 - 2.0 * c) * q)
        return total

    def set_params(self, threads=None, time_limit=None):
        """Gurobi threads, and a time limit in seconds per Gurobi call (None/0 = none)."""
        if threads:
            self.m.setParam("Threads", int(threads))
        if time_limit:
            self.m.setParam("TimeLimit", float(time_limit))

    # -- solving ---------------------------------------------------------------
    def _read_solution(self):
        arcs = sorted((u, v) for (u, v) in self.arcs if self.e[u, v].X > 0.5)
        tads = dict(self.leaf_tads)
        for v in self.internal:
            tads[v] = {t for t in self.candidates if self.x[v, t].X > 0.5}
        return arcs, tads

    def _add_cut_pair(self, pair):
        """x[w, t1] + x[w, t2] <= 1 at every internal vertex w: the cut is valid
        everywhere, and a crossing pair tends to move to another vertex otherwise."""
        if pair in self.cut_pairs:
            return False
        self.cut_pairs.add(pair)
        t1, t2 = pair
        for w in self.internal:
            self.m.addConstr(self.x[w, t1] + self.x[w, t2] <= 1)
        return True

    def _add_laminarity_cuts(self, tads):
        """Add a model cut for every crossing pair of the solution; returns how many."""
        return sum(self._add_cut_pair(pair) for v in self.internal for pair in crossing_pairs(tads[v]))

    def _lazy_callback(self, lazy_pairs):
        GRB = self.GRB

        def callback(model, where):
            if where != GRB.Callback.MIPSOL:
                return
            for v in self.internal:
                vals = model.cbGetSolution(self.x_lists[v])
                tads = [t for t, val in zip(self.candidates, vals) if val > 0.5]
                for pair in crossing_pairs(tads):
                    if pair in lazy_pairs or pair in self.cut_pairs:
                        continue
                    lazy_pairs.add(pair)
                    t1, t2 = pair
                    for w in self.internal:
                        model.cbLazy(self.x[w, t1] + self.x[w, t2] <= 1)

        return callback

    def lp_bound(self):
        """Optimum of the LP relaxation of the current model (a lower bound on solve())."""
        relaxed = self.m.relax()
        relaxed.setParam("OutputFlag", 0)
        relaxed.setParam("LazyConstraints", 0)
        relaxed.optimize()
        if relaxed.Status != self.GRB.OPTIMAL:
            raise RuntimeError(f"LP relaxation status {relaxed.Status}")
        return relaxed.ObjVal

    def solve(self, pin_partner=None, cutoff=None):
        """Solve to optimality (with laminarity separation). With k = 4,
        ``pin_partner=c`` restricts the tree to the quartet {1, c} | {rest}."""
        GRB = self.GRB
        m = self.m
        pins = []
        if pin_partner is not None:
            if self.k != 4:
                raise ValueError("pin_partner is only defined for 4 leaves.")
            if pin_partner not in (2, 3, 4):
                raise ValueError("pin_partner must be leaf 2, 3 or 4.")
            # Vertex 5 can only have leaf children, so its two children are the cherry.
            c = pin_partner
            d, f = sorted({2, 3, 4} - {c})
            pins.append(m.addConstr(self.e[5, 1] == self.e[5, c], name="pin_a"))
            pins.append(m.addConstr(self.e[5, d] == self.e[5, f], name="pin_b"))

        if cutoff is not None:
            # prune anything worse than cutoff; the solve then reports "cutoff" if
            # no solution is at least that good
            m.setParam("Cutoff", float(cutoff))
        t0 = time.time()
        rounds = 0
        lazy_pairs = set()
        try:
            while True:
                if self.use_lazy:
                    m.optimize(self._lazy_callback(lazy_pairs))
                else:
                    m.optimize()
                if m.Status != GRB.OPTIMAL:
                    status = {GRB.TIME_LIMIT: "time_limit", GRB.INFEASIBLE: "infeasible",
                              GRB.INTERRUPTED: "interrupted", GRB.CUTOFF: "cutoff"}.get(m.Status, f"gurobi_status_{m.Status}")
                    return {"status": status, "seconds": time.time() - t0, "cut_rounds": rounds,
                            "gap": m.MIPGap if m.SolCount else None}
                arcs, tads = self._read_solution()
                # read before the model changes: modifying it discards the solution
                model_value = m.ObjVal if self.mode == "convex" else None
                # Lazy cuts live only for one optimize(); make them permanent so the
                # next solve (another quartet) starts with them.
                for pair in lazy_pairs:
                    self._add_cut_pair(pair)
                lazy_pairs.clear()
                # Safety net (and the whole mechanism in the lexicographic modes).
                if self.laminarity == "off" or self._add_laminarity_cuts(tads) == 0:
                    break
                rounds += 1
                if self.verbose:
                    print(f"  laminarity round {rounds}: {len(self.cut_pairs)} cut pairs so far")
                if rounds > self.max_cut_rounds:
                    return {"status": "cut_limit", "seconds": time.time() - t0, "cut_rounds": rounds}
        finally:
            for c in pins:
                m.remove(c)
            m.update()

        # Recount the objective from the solution itself: exact integers, and a
        # guard that the linearisation charged exactly the chosen arcs.
        t_changes = b_changes = 0
        for (u, v) in arcs:
            dt, db = tad_distance_terms(tads[u], tads[v])
            t_changes += dt
            b_changes += db
        parsimony = (sum(self.distance(tads[u], tads[v]) for (u, v) in arcs)
                     if self.distance_weighted else self.Ct * t_changes + self.Cb * b_changes)
        weight_term = self.importance_value(arcs, tads) if self.weighted else 0.0
        cost = parsimony + weight_term
        if self.mode == "convex":
            if abs(model_value - cost) > 1e-6 * max(1.0, abs(cost)):
                raise RuntimeError(f"Objective {model_value} != recount {cost}")

        return {
            "status": "optimal",
            "arcs": arcs,
            "tads": tads,
            "tad_changes": t_changes,
            "boundary_changes": b_changes,
            "parsimony_cost": parsimony,
            "importance_term": weight_term,
            "cost": cost,
            "key": ((round(cost, 6),) if self.distance_weighted
                    else score_key(self.mode, self.Ct, self.Cb, t_changes, b_changes, weight_term)),
            "seconds": time.time() - t0,
            "cut_rounds": rounds,
            "cut_pairs_total": len(self.cut_pairs),
        }
