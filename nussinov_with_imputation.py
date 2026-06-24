import re
import random
from tabnanny import verbose
from random_simulation import (
        generate_random_intervals,
        generate_random_laminar_intervals_from_list
    )
from check_lp_constrained import solve_lp_from_edges_and_leaves


import re
from functools import lru_cache
from copy import deepcopy

def infer_tads_with_dp(edges,
                       leaf_intervals,
                       start_boundaries,
                       end_boundaries,
                       Ct=1.0,
                       Cb=1.0,
                       trace=False):
    """
    DP that uses full child maps (no deltas). Each DP cell returns:
      (parent_intervals_set, score, choice_struct, children_map)
    where choice_struct is either:
      - None
      - ("call", func_name, i, j)           -- single recursive call
      - ("split", [(func_name, i, j), ...]) -- multiple recursive calls
    and children_map: dict child_id -> frozenset({(a,b), ...})
    """

    import re
    from copy import deepcopy

    def parse_interval(s):
        a, b = map(int, re.findall(r"\d+", s))
        return (a, b)

    children = [v for (u, v) in edges if u == 0]

    orig_child_intervals = {c: set(parse_interval(x) for x in leaf_intervals.get(c, []))
                            for c in children}
    initial_children_map = {c: frozenset(orig_child_intervals[c]) for c in children}

    positions = sorted(set(start_boundaries) | set(end_boundaries))
    n = len(positions)
    if n < 2:
        if trace:
            return set(), 0.0, initial_children_map, []
        else:
            return set(), 0.0, initial_children_map

    pos_index = {v:i for i,v in enumerate(positions)}
    num_children = len(children)

    # -------------------------
    # Helpers
    # -------------------------
    def merge_child_maps(m1, m2):
        merged = {}
        keys = set(m1.keys()) | set(m2.keys())
        for k in keys:
            s1 = set(m1.get(k, ()))
            s2 = set(m2.get(k, ()))
            merged[k] = frozenset(s1 | s2)
        return merged

    def child_has_start_at(c, pos, cmap):
        return any(x == pos for (x,y) in cmap.get(c, ()))

    def child_has_end_at(c, pos, cmap):
        return any(y == pos for (x,y) in cmap.get(c, ()))

    def count_children_with_interval(a,b, cmap):
        return sum(1 for c in children if (a,b) in cmap.get(c, ()))

    def count_children_missing_interval(a,b, cmap):
        return sum(1 for c in children if (a,b) not in cmap.get(c, ()))

    def count_child_intervals_start_ge_i_end_eq_j(i,j, cmap):
        a = positions[i]; b = positions[j]
        cnt = 0
        for c in children:
            for (x,y) in cmap.get(c, ()):
                if (x,y) == (a,b):
                    continue
                if x >= a and y == b:
                    cnt += 1
        return cnt

    def count_child_intervals_start_eq_i_end_le_j(i,j, cmap):
        a = positions[i]; b = positions[j]
        cnt = 0
        for c in children:
            for (x,y) in cmap.get(c, ()):
                if (x,y) == (a,b):
                    continue
                if x == a and y <= b:
                    cnt += 1
        return cnt

    def count_child_intervals_crossing_k(i,k,j, cmap):
        a = positions[i]; pk = positions[k]; b = positions[j]
        cnt = 0
        for c in children:
            for (x,y) in cmap.get(c, ()):
                if (x,y) == (a,b):
                    continue
                if a <= x < pk and pk < y <= b:
                    cnt += 1
        return cnt
    def boundary_cost_include_start(a_pos, cmap):
        return Cb * sum(1 for c in children if not child_has_start_at(c, a_pos, cmap))
    def boundary_cost_include_end(b_pos, cmap):
        return Cb * sum(1 for c in children if not child_has_end_at(c, b_pos, cmap))  
    def boundary_cost_exclude_start(a_pos, cmap):
        return Cb * sum(1 for c in children if child_has_start_at(c, a_pos, cmap))
    def boundary_cost_exclude_end(b_pos, cmap):
        return Cb * sum(1 for c in children if child_has_end_at(c, b_pos, cmap))


    def laminar_ok_to_impute_for_child(c, interval, cmap):
        a,b = interval
        for (x,y) in cmap.get(c, ()):
            if (x,y) == (a,b):
                continue
            # crossing patterns
            if (x < a < y < b) or (a < x < b < y):
                return False
        return True

    # -------------------------
    # Memo tables
    # -------------------------
    inf = float("inf")
    dp0_done = [[False]*n for _ in range(n)]
    dpl_done = [[False]*n for _ in range(n)]
    dpr_done = [[False]*n for _ in range(n)]
    dp0_val = [[None]*n for _ in range(n)]
    dpl_val = [[None]*n for _ in range(n)]
    dpr_val = [[None]*n for _ in range(n)]

    # -------------------------
    # Core DP functions
    # -------------------------
    def dp0(i, j):
        if i >= j:
            return frozenset(), inf, None, initial_children_map
        if dp0_done[i][j]:
            return dp0_val[i][j]

        a = positions[i]; b = positions[j]
        if a not in start_boundaries or b not in end_boundaries:
            dp0_done[i][j] = True
            dp0_val[i][j] = (frozenset(), inf, None, initial_children_map)
            return dp0_val[i][j]

        # base case adjacent
        if j == i + 1:
            # INCLUDE: impute into children if laminar
            cmap = {c: set(initial_children_map[c]) for c in children}
            for c in children:
                if (a,b) in cmap[c]:
                    continue
                cmap_view = {k: frozenset(v) for k,v in cmap.items()}
                if laminar_ok_to_impute_for_child(c, (a,b), cmap_view):
                    cmap[c].add((a,b))
            cmap_frozen = {c: frozenset(cmap[c]) for c in children}
            miss = count_children_missing_interval(a,b, cmap_frozen)
            bcost = boundary_cost_include_start(a, cmap_frozen) + boundary_cost_include_end(b, cmap_frozen)
            include_score = Ct * miss + bcost
            include_parent = frozenset({(a,b)})

            # EXCLUDE: keep children unchanged
            cmap_ex = initial_children_map
            have = count_children_with_interval(a,b, cmap_ex)
            bcost_ex = boundary_cost_exclude_start(a, cmap_ex) + boundary_cost_exclude_end(b, cmap_ex)
            exclude_score = Ct * have + bcost_ex
            exclude_parent = frozenset()

            if include_score <= exclude_score:
                res = (include_parent, include_score, ("base", "INCLUDE"), cmap_frozen)
            else:
                res = (exclude_parent, exclude_score, ("base", "EXCLUDE"), cmap_ex)

            dp0_done[i][j] = True
            dp0_val[i][j] = res
            return res

        # general case
        best_score = inf
        best_parent_set = frozenset()
        best_choice = None
        best_cmap = initial_children_map

        # 1. INCLUDE [a,b] branches
        # 1.1 call dpl(i, j-1)
        pset, sc, ch, cmap_sub = dpl(i, j-1)
        cmap_mut = {c: set(cmap_sub.get(c, frozenset())) for c in children}
        for c in children:
            if (a,b) in cmap_mut[c]:
                continue
            cmap_view = {k: frozenset(v) for k,v in cmap_mut.items()}
            if laminar_ok_to_impute_for_child(c, (a,b), cmap_view):
                cmap_mut[c].add((a,b))
        cmap_after = {c: frozenset(cmap_mut[c]) for c in children}

        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i,j, cmap_after)

        miss = count_children_missing_interval(a,b, cmap_after)
        c=positions[j-1]
        bcost = boundary_cost_include_start(a, cmap_after) + boundary_cost_include_end(b, cmap_after)+boundary_cost_exclude_start(c, cmap_after)
        total = sc + Ct * extra_cnt + Ct * miss + bcost
        cand_parent = frozenset({(a,b)}) | pset
        if total <= best_score:
            best_score = total
            best_parent_set = cand_parent
            best_choice = ("call", "dpl", i, j-1)
            best_cmap = cmap_after

        # 1.2 call dpr(i+1, j)
        pset, sc, ch, cmap_sub = dpr(i+1, j)
        cmap_mut = {c: set(cmap_sub.get(c, frozenset())) for c in children}
        for c in children:
            if (a,b) in cmap_mut[c]:
                continue
            cmap_view = {k: frozenset(v) for k,v in cmap_mut.items()}
            if laminar_ok_to_impute_for_child(c, (a,b), cmap_view):
                cmap_mut[c].add((a,b))
        cmap_after = {c: frozenset(cmap_mut[c]) for c in children}
        extra_cnt = count_child_intervals_start_eq_i_end_le_j(i,j, cmap_after)
        miss = count_children_missing_interval(a,b, cmap_after)
        c=positions[i+1]
        bcost = boundary_cost_include_start(a, cmap_after) + boundary_cost_include_end(b, cmap_after)+boundary_cost_exclude_end(c, cmap_after)
        total = sc + Ct * extra_cnt + Ct * miss + bcost
        cand_parent = frozenset({(a,b)}) | pset
        if total <= best_score:
            best_score = total
            best_parent_set = cand_parent
            best_choice = ("call", "dpr", i+1, j)
            best_cmap = cmap_after

        # 1.3 splits
        for k in range(i+1, j):
            pk = positions[k]
            if (pk in start_boundaries) and (pk in end_boundaries):
                left_parent, left_score, _, left_cmap = dpl(i, k)
                right_parent, right_score, _, right_cmap = dpr(k, j)
                merged = merge_child_maps(left_cmap, right_cmap)
                cmap_mut = {c: set(merged.get(c, frozenset())) for c in children}
                for c in children:
                    if (a,b) in cmap_mut[c]:
                        continue
                    cmap_view = {kk: frozenset(vv) for kk,vv in cmap_mut.items()}
                    if laminar_ok_to_impute_for_child(c, (a,b), cmap_view):
                        cmap_mut[c].add((a,b))
                cmap_after = {c: frozenset(cmap_mut[c]) for c in children}
                cross_cnt = count_child_intervals_crossing_k(i,k,j, cmap_after)
                miss = count_children_missing_interval(a,b, cmap_after)
                bcost = boundary_cost_include_start(a, cmap_after) + boundary_cost_include_end(b, cmap_after)
                total = left_score + right_score + Ct * cross_cnt + Ct * miss + bcost
                cand_parent = frozenset({(a,b)}) | left_parent | right_parent
                if total <= best_score:
                    best_score = total
                    best_parent_set = cand_parent
                    best_choice = ("split", [("dpl", i, k), ("dpr", k, j)])
                    best_cmap = cmap_after

        # 2. EXCLUDE [a,b] branches (no imputation)
        # 2.1 dp0(i, j-1)
        pset, sc, ch, cmap_sub = dp0(i, j-1)
        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i,j, cmap_sub)
        have = count_children_with_interval(a,b, cmap_sub)
        c=positions[j-1]
        bcost = boundary_cost_exclude_end(b, cmap_sub)+boundary_cost_exclude_start(c, cmap_sub)
        total = sc + Ct * extra_cnt + Ct * have + bcost
        if total <= best_score:
            best_score = total
            best_parent_set = pset
            best_choice = ("call", "dp0", i, j-1)
            best_cmap = cmap_sub

        # 2.2 dp0(i+1, j)
        pset, sc, ch, cmap_sub = dp0(i+1, j)
        extra_cnt = count_child_intervals_start_eq_i_end_le_j(i,j, cmap_sub)
        have = count_children_with_interval(a,b, cmap_sub)
        c=positions[i+1]
        bcost = boundary_cost_exclude_start(a, cmap_sub) + boundary_cost_exclude_end(c, cmap_sub)
        total = sc + Ct * extra_cnt + Ct * have + bcost
        if total <= best_score:
            best_score = total
            best_parent_set = pset
            best_choice = ("call", "dp0", i+1, j)
            best_cmap = cmap_sub

        # 2.3 splits dp0(i,k) + dp0(k,j)
        for k in range(i+1, j):
            pk = positions[k]
            if (pk in start_boundaries) and (pk in end_boundaries):
                left_parent, left_score, _, left_cmap = dp0(i, k)
                right_parent, right_score, _, right_cmap = dp0(k, j)
                merged = merge_child_maps(left_cmap, right_cmap)
                cross_cnt = count_child_intervals_crossing_k(i,k,j, merged)
                have = count_children_with_interval(a,b, merged)
                total = left_score + right_score + Ct * cross_cnt + Ct * have
                cand_parent = left_parent | right_parent
                if total < best_score:
                    best_score = total
                    best_parent_set = cand_parent
                    best_choice = ("split", [("dp0", i, k), ("dp0", k, j)])
                    best_cmap = merged

        dp0_done[i][j] = True
        dp0_val[i][j] = (best_parent_set, best_score, best_choice, best_cmap)
        return dp0_val[i][j]

    def dpl(i, j):
        if i >= j:
            return frozenset(), inf, None, initial_children_map
        if dpl_done[i][j]:
            return dpl_val[i][j]

        a = positions[i]; b = positions[j]
        if a not in start_boundaries or b not in end_boundaries:
            dpl_done[i][j] = True
            dpl_val[i][j] = (frozenset(), inf, None, initial_children_map)
            return dpl_val[i][j]

        if j == i + 1:
            cmap = {c: set(initial_children_map[c]) for c in children}
            for c in children:
                if (a,b) in cmap[c]:
                    continue
                cmap_view = {k: frozenset(v) for k,v in cmap.items()}
                if laminar_ok_to_impute_for_child(c, (a,b), cmap_view):
                    cmap[c].add((a,b))
            cmap_f = {c: frozenset(cmap[c]) for c in children}
            miss = count_children_missing_interval(a,b, cmap_f)
            bcost = boundary_cost_include_end(b, cmap_f)
            include_score = Ct * miss + bcost
            include_parent = frozenset({(a,b)})

            cmap_ex = initial_children_map
            have = count_children_with_interval(a,b, cmap_ex)
            bcost_ex = boundary_cost_exclude_end(b, cmap_ex)
            exclude_score = Ct * have + bcost_ex
            exclude_parent = frozenset()

            if include_score <= exclude_score:
                res = (include_parent, include_score, ("base", "INCLUDE"), cmap_f)
            else:
                res = (exclude_parent, exclude_score, ("base", "EXCLUDE"), cmap_ex)

            dpl_done[i][j] = True
            dpl_val[i][j] = res
            return res

        best_score = inf
        best_parent_set = frozenset()
        best_choice = None
        best_cmap = initial_children_map

        # INCLUDE via dpl(i,j-1)
        pset, sc, ch, cmap_sub = dpl(i, j-1)
        cmap_mut = {c: set(cmap_sub.get(c, frozenset())) for c in children}
        for c in children:
            if (a,b) in cmap_mut[c]:
                continue
            cmap_view = {k: frozenset(v) for k,v in cmap_mut.items()}
            if laminar_ok_to_impute_for_child(c, (a,b), cmap_view):
                cmap_mut[c].add((a,b))
        cmap_after = {c: frozenset(cmap_mut[c]) for c in children}
        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i,j, cmap_after)
        miss = count_children_missing_interval(a,b, cmap_after)
        c = positions[j-1]
        bcost = boundary_cost_include_end(b, cmap_after) + boundary_cost_exclude_start(c, cmap_after)
        total = sc + Ct * extra_cnt + Ct * miss + bcost
        cand_parent = frozenset({(a,b)}) | pset
        if total <= best_score:
            best_score = total
            best_parent_set = cand_parent
            best_choice = ("call", "dpl", i, j-1)
            best_cmap = cmap_after

        # INCLUDE via dpr(i+1,j)
        pset, sc, ch, cmap_sub = dpr(i+1, j)
        cmap_mut = {c: set(cmap_sub.get(c, frozenset())) for c in children}
        for c in children:
            if (a,b) in cmap_mut[c]:
                continue
            cmap_view = {k: frozenset(v) for k,v in cmap_mut.items()}
            if laminar_ok_to_impute_for_child(c, (a,b), cmap_view):
                cmap_mut[c].add((a,b))
        cmap_after = {c: frozenset(cmap_mut[c]) for c in children}
        extra_cnt = count_child_intervals_start_eq_i_end_le_j(i,j, cmap_after)
        miss = count_children_missing_interval(a,b, cmap_after)
        c = positions[i+1]
        bcost = boundary_cost_include_end(b, cmap_after) + boundary_cost_exclude_end(c, cmap_after)
        total = sc + Ct * extra_cnt + Ct * miss + bcost
        cand_parent = frozenset({(a,b)}) | pset
        if total <= best_score:
            best_score = total
            best_parent_set = cand_parent
            best_choice = ("call", "dpr", i+1, j)
            best_cmap = cmap_after

        # splits
        for k in range(i+1, j):
            pk = positions[k]
            if (pk in start_boundaries) and (pk in end_boundaries):
                lp, lsc, _, lcmap = dpl(i,k)
                rp, rsc, _, rcmap = dpr(k,j)
                merged = merge_child_maps(lcmap, rcmap)
                cmap_mut = {c: set(merged.get(c, frozenset())) for c in children}
                for c in children:
                    if (a,b) in cmap_mut[c]:
                        continue
                    cmap_view = {kk: frozenset(vv) for kk,vv in cmap_mut.items()}
                    if laminar_ok_to_impute_for_child(c, (a,b), cmap_view):
                        cmap_mut[c].add((a,b))
                cmap_after = {c: frozenset(cmap_mut[c]) for c in children}
                cross_cnt = count_child_intervals_crossing_k(i,k,j, cmap_after)
                miss = count_children_missing_interval(a,b, cmap_after)
                bcost = boundary_cost_include_end(b, cmap_after)
                total = lsc + rsc + Ct * cross_cnt + Ct * miss + bcost
                cand_parent = frozenset({(a,b)}) | lp | rp
                if total <= best_score:
                    best_score = total
                    best_parent_set = cand_parent
                    best_choice = ("split", [("dpl", i, k), ("dpr", k, j)])
                    best_cmap = cmap_after

        # EXCLUDE branches
        pset, sc, ch, cmap_sub = dpl(i, j-1)
        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i,j, cmap_sub)
        have = count_children_with_interval(a,b, cmap_sub)
        c=positions[j-1]
        bcost = boundary_cost_exclude_end(b, cmap_sub) + boundary_cost_exclude_start(c, cmap_sub)
        total = sc + Ct * extra_cnt + Ct * have + bcost
        if total < best_score:
            best_score = total
            best_parent_set = pset
            best_choice = ("call", "dpl", i, j-1)
            best_cmap = cmap_sub

        pset, sc, ch, cmap_sub = dp0(i+1, j)
        extra_cnt = count_child_intervals_start_eq_i_end_le_j(i,j, cmap_sub)
        have = count_children_with_interval(a,b, cmap_sub)
        c = positions[i+1]
        bcost = boundary_cost_exclude_end(c, cmap_sub)
        total = sc + Ct * extra_cnt + Ct * have + bcost
        if total <= best_score:
            best_score = total
            best_parent_set = pset
            best_choice = ("call", "dp0", i+1, j)
            best_cmap = cmap_sub

        for k in range(i+1, j):
            pk = positions[k]
            if (pk in start_boundaries) and (pk in end_boundaries):
                lp, lsc, _, lcmap = dpl(i,k)
                rp, rsc, _, rcmap = dp0(k,j)
                merged = merge_child_maps(lcmap, rcmap)
                cross_cnt = count_child_intervals_crossing_k(i,k,j, merged)
                have = count_children_with_interval(a,b, merged)
                total = lsc + rsc + Ct * cross_cnt + Ct * have
                cand_parent = lp | rp
                if total <= best_score:
                    best_score = total
                    best_parent_set = cand_parent
                    best_choice = ("split", [("dpl", i, k), ("dp0", k, j)])
                    best_cmap = merged

        dpl_done[i][j] = True
        dpl_val[i][j] = (best_parent_set, best_score, best_choice, best_cmap)
        return dpl_val[i][j]

    def dpr(i, j):
        if i >= j:
            return frozenset(), inf, None, initial_children_map
        if dpr_done[i][j]:
            return dpr_val[i][j]

        a = positions[i]; b = positions[j]
        if a not in start_boundaries or b not in end_boundaries:
            dpr_done[i][j] = True
            dpr_val[i][j] = (frozenset(), inf, None, initial_children_map)
            return dpr_val[i][j]

        if j == i + 1:
            cmap = {c: set(initial_children_map[c]) for c in children}
            for c in children:
                if (a,b) in cmap[c]:
                    continue
                cmap_view = {k: frozenset(v) for k,v in cmap.items()}
                if laminar_ok_to_impute_for_child(c, (a,b), cmap_view):
                    cmap[c].add((a,b))
            cmap_f = {c: frozenset(cmap[c]) for c in children}
            miss = count_children_missing_interval(a,b, cmap_f)
            bcost = boundary_cost_include_start(a, cmap_f)
            include_score = Ct * miss + bcost
            include_parent = frozenset({(a,b)})

            cmap_ex = initial_children_map
            have = count_children_with_interval(a,b, cmap_ex)
            bcost_ex = boundary_cost_exclude_start(a, cmap_ex)
            exclude_score = Ct * have + bcost_ex
            exclude_parent = frozenset()

            if include_score <= exclude_score:
                res = (include_parent, include_score, ("base", "INCLUDE"), cmap_f)
            else:
                res = (exclude_parent, exclude_score, ("base", "EXCLUDE"), cmap_ex)

            dpr_done[i][j] = True
            dpr_val[i][j] = res
            return res

        best_score = inf
        best_parent_set = frozenset()
        best_choice = None
        best_cmap = initial_children_map

        # INCLUDE via dpl(i,j-1)
        pset, sc, ch, cmap_sub = dpl(i, j-1)
        cmap_mut = {c: set(cmap_sub.get(c, frozenset())) for c in children}
        for c in children:
            if (a,b) in cmap_mut[c]:
                continue
            cmap_view = {k: frozenset(v) for k,v in cmap_mut.items()}
            if laminar_ok_to_impute_for_child(c, (a,b), cmap_view):
                cmap_mut[c].add((a,b))
        cmap_after = {c: frozenset(cmap_mut[c]) for c in children}
        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i,j, cmap_after)
        miss = count_children_missing_interval(a,b, cmap_after)
        c=positions[j-1]
        bcost = boundary_cost_include_start(a, cmap_after)  + boundary_cost_exclude_start(c, cmap_after)
        total = sc + Ct * extra_cnt + Ct * miss + bcost
        cand_parent = frozenset({(a,b)}) | pset
        if total <= best_score:
            best_score = total
            best_parent_set = cand_parent
            best_choice = ("call", "dpl", i, j-1)
            best_cmap = cmap_after

        # INCLUDE via dpr(i+1,j)
        pset, sc, ch, cmap_sub = dpr(i+1, j)
        cmap_mut = {c: set(cmap_sub.get(c, frozenset())) for c in children}
        for c in children:
            if (a,b) in cmap_mut[c]:
                continue
            cmap_view = {k: frozenset(v) for k,v in cmap_mut.items()}
            if laminar_ok_to_impute_for_child(c, (a,b), cmap_view):
                cmap_mut[c].add((a,b))
        cmap_after = {c: frozenset(cmap_mut[c]) for c in children}
        extra_cnt = count_child_intervals_start_eq_i_end_le_j(i,j, cmap_after)
        miss = count_children_missing_interval(a,b, cmap_after)
        c = positions[i+1]
        bcost = boundary_cost_include_start(a, cmap_after) + boundary_cost_exclude_end(c, cmap_after)
        total = sc + Ct * extra_cnt + Ct * miss + bcost
        cand_parent = frozenset({(a,b)}) | pset
        if total <= best_score:
            best_score = total
            best_parent_set = cand_parent
            best_choice = ("call", "dpr", i+1, j)
            best_cmap = cmap_after

        # splits
        for k in range(i+1, j):
            pk = positions[k]
            if (pk in start_boundaries) and (pk in end_boundaries):
                lp, lsc, _, lcmap = dpl(i,k)
                rp, rsc, _, rcmap = dpr(k,j)
                merged = merge_child_maps(lcmap, rcmap)
                cmap_mut = {c: set(merged.get(c, frozenset())) for c in children}
                for c in children:
                    if (a,b) in cmap_mut[c]:
                        continue
                    cmap_view = {kk: frozenset(vv) for kk,vv in cmap_mut.items()}
                    if laminar_ok_to_impute_for_child(c, (a,b), cmap_view):
                        cmap_mut[c].add((a,b))
                cmap_after = {c: frozenset(cmap_mut[c]) for c in children}
                cross_cnt = count_child_intervals_crossing_k(i,k,j, cmap_after)
                miss = count_children_missing_interval(a,b, cmap_after)
                bcost = boundary_cost_include_start(a, cmap_after)
                total = lsc + rsc + Ct * cross_cnt + Ct * miss + bcost
                cand_parent = frozenset({(a,b)}) | lp | rp
                if total <= best_score:
                    best_score = total
                    best_parent_set = cand_parent
                    best_choice = ("split", [("dpl", i, k), ("dpr", k, j)])
                    best_cmap = cmap_after

        # EXCLUDE branches
        pset, sc, ch, cmap_sub = dp0(i, j-1)
        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i,j, cmap_sub)
        have = count_children_with_interval(a,b, cmap_sub)
        c = positions[j-1]
        bcost = boundary_cost_exclude_start(c, cmap_sub)
        total = sc + Ct * extra_cnt + Ct * have + bcost
        if total <= best_score:
            best_score = total
            best_parent_set = pset
            best_choice = ("call", "dp0", i, j-1)
            best_cmap = cmap_sub

        pset, sc, ch, cmap_sub = dp0(i+1, j)
        extra_cnt = count_child_intervals_start_eq_i_end_le_j(i,j, cmap_sub)
        have = count_children_with_interval(a,b, cmap_sub)
        c = positions[i+1]
        bcost = boundary_cost_exclude_start(a, cmap_sub)+boundary_cost_exclude_end(c, cmap_sub)
        total = sc + Ct * extra_cnt + Ct * have + bcost
        if total <= best_score:
            best_score = total
            best_parent_set = pset
            best_choice = ("call", "dp0", i+1, j)
            best_cmap = cmap_sub

        for k in range(i+1, j):
            pk = positions[k]
            if (pk in start_boundaries) and (pk in end_boundaries):
                lp, lsc, _, lcmap = dp0(i,k)
                rp, rsc, _, rcmap = dpr(k,j)
                merged = merge_child_maps(lcmap, rcmap)
                cross_cnt = count_child_intervals_crossing_k(i,k,j, merged)
                have = count_children_with_interval(a,b, merged)
                total = lsc + rsc + Ct * cross_cnt + Ct * have
                cand_parent = lp | rp
                if total <= best_score:
                    best_score = total
                    best_parent_set = cand_parent
                    best_choice = ("split", [("dp0", i, k), ("dpr", k, j)])
                    best_cmap = merged

        dpr_done[i][j] = True
        dpr_val[i][j] = (best_parent_set, best_score, best_choice, best_cmap)
        return dpr_val[i][j]

    # -------------------------
    # Traceback using structured choices (no string parsing)
    # -------------------------
    def traceback(state, i, j, path):
        if i >= j:
            return
        if state == "dp0":
            if not dp0_done[i][j]:
                return
            parent_set, score, choice, cmap = dp0_val[i][j]
        elif state == "dpl":
            if not dpl_done[i][j]:
                return
            parent_set, score, choice, cmap = dpl_val[i][j]
        elif state == "dpr":
            if not dpr_done[i][j]:
                return
            parent_set, score, choice, cmap = dpr_val[i][j]
        else:
            return

        a = positions[i]; b = positions[j]
        path.append({
            "state": f"{state}[{i},{j}] ({a},{b})",
            "choice": choice,
            "intervals": set(parent_set),
            "score": score,
            "children_map": cmap
        })

        if not choice:
            return

        typ = choice[0]
        if typ == "call":
            _, func_name, ni, nj = choice
            traceback(func_name, ni, nj, path)
        elif typ == "split":
            _, list_calls = choice
            for func_name, ni, nj in list_calls:
                traceback(func_name, ni, nj, path)
        elif typ == "base":
            # no recursive calls for base
            return
        else:
            # unknown choice type -> safe stop
            return

    # -------------------------
    # Run root
    # -------------------------
    root_parent_set, root_score, root_choice, root_children_map = dp0(0, n-1)

    if trace:
        optimal_path = []
        traceback("dp0", 0, n-1, optimal_path)
        return set(root_parent_set), root_score, root_children_map, optimal_path

    return set(root_parent_set), root_score, root_children_map

# -------------------------
# Testing Framework
# -------------------------

def generate_star_topology(num_children):
    """
    Generate a star topology with root 0 and num_children leaf children.
    Returns: edges, leaf_nodes
    """
    edges = [(0, i) for i in range(1, num_children + 1)]
    leaf_nodes = list(range(1, num_children + 1))
    return edges, leaf_nodes


def extract_boundaries(leaf_intervals):
    """
    Extract all start and end boundaries from leaf intervals.
    """
    import re
    
    start_boundaries = set()
    end_boundaries = set()
    
    for node_intervals in leaf_intervals.values():
        for interval_str in node_intervals:
            a, b = map(int, re.findall(r"\d+", interval_str))
            start_boundaries.add(a)
            start_boundaries.add(b)
            end_boundaries.add(b)
            end_boundaries.add(a)
    
    return sorted(start_boundaries), sorted(end_boundaries)


def calculate_parsimony_score(root_intervals, edges, leaf_intervals, Ct=1.0, Cb=1.0):
    """Full parsimony score from original child intervals."""
    def parse_interval(s):
        a, b = map(int, re.findall(r"\d+", s))
        return (a, b)
    
    children = [v for (u, v) in edges if u == 0]
    child_intervals = {}
    for c in children:
        child_intervals[c] = set(parse_interval(x) for x in leaf_intervals.get(c, []))
    
    total_score = 0.0
    root_starts = {a for (a, b) in root_intervals}
    root_ends = {b for (a, b) in root_intervals}
    
    for c in children:
        child_set = child_intervals[c]
        child_starts = {a for (a, b) in child_set}
        child_ends = {b for (a, b) in child_set}
        total_score += Ct * (len(root_intervals - child_set) + len(child_set - root_intervals))
        total_score += Cb * (len(root_starts - child_starts) + len(child_starts - root_starts))
        total_score += Cb * (len(root_ends - child_ends) + len(child_ends - root_ends))
    
    return total_score


def calculate_parsimony_score_from_sets(root_intervals, children_intervals, Ct=1.0, Cb=1.0):
    """Parsimony score using explicit child interval sets."""
    total_score = 0.0
    root_starts = {a for (a, b) in root_intervals}
    root_ends = {b for (a, b) in root_intervals}

    for child_set in children_intervals.values():
        child_starts = {a for (a, b) in child_set}
        child_ends = {b for (a, b) in child_set}
        total_score += Ct * (len(root_intervals - child_set) + len(child_set - root_intervals))
        total_score += Cb * (len(root_starts - child_starts) + len(child_starts - root_starts))
        total_score += Cb * (len(root_ends - child_ends) + len(child_ends - root_ends))

    return total_score


def calculate_tad_only_score(root_intervals, children_intervals, Ct=1.0):
    """TAD-only score (no boundary RF)."""
    total_score = 0.0
    for child_set in children_intervals.values():
        total_score += Ct * (len(root_intervals - child_set) + len(child_set - root_intervals))
    return total_score


def greedy_imputation_tad_only(edges, leaf_intervals, start_boundaries, end_boundaries, Ct=1.0, Cb=1.0):
    """Greedy imputation: iterates over candidate intervals, imputes greedily based on Ct only."""
    def parse_interval(s):
        a, b = map(int, re.findall(r"\d+", s))
        return (a, b)

    def is_laminar(interval, interval_set):
        a, b = interval
        for (x, y) in interval_set:
            if (x, y) == (a, b):
                continue
            if (x < a < y < b) or (a < x < b < y):
                return False
        return True

    children = [v for (u, v) in edges if u == 0]
    original_child_intervals = {
        c: set(parse_interval(x) for x in leaf_intervals.get(c, [])) for c in children
    }
    working_child_intervals = {c: set(v) for c, v in original_child_intervals.items()}

    parent_intervals = set()
    boundaries = sorted(set(start_boundaries) | set(end_boundaries))

    for i in range(len(boundaries)):
        a = boundaries[i]
        if a not in start_boundaries:
            continue
        for j in range(i + 1, len(boundaries)):
            b = boundaries[j]
            if b not in end_boundaries:
                continue

            interval = (a, b)
            impute_children = []
            for c in children:
                if interval in working_child_intervals[c]:
                    continue
                if is_laminar(interval, working_child_intervals[c]):
                    impute_children.append(c)

            missing_after = 0
            for c in children:
                if interval in working_child_intervals[c] or c in impute_children:
                    continue
                missing_after += 1

            include_cost = Ct * missing_after
            exclude_cost = Ct * sum(
                1 for c in children if interval in original_child_intervals[c]
            )

            if include_cost <= exclude_cost:
                parent_intervals.add(interval)
                for c in impute_children:
                    working_child_intervals[c].add(interval)

    greedy_score = calculate_parsimony_score_from_sets(
        parent_intervals, working_child_intervals, Ct=Ct, Cb=Cb
    )
    return parent_intervals, greedy_score, working_child_intervals


def brute_force_imputation(edges, leaf_intervals, start_boundaries, end_boundaries, Ct=1.0, Cb=1.0):
    """
    Brute force solution that tries all possible root intervals and all valid leaf imputations.
    
    Strategy:
    1. Generate all possible candidate intervals from start_boundaries x end_boundaries
    2. For each candidate root interval subset:
       a. For each child, enumerate all valid ways to impute intervals (respecting laminar property)
       b. Calculate parsimony score
    3. Return the optimal root intervals and leaf imputations
    
    This is exponential but gives the true optimal solution for small instances.
    """
    def parse_interval(s):
        a, b = map(int, re.findall(r"\d+", s))
        return (a, b)

    def is_laminar(interval, interval_set):
        """Check if interval is laminar with all intervals in interval_set."""
        a, b = interval
        for (x, y) in interval_set:
            if (x, y) == (a, b):
                continue
            # Crossing patterns violate laminarity
            if (x < a < y < b) or (a < x < b < y):
                return False
        return True

    def get_valid_imputations_for_child(c, original_child_set, candidate_intervals):
        """
        Generate all valid ways to impute intervals in child c.
        A valid imputation is a superset of original_child_set where all added intervals
        are laminar with the existing ones and form a laminar family.
        
        This returns a list of frozensets, each representing a valid child interval set.
        """
        def backtrack(idx, current_set):
            if idx == len(candidate_intervals):
                results.append(frozenset(current_set))
                return
            
            interval = candidate_intervals[idx]
            # Option 1: Don't add this interval
            backtrack(idx + 1, current_set)
            
            # Option 2: Try to add this interval (if laminar and not already present)
            if interval not in current_set and is_laminar(interval, current_set):
                current_set.add(interval)
                backtrack(idx + 1, current_set)
                current_set.remove(interval)
        
        results = []
        backtrack(0, set(original_child_set))
        return results

    children = [v for (u, v) in edges if u == 0]
    original_child_intervals = {
        c: set(parse_interval(x) for x in leaf_intervals.get(c, [])) for c in children
    }
    
    # Generate all possible candidate intervals
    valid_candidates = []
    for i, a in enumerate(start_boundaries):
        for j, b in enumerate(end_boundaries):
            if a < b:
                valid_candidates.append((a, b))
    
    best_score = float("inf")
    best_root_intervals = frozenset()
    best_children_intervals = {}
    
    # Try all possible root interval subsets
    n_candidates = len(valid_candidates)
    for root_mask in range(1 << n_candidates):
        root_intervals = frozenset(
            valid_candidates[i] for i in range(n_candidates) if (root_mask >> i) & 1
        )
        
        # For this root, enumerate all valid child imputations
        # Generate all valid imputation combinations for all children
        child_imputation_lists = {}
        for c in children:
            child_imputation_lists[c] = get_valid_imputations_for_child(
                c, original_child_intervals[c], valid_candidates
            )
        
        # Iterate over all combinations of child imputations
        def iterate_child_imputations(child_idx, current_children):
            nonlocal best_score, best_root_intervals, best_children_intervals
            
            if child_idx == len(children):
                # Calculate score for this configuration
                score = calculate_parsimony_score_from_sets(root_intervals, current_children, Ct=Ct, Cb=Cb)
                if score < best_score:
                    best_score = score
                    best_root_intervals = root_intervals
                    best_children_intervals = dict(current_children)
                return
            
            c = children[child_idx]
            for child_set in child_imputation_lists[c]:
                current_children[c] = child_set
                iterate_child_imputations(child_idx + 1, current_children)
        
        iterate_child_imputations(0, {})
    
    return best_root_intervals, best_score, best_children_intervals


def test_dp_vs_greedy(num_tests=10, min_children=2, max_children=5,
                      min_val=1, max_val=20, num_intervals=10,
                      max_per_leaf=5, Ct=1.0, Cb=1.0, verbose=True):
    """
    Compare DP vs greedy imputation on random star topologies.
    Greedy include/exclude uses only Ct, final score includes Cb.
    """
    print(f"\n{'='*70}")
    print("Testing DP vs Greedy (TAD-only) on Star Topology Trees")
    print(f"{'='*70}")
    print(f"Number of tests: {num_tests}")
    print(f"Children per tree: {min_children}-{max_children}")
    print(f"Interval range: [{min_val}, {max_val}]")
    print(f"Intervals pool size: {num_intervals}")
    print(f"Max intervals per leaf: {max_per_leaf}")
    print(f"Costs: Ct={Ct}, Cb={Cb}\n")

    greedy_wins = 0
    dp_wins = 0
    ties = 0

    for test_num in range(1, num_tests + 1):
        num_children = random.randint(min_children, max_children)
        edges, leaf_nodes = generate_star_topology(num_children)

        available_intervals = generate_random_intervals(min_val, max_val, num_intervals)
        leaf_intervals = generate_random_laminar_intervals_from_list(
            leaf_nodes, available_intervals, max_per_leaf=max_per_leaf
        )

        start_boundaries, end_boundaries = extract_boundaries(leaf_intervals)

        if verbose:
            print(f"\nTest {test_num}:")
            print(f"  Tree: Star topology with {num_children} children")
            print(f"  Edges: {edges}")
            print("  Leaf intervals:")
            for leaf in sorted(leaf_nodes):
                print(f"    Node {leaf}: {sorted(leaf_intervals[leaf])}")

        dp_intervals, dp_score, dp_children_map = infer_tads_with_dp(
            edges, leaf_intervals, start_boundaries, end_boundaries,
            Ct=Ct, Cb=Cb, trace=False
        )



        greedy_intervals, greedy_score, greedy_children = greedy_imputation_tad_only(
            edges, leaf_intervals, start_boundaries, end_boundaries, Ct=Ct, Cb=Cb
        )

        score_diff = dp_score - greedy_score
        if abs(score_diff) < 1e-9:
            ties += 1
            outcome = "TIE"
        elif score_diff < 0:
            dp_wins += 1
            outcome = "DP better"
        else:
            greedy_wins += 1
            outcome = "Greedy better"


        if verbose:
            print(f"  DP score:     {dp_score:.6f}")
            print(f"  Greedy score: {greedy_score:.6f}")
            print(f"  Outcome: {outcome}")
            print("  DP assignments:")
            print(f"    Parent: {sorted(dp_intervals)}")
            for child in sorted(dp_children_map):
                print(f"    Child {child}: {sorted(dp_children_map[child])}")
            print("  Greedy assignments:")
            print(f"    Parent: {sorted(greedy_intervals)}")
            for child in sorted(greedy_children):
                print(f"    Child {child}: {sorted(greedy_children[child])}")
            if outcome == "Greedy better":
                print("  DP was suboptimal compared to Greedy.")
                break

    print(f"\n{'='*70}")
    print("Results Summary")
    print(f"{'='*70}")
    print(f"Total tests: {num_tests}")
    print(f"DP better:   {dp_wins}")
    print(f"Greedy better: {greedy_wins}")
    print(f"Ties:        {ties}")
    print(f"{'='*70}\n")


def test_brute_vs_dp_vs_greedy(num_tests=10, min_children=2, max_children=3,
                               min_val=1, max_val=10, num_intervals=8,
                               max_per_leaf=2, Ct=1.0, Cb=1.0, verbose=True):
    """
    Compare Brute Force, DP, and Greedy imputation on random star topologies.
    Brute force gives the true optimal, so we can verify DP and greedy correctness.
    """
    print(f"\n{'='*100}")
    print("Testing Brute Force vs DP vs Greedy on Star Topology Trees")
    print(f"{'='*100}")
    print(f"Number of tests: {num_tests}")
    print(f"Children per tree: {min_children}-{max_children}")
    print(f"Interval range: [{min_val}, {max_val}]")
    print(f"Intervals pool size: {num_intervals}")
    print(f"Max intervals per leaf: {max_per_leaf}")
    print(f"Costs: Ct={Ct}, Cb={Cb}\n")

    bf_wins = 0
    dp_wins = 0
    greedy_wins = 0
    dp_ties_bf = 0
    greedy_ties_bf = 0
    all_tie = 0
    dp_suboptimal = 0
    greedy_suboptimal = 0

    for test_num in range(1, num_tests + 1):
        num_children = random.randint(min_children, max_children)
        edges, leaf_nodes = generate_star_topology(num_children)

        available_intervals = generate_random_intervals(min_val, max_val, num_intervals)
        leaf_intervals = generate_random_laminar_intervals_from_list(
            leaf_nodes, available_intervals, max_per_leaf=max_per_leaf
        )

        start_boundaries, end_boundaries = extract_boundaries(leaf_intervals)

        if verbose:
            print(f"\n{'='*100}")
            print(f"Test {test_num}:")
            print(f"  Tree: Star topology with {num_children} children")
            print(f"  Edges: {edges}")
            print("  Leaf intervals:")
            for leaf in sorted(leaf_nodes):
                print(f"    Node {leaf}: {sorted(leaf_intervals[leaf])}")
            print()

        try:
            # Brute force (true optimal)
            bf_intervals, bf_score, bf_children = brute_force_imputation(
                edges, leaf_intervals, start_boundaries, end_boundaries,
                Ct=Ct, Cb=Cb
            )

            # DP solution
            dp_intervals, dp_score, dp_children_map = infer_tads_with_dp(
                edges, leaf_intervals, start_boundaries, end_boundaries,
                Ct=Ct, Cb=Cb, trace=False
            )

            # Greedy solution
            greedy_intervals, greedy_score, greedy_children = greedy_imputation_tad_only(
                edges, leaf_intervals, start_boundaries, end_boundaries, Ct=Ct, Cb=Cb
            )

            if verbose:
                print(f"  Brute Force score: {bf_score:.6f}")
                print(f"    Root: {sorted(bf_intervals)}")
                for child in sorted(bf_children):
                    print(f"    Child {child}: {sorted(bf_children[child])}")
                print()
                print(f"  DP score:          {dp_score:.6f}")
                print(f"    Root: {sorted(dp_intervals)}")
                for child in sorted(dp_children_map):
                    print(f"    Child {child}: {sorted(dp_children_map[child])}")
                print()
                print(f"  Greedy score:      {greedy_score:.6f}")
                print(f"    Root: {sorted(greedy_intervals)}")
                for child in sorted(greedy_children):
                    print(f"    Child {child}: {sorted(greedy_children[child])}")
                print()

            # Compare scores
            dp_diff = dp_score - bf_score
            greedy_diff = greedy_score - bf_score

            if abs(dp_diff) < 1e-9:
                dp_ties_bf += 1
                dp_outcome = "OPTIMAL (tie with BF)"
            elif dp_diff > 1e-9:
                dp_suboptimal += 1
                dp_outcome = f"SUBOPTIMAL (+{dp_diff:.6f})"
            else:
                dp_wins += 1
                dp_outcome = "BETTER than BF?! (error)"

            if abs(greedy_diff) < 1e-9:
                greedy_ties_bf += 1
                greedy_outcome = "OPTIMAL (tie with BF)"
            elif greedy_diff > 1e-9:
                greedy_suboptimal += 1
                greedy_outcome = f"SUBOPTIMAL (+{greedy_diff:.6f})"
            else:
                greedy_wins += 1
                greedy_outcome = "BETTER than BF?! (error)"

            if abs(dp_diff) < 1e-9 and abs(greedy_diff) < 1e-9:
                all_tie += 1

            if verbose:
                print(f"  Outcomes:")
                print(f"    DP vs BF:     {dp_outcome}")
                print(f"    Greedy vs BF: {greedy_outcome}")
                if dp_suboptimal > 0 or greedy_suboptimal > 0:
                    print(f"  ⚠️  SUBOPTIMAL SOLUTION FOUND!")

        except Exception as e:
            if verbose:
                print(f"  Error during test: {e}")
            continue

    print(f"\n{'='*100}")
    print("Results Summary")
    print(f"{'='*100}")
    print(f"Total tests: {num_tests}")
    print(f"\nDP vs Brute Force:")
    print(f"  DP Optimal (ties with BF): {dp_ties_bf}")
    print(f"  DP Suboptimal:             {dp_suboptimal}")
    print(f"  DP Better (impossible):    {dp_wins}")
    print(f"\nGreedy vs Brute Force:")
    print(f"  Greedy Optimal (ties with BF): {greedy_ties_bf}")
    print(f"  Greedy Suboptimal:             {greedy_suboptimal}")
    print(f"  Greedy Better (impossible):    {greedy_wins}")
    print(f"\nAll three tie: {all_tie}")
    print(f"{'='*100}\n")
def test_ilp_vs_dp_vs_greedy(num_tests=10, min_children=2, max_children=3,
                               min_val=1, max_val=10, num_intervals=8,
                               max_per_leaf=2, Ct=1.0, Cb=1.0, verbose=True, impute=False):
    """
    Compare ILP (if implemented), DP, and Greedy imputation on random star topologies.
    ILP gives the true optimal, so we can verify DP and greedy correctness.
    """
    print(f"\n{'='*100}")
    print("Testing ILP vs DP vs Greedy on Star Topology Trees")
    print(f"{'='*100}")
    print(f"Number of tests: {num_tests}")
    print(f"Children per tree: {min_children}-{max_children}")
    print(f"Interval range: [{min_val}, {max_val}]")
    print(f"Intervals pool size: {num_intervals}")
    print(f"Max intervals per leaf: {max_per_leaf}")
    print(f"Costs: Ct={Ct}, Cb={Cb}\n")

    ilp_wins = 0
    dp_wins = 0
    greedy_wins = 0
    dp_ties_ilp = 0
    greedy_ties_ilp = 0
    all_tie = 0
    dp_suboptimal = 0
    greedy_suboptimal = 0

    for test_num in range(1, num_tests + 1):
        num_children = random.randint(min_children, max_children)
        edges, leaf_nodes = generate_star_topology(num_children)

        available_intervals = generate_random_intervals(min_val, max_val, num_intervals)
        leaf_intervals = generate_random_laminar_intervals_from_list(
            leaf_nodes, available_intervals, max_per_leaf=max_per_leaf
        )

        start_boundaries, end_boundaries = extract_boundaries(leaf_intervals)

        if verbose:
            print(f"\n{'='*100}")
            print(f"Test {test_num}:")
            print(f"  Tree: Star topology with {num_children} children")
            print(f"  Edges: {edges}")
            print("  Leaf intervals:")
            for leaf in sorted(leaf_nodes):
                print(f"    Node {leaf}: {sorted(leaf_intervals[leaf])}")
            print()

        try:
            # ILP (true optimal)
            lp_cost, is_integral, selected_assignment, x, nodes, intervals = solve_lp_from_edges_and_leaves(
                edges, leaf_intervals, Ct=Ct, Cb=Cb, mode="convex", impute=impute
            )

            # DP solution
            dp_intervals, dp_score, dp_children_map = infer_tads_with_dp(
                edges, leaf_intervals, start_boundaries, end_boundaries,
                Ct=Ct, Cb=Cb, trace=False
            )

            # Greedy solution
            greedy_intervals, greedy_score, greedy_children = greedy_imputation_tad_only(
                edges, leaf_intervals, start_boundaries, end_boundaries, Ct=Ct, Cb=Cb
            )

            if verbose:
                print(f"  ILP score:         {lp_cost:.6f}")
                print(f"    Root: {sorted(selected_assignment)}")
                for child in sorted(nodes):
                    if child in selected_assignment:
                        print(f"    Child {child}: {sorted(selected_assignment[child])}")
                print()
                print(f"  DP score:          {dp_score:.6f}")
                print(f"    Root: {sorted(dp_intervals)}")
                for child in sorted(dp_children_map):
                    print(f"    Child {child}: {sorted(dp_children_map[child])}")
                print()
                print(f"  Greedy score:      {greedy_score:.6f}")
                print(f"    Root: {sorted(greedy_intervals)}")
                for child in sorted(greedy_children):
                    print(f"    Child {child}: {sorted(greedy_children[child])}")
                print()

            # Compare scores
            dp_diff = dp_score - lp_cost
            greedy_diff = greedy_score - lp_cost

            if abs(dp_diff) < 1e-9:
                dp_ties_ilp += 1
                dp_outcome = "OPTIMAL (tie with ILP)"
            elif dp_diff > 1e-9:
                dp_suboptimal += 1
                dp_outcome = f"SUBOPTIMAL (+{dp_diff:.6f})"
            else:
                dp_wins += 1
                dp_outcome = "BETTER than ILP?! (error)"

            if abs(greedy_diff) < 1e-9:
                greedy_ties_ilp += 1
                greedy_outcome = "OPTIMAL (tie with ILP)"
            elif greedy_diff > 1e-9:
                greedy_suboptimal += 1
                greedy_outcome = f"SUBOPTIMAL (+{greedy_diff:.6f})"
            else:
                greedy_wins += 1
                greedy_outcome = "BETTER than ILP?! (error)"
                break

            if abs(dp_diff) < 1e-9 and abs(greedy_diff) < 1e-9:
                all_tie += 1

            if verbose:
                print(f"  Outcomes:")
                print(f"    DP vs ILP:     {dp_outcome}")
                print(f"    Greedy vs ILP: {greedy_outcome}")
                if dp_suboptimal > 0 or greedy_suboptimal > 0:
                    print(f"  ⚠️  SUBOPTIMAL SOLUTION FOUND!")
                    break

        except Exception as e:
            if verbose:
                print(f"  Error during test: {e}")
            continue

    print(f"\n{'='*100}")
    print("Results Summary")
    print(f"{'='*100}")
    print(f"Total tests: {num_tests}")
    print(f"\nDP vs ILP:")
    print(f"  DP Optimal (ties with ILP): {dp_ties_ilp}")
    print(f"  DP Suboptimal:             {dp_suboptimal}")
    print(f"  DP Better (impossible):    {dp_wins}")
    print(f"\nGreedy vs ILP:")
    print(f"  Greedy Optimal (ties with ILP): {greedy_ties_ilp}")
    print(f"  Greedy Suboptimal:             {greedy_suboptimal}")
    print(f"  Greedy Better (impossible):    {greedy_wins}")
    print(f"\nAll three tie: {all_tie}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    # Test with small instances where brute force is feasible
    # test_ilp_vs_dp_vs_greedy(
    #     num_tests=10000, min_children=2, max_children=4,
    #     min_val=1, max_val=10, num_intervals=100,
    #     max_per_leaf=5, Ct=1.0, Cb=1.0, verbose=True, impute=True
    # )
    
    # Uncomment to run large-scale DP vs Greedy test (no brute force)

    # test_dp_vs_greedy(
    #     num_tests=1000000, min_children=2, max_children=4,
    #     min_val=1, max_val=10, num_intervals=30,
    #     max_per_leaf=4, Ct=1.0, Cb=1.0, verbose=False
    # )
    edges = [(0, 1), (0, 2),(0,3)]
    leaf_intervals = { 
        1: ['[3,6]'],
        2: ['[1,5]', '[2,5]'],
        3:['[2,4]', '[2,5]']

    }
    start_boundaries, end_boundaries = extract_boundaries(leaf_intervals)
    
    dp_intervals, dp_score, dp_children_map = infer_tads_with_dp(
        edges, leaf_intervals, start_boundaries, end_boundaries,
        Ct=1.0, Cb=1.0, trace=False
    )
    
    print(f"DP intervals: {sorted(dp_intervals)} with score {dp_score:.6f}")
    for child in sorted(dp_children_map):
        print(f"Child {child} intervals: {sorted(dp_children_map[child])}")

    # print("\nTrace Log:")
    # for entry in trace_log:
    #     print(f"State: {entry['state']}, Choice: {entry['choice']}, "
    #           f"Intervals: {sorted(entry['intervals'])}, Score: {entry['score']:.6f}")
    #     for child in sorted(entry['children_map']):
    #         print(f"  Child {child} intervals: {sorted(entry['children_map'][child])}")
    #     print("-" * 50)


    #  check greedy
    greedy_intervals, greedy_score, greedy_children = greedy_imputation_tad_only(
        edges, leaf_intervals, start_boundaries, end_boundaries, Ct=1.0, Cb=1.0
    )
    print(f"Greedy intervals: {sorted(greedy_intervals)} with score {greedy_score:.6f}")
    for child in sorted(greedy_children):
        print(f"Child {child} intervals: {sorted(greedy_children[child])}")