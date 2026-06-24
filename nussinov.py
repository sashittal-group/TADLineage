import re
import random
from random_simulation import (
        generate_random_intervals,
        generate_random_laminar_intervals_from_list
    )
from check_lp_constrained import solve_lp_from_edges_and_leaves

def infer_tads_with_dp(edges,
                       leaf_intervals,
                       start_boundaries,
                       end_boundaries,
                       Ct=1.0,
                       Cb=1.0,
                       trace=False):

    # -------------------------
    # Parse helpers
    # -------------------------
    def parse_interval(s):
        a, b = map(int, re.findall(r"\d+", s))
        return (a, b)

    children = [v for (u, v) in edges if u == 0]

    child_intervals = {}
    for c in children:
        child_intervals[c] = set(parse_interval(x)
                                 for x in leaf_intervals.get(c, []))

    positions = sorted(set(start_boundaries) | set(end_boundaries))
    n = len(positions)
    if n < 2:
        return set(), 0.0

    pos_index = {v: i for i, v in enumerate(positions)}
    pos_set = set(positions)
    num_children = len(children)

    start_boundary_counts = {}
    end_boundary_counts = {}
    count_matrix = [[0] * n for _ in range(n)]
    for c in children:
        child_starts = set()
        child_ends = set()
        for (x, y) in child_intervals[c]:
            child_starts.add(x)
            child_ends.add(y)
            if x in pos_set and y in pos_set:
                count_matrix[pos_index[x]][pos_index[y]] += 1
        for x in child_starts:
            start_boundary_counts[x] = start_boundary_counts.get(x, 0) + 1
        for y in child_ends:
            end_boundary_counts[y] = end_boundary_counts.get(y, 0) + 1

    prefix = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        row_sum = 0
        for j in range(n):
            row_sum += count_matrix[i][j]
            prefix[i + 1][j + 1] = prefix[i][j + 1] + row_sum

    # Trace storage (only solution path)
    trace_requested = trace
    trace = False
    trace_log = [] if trace_requested else None

    # -------------------------
    # Small helpers for counting child features
    # -------------------------
    def rect_sum(x1, x2, y1, y2):
        if x1 > x2 or y1 > y2:
            return 0
        return (
            prefix[x2 + 1][y2 + 1]
            - prefix[x1][y2 + 1]
            - prefix[x2 + 1][y1]
            + prefix[x1][y1]
        )

    def count_child_intervals_start_ge_i_end_eq_j(i, j):
        # intervals with x >= pi and y == pj (excluding (pi,pj))
        a = positions[i]
        b = positions[j]
        cnt = rect_sum(i, n - 1, j, j)
        cnt -= count_matrix[i][j]
        return cnt

    def count_child_intervals_start_eq_i_end_le_j(i, j):
        # intervals with x == pi and y <= pj (excluding (pi,pj))
        a = positions[i]
        b = positions[j]
        cnt = rect_sum(i, i, 0, j)
        cnt -= count_matrix[i][j]
        return cnt

    def count_child_intervals_crossing_k(i, k, j):
        # intervals with start >= pi and < pk and end > pk and <= pj (exclude (pi,pj))
        a = positions[i]
        pk = positions[k]
        b = positions[j]
        cnt = rect_sum(i, k - 1, k + 1, j)
        cnt -= count_matrix[i][j]
        return cnt

    def count_children_missing_interval(a,b):
        # number of children that DO NOT have (a,b)
        if a not in pos_set or b not in pos_set:
            return num_children
        have = count_matrix[pos_index[a]][pos_index[b]]
        return num_children - have

    def count_children_with_interval(a, b):
        if a not in pos_set or b not in pos_set:
            return 0
        return count_matrix[pos_index[a]][pos_index[b]]

    def boundary_cost_include_start(a):

        missing = num_children - start_boundary_counts.get(a, 0)
        return Cb * missing

    def boundary_cost_include_end(b):

        missing = num_children - end_boundary_counts.get(b, 0)
        return Cb * missing

    def boundary_cost_exclude_start(a):
    # parent does NOT have start boundary a
        have = start_boundary_counts.get(a, 0)
        return Cb * have

    def boundary_cost_exclude_end(b):
    # parent does NOT have end boundary b
        have = end_boundary_counts.get(b, 0)
        return Cb * have

    # -------------------------
    # DP tables: dp0, dpl, dpr
    # Each returns (frozenset_of_intervals, score, choice_string)
    # -------------------------
    inf = float("inf")
    dp0_score = [[inf] * n for _ in range(n)]
    dpl_score = [[inf] * n for _ in range(n)]
    dpr_score = [[inf] * n for _ in range(n)]
    dp0_set = [[None] * n for _ in range(n)]
    dpl_set = [[None] * n for _ in range(n)]
    dpr_set = [[None] * n for _ in range(n)]
    dp0_choice = [[None] * n for _ in range(n)]
    dpl_choice = [[None] * n for _ in range(n)]
    dpr_choice = [[None] * n for _ in range(n)]
    dp0_done = [[False] * n for _ in range(n)]
    dpl_done = [[False] * n for _ in range(n)]
    dpr_done = [[False] * n for _ in range(n)]

    def dp0(i, j):
        # dp0: neither boundary pi nor pj are "already created" outside;
        # they will be charged here if created/used.
        if i >= j:
            return (frozenset(), inf, None)
        if dp0_done[i][j]:
            return (dp0_set[i][j], dp0_score[i][j], dp0_choice[i][j])

        a = positions[i]
        b = positions[j]

        # endpoints must be valid candidate endpoints
        if a not in start_boundaries or b not in end_boundaries:
            dp0_done[i][j] = True
            dp0_set[i][j] = frozenset()
            dp0_score[i][j] = inf
            dp0_choice[i][j] = None
            return (dp0_set[i][j], dp0_score[i][j], dp0_choice[i][j])

        # base case: adjacent
        if j == i + 1:
            # include [a,b]
            include_set = frozenset({(a,b)})
            # interval penalties: Ct per child not having (a,b)
            miss = count_children_missing_interval(a,b)
            # start & end boundary penalties: both considered in dp0
            bcost = boundary_cost_include_start(a) + boundary_cost_include_end(b)
            # plus any child intervals inside [a,b] (none except (a,b) since adjacent) -> none
            include_score = Ct * miss + bcost

            # exclude [a,b]
            exclude_set = frozenset()
            # cost: if child has (a,b) while parent doesn't -> Ct per such child
            have = count_children_with_interval(a, b)
            exclude_score = Ct * have
            # boundaries: since parent doesn't have a or b here, dp0 charges both boundaries mismatches:
            # if a child has boundary inside [a,b] (i.e start==a or end==b) and parent doesn't, that would be charged
            # but in the local base we treat boundary RF by counting child's boundaries mismatching parent having none:
            # parent_starts empty -> each child start inside costs Cb if present
            # parent_ends empty -> each child end inside costs Cb if present
            bcost = boundary_cost_exclude_start(a) + boundary_cost_exclude_end(b)
            exclude_score += bcost

            if include_score <= exclude_score:
                choice = "INCLUDE"
                if trace:
                    trace_log.append({
                        'region': f"[{i},{j}] ({a},{b})",
                        'choice': choice,
                        'intervals': set(include_set),
                        'score': include_score,
                        'details': f"base include: miss={miss}, boundary={bcost:.2f}"
                    })
                dp0_done[i][j] = True
                dp0_set[i][j] = include_set
                dp0_score[i][j] = include_score
                dp0_choice[i][j] = choice
                return (dp0_set[i][j], dp0_score[i][j], dp0_choice[i][j])
            else:
                choice = "EXCLUDE"
                if trace:
                    trace_log.append({
                        'region': f"[{i},{j}] ({a},{b})",
                        'choice': choice,
                        'intervals': set(),
                        'score': exclude_score,
                        'details': f"base exclude: have={have}, boundary={bcost:.2f}"
                    })
                dp0_done[i][j] = True
                dp0_set[i][j] = exclude_set
                dp0_score[i][j] = exclude_score
                dp0_choice[i][j] = choice
                return (dp0_set[i][j], dp0_score[i][j], dp0_choice[i][j])

        # general case
        best_score = float("inf")
        best_set = frozenset()
        best_choice = None
        best_details = ""

        # -------------------------
        # 1. INCLUDE [a,b] at parent
        # -------------------------
        # 1.1.1 -> call dpl(i, j-1)
        sub_set, sub_score, sub_choice = dpl(i, j-1)
        # extra tad penalties: intervals with start >= a and end == b (exclude (a,b))
        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i, j)
        # penalty for children missing (a,b)
        miss = count_children_missing_interval(a,b)
        c=positions[j-1]
        # boundary costs: consider both pi and pj in dp0 include
        bcost = boundary_cost_include_start(a) + boundary_cost_include_end(b)+ boundary_cost_exclude_start(c)
        total_score = sub_score + Ct * extra_cnt + Ct * miss + bcost
        candidate = frozenset({(a,b)}) | sub_set
        if total_score < best_score:
            best_score = total_score
            best_set = candidate
            best_choice = f"INCLUDE+dpl({i},{j-1})"
            best_details = f"dpl subchoice={sub_choice}, extra_cnt={extra_cnt}, miss={miss}, bcost={bcost:.2f}"

        # 1.1.2 -> call dpr(i+1, j)
        sub_set, sub_score, sub_choice = dpr(i+1, j)
        extra_cnt = count_child_intervals_start_eq_i_end_le_j(i, j)
        miss = count_children_missing_interval(a,b)
        c=positions[i+1]
        bcost = boundary_cost_include_start(a) + boundary_cost_include_end(b)+ boundary_cost_exclude_end(c)
        total_score = sub_score + Ct * extra_cnt + Ct * miss + bcost
        candidate = frozenset({(a,b)}) | sub_set
        if total_score < best_score:
            best_score = total_score
            best_set = candidate
            best_choice = f"INCLUDE+dpr({i+1},{j})"
            best_details = f"dpr subchoice={sub_choice}, extra_cnt={extra_cnt}, miss={miss}, bcost={bcost:.2f}"

        # 1.1.3 -> splits: for all k, call dpl(i,k) and dpr(k,j)
        for k in range(i+1, j):
            # only allow splits that are legal: either pk is both start & end, or pk in end and pk+1 in start
            pk = positions[k]
            # allow if pk is both
            if (pk in start_boundaries) and (pk in end_boundaries):
                left_set, left_score, _ = dpl(i, k)
                right_set, right_score, _ = dpr(k, j)
                # combine sub-scores and extra crossing penalties + missing interval + boundaries
                cross_cnt = count_child_intervals_crossing_k(i, k, j)
                miss = count_children_missing_interval(a,b)
                bcost = boundary_cost_include_start(a) + boundary_cost_include_end(b)
                total_score = left_score + right_score + Ct * cross_cnt + Ct * miss + bcost
                candidate = frozenset({(a,b)}) | left_set | right_set
                if total_score < best_score:
                    best_score = total_score
                    best_set = candidate
                    best_choice = f"INCLUDE+split@{k}(both)"
                    best_details = f"k={k}, cross={cross_cnt}, miss={miss}, bcost={bcost:.2f}"

        # -------------------------
        # 2. EXCLUDE [a,b] at parent
        # -------------------------
        # 2.1 call dp0(i, j-1) -> do not consider start pi here (dp0 will handle pi)
        sub_set, sub_score, sub_choice = dp0(i, j-1)
        # extra tad penalties: intervals with start >= a and end == b (exclude (a,b))
        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i, j)
        # since parent does NOT include [a,b], children that DO have (a,b) cost Ct each
        have = count_children_with_interval(a, b)
        # only end boundary pj is considered here (pi is handled inside dp0 subproblem)
        c=positions[j-1]
        bcost = boundary_cost_exclude_end(b)+boundary_cost_exclude_start(c)
        total_score = sub_score + Ct * extra_cnt + Ct * have + bcost
        candidate = sub_set
        if total_score < best_score:
            best_score = total_score
            best_set = frozenset(candidate)
            best_choice = f"EXCLUDE+dp0({i},{j-1})"
            best_details = f"sub={sub_choice}, extra_cnt={extra_cnt}, have={have}, bcost={bcost:.2f}"

        # 2.2 call dp0(i+1, j) -> symmetric
        sub_set, sub_score, sub_choice = dp0(i+1, j)
        extra_cnt = count_child_intervals_start_eq_i_end_le_j(i, j)
        have = count_children_with_interval(a, b)
        c=positions[i+1]
        bcost = boundary_cost_exclude_start(a)+boundary_cost_exclude_end(c)
        total_score = sub_score + Ct * extra_cnt + Ct * have + bcost
        candidate = sub_set
        if total_score < best_score:
            best_score = total_score
            best_set = frozenset(candidate)
            best_choice = f"EXCLUDE+dp0({i+1},{j})"
            best_details = f"sub={sub_choice}, extra_cnt={extra_cnt}, have={have}, bcost={bcost:.2f}"

        # 2.3 split -> dp0(i,k) and dp0(k,j)
        for k in range(i+1, j):
            pk = positions[k]
            if (pk in start_boundaries) and (pk in end_boundaries):
                left_set, left_score, _ = dp0(i, k)
                right_set, right_score, _ = dp0(k, j)
                cross_cnt = count_child_intervals_crossing_k(i, k, j)
                have = count_children_with_interval(a, b)
                # dp0 splits: neither pi nor pj are considered here
                total_score = left_score + right_score + Ct * cross_cnt + Ct * have
                candidate = left_set | right_set
                if total_score < best_score:
                    best_score = total_score
                    best_set = frozenset(candidate)
                    best_choice = f"EXCLUDE+split@{k}(both)"

        if trace:
            trace_log.append({
                'region': f"[{i},{j}] ({a},{b})",
                'choice': best_choice,
                'intervals': set(best_set),
                'score': best_score,
                'details': best_details
            })

        dp0_done[i][j] = True
        dp0_set[i][j] = best_set
        dp0_score[i][j] = best_score
        dp0_choice[i][j] = best_choice
        return dp0_set[i][j], dp0_score[i][j], dp0_choice[i][j]

    def dpl(i, j):
        # dpl: left boundary pi is already "created" outside (do NOT charge pi inside)
        if i >= j:
            return (frozenset(), inf, None)
        if dpl_done[i][j]:
            return (dpl_set[i][j], dpl_score[i][j], dpl_choice[i][j])

        a = positions[i]
        b = positions[j]
        if a not in start_boundaries or b not in end_boundaries:
            dpl_done[i][j] = True
            dpl_set[i][j] = frozenset()
            dpl_score[i][j] = inf
            dpl_choice[i][j] = None
            return (dpl_set[i][j], dpl_score[i][j], dpl_choice[i][j])

        # base case
        if j == i + 1:
            # include
            include_set = frozenset({(a,b)})
            miss = count_children_missing_interval(a,b)
            # only charge end boundary b (not start a)
            bcost = boundary_cost_include_end(b)
            include_score = Ct * miss + bcost
            # exclude
            exclude_set = frozenset()
            have = count_children_with_interval(a, b)
            # since pi is ignored, we only count child's end boundary if present
            bcost = boundary_cost_exclude_end(b)
            exclude_score = Ct * have + bcost
            if include_score <= exclude_score:
                if trace:
                    trace_log.append({'region': f"dpl [{i},{j}] ({a},{b})", 'choice': 'INCLUDE', 'score': include_score})
                dpl_done[i][j] = True
                dpl_set[i][j] = include_set
                dpl_score[i][j] = include_score
                dpl_choice[i][j] = "INCLUDE"
                return dpl_set[i][j], dpl_score[i][j], dpl_choice[i][j]
            else:
                if trace:
                    trace_log.append({'region': f"dpl [{i},{j}] ({a},{b})", 'choice': 'EXCLUDE', 'score': exclude_score})
                dpl_done[i][j] = True
                dpl_set[i][j] = exclude_set
                dpl_score[i][j] = exclude_score
                dpl_choice[i][j] = "EXCLUDE"
                return dpl_set[i][j], dpl_score[i][j], dpl_choice[i][j]

        best_score = float("inf")
        best_set = frozenset()
        best_choice = None
        best_details = ""

        # 1. INCLUDE [a,b]
        # 1.1 call dpl(i, j-1)
        sub_set, sub_score, sub_choice = dpl(i, j-1)
        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i, j)
        miss = count_children_missing_interval(a,b)
        # only charge end boundary b (not start a)
        c=positions[j-1]
        bcost = boundary_cost_include_end(b)+ boundary_cost_exclude_start(c)
        total_score = sub_score + Ct * extra_cnt + Ct * miss + bcost
        candidate = frozenset({(a,b)}) | sub_set
        if total_score < best_score:
            best_score = total_score
            best_set = candidate
            best_choice = f"dpl INCLUDE+dpl({i},{j-1})"
            best_details = f"extra_cnt={extra_cnt}, miss={miss}, bcost={bcost:.2f}"

        # 1.2 call dpr(i+1, j)
        sub_set, sub_score, sub_choice = dpr(i+1, j)
        extra_cnt = count_child_intervals_start_eq_i_end_le_j(i, j)
        miss = count_children_missing_interval(a,b)
        c=positions[i+1]
        bcost = boundary_cost_include_end(b)+ boundary_cost_exclude_end(c)
        total_score = sub_score + Ct * extra_cnt + Ct * miss + bcost
        candidate = frozenset({(a,b)}) | sub_set
        if total_score < best_score:
            best_score = total_score
            best_set = candidate
            best_choice = f"dpl INCLUDE+dpr({i+1},{j})"
            best_details = f"extra_cnt={extra_cnt}, miss={miss}, bcost={bcost:.2f}"

        # 1.3 splits: dpl(i,k) + dpr(k,j)
        for k in range(i+1, j):
            pk = positions[k]
            if (pk in start_boundaries) and (pk in end_boundaries):
                left_set, left_score, _ = dpl(i, k)
                right_set, right_score, _ = dpr(k, j)
                cross_cnt = count_child_intervals_crossing_k(i, k, j)
                miss = count_children_missing_interval(a,b)
                bcost = boundary_cost_include_end(b)
                total_score = left_score + right_score + Ct * cross_cnt + Ct * miss + bcost
                candidate = frozenset({(a,b)}) | left_set | right_set
                if total_score < best_score:
                    best_score = total_score
                    best_set = candidate
                    best_choice = f"dpl INCLUDE+split@{k}(both)"
                    best_details = f"k={k}, cross={cross_cnt}, miss={miss}"

        # 2. EXCLUDE [a,b]
        # 2.1 call dpl(i, j-1)
        sub_set, sub_score, sub_choice = dpl(i, j-1)
        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i, j)
        have = count_children_with_interval(a, b)
        # only charge end boundary b (not start a)
        c=positions[j-1]
        bcost = boundary_cost_exclude_end(b)+boundary_cost_exclude_start(c)
        total_score = sub_score + Ct * extra_cnt + Ct * have + bcost
        candidate = sub_set
        if total_score < best_score:
            best_score = total_score
            best_set = frozenset(candidate)
            best_choice = f"dpl EXCLUDE+dpl({i},{j-1})"
            best_details = f"extra_cnt={extra_cnt}, have={have}, bcost={bcost:.2f}"

        # 2.2 call dp0(i+1, j)
        sub_set, sub_score, sub_choice = dp0(i+1, j)
        extra_cnt = count_child_intervals_start_eq_i_end_le_j(i, j)
        have = count_children_with_interval(a, b)
        # start boundary a ignored, so only child-related have + subscore
        c=positions[i+1]
        bcost = boundary_cost_exclude_end(c)
        total_score = sub_score + Ct * extra_cnt + Ct * have + bcost
        candidate = sub_set
        if total_score < best_score:
            best_score = total_score
            best_set = frozenset(candidate)
            best_choice = f"dpl EXCLUDE+dp0({i+1},{j})"
            best_details = f"extra_cnt={extra_cnt}, have={have}"

        # 2.3 split: dpl(i,k) and dp0(k,j)
        for k in range(i+1, j):
            pk = positions[k]
            if (pk in start_boundaries) and (pk in end_boundaries):
                left_set, left_score, _ = dpl(i, k)
                right_set, right_score, _ = dp0(k, j)
                cross_cnt = count_child_intervals_crossing_k(i, k, j)
                have = count_children_with_interval(a, b)
                total_score = left_score + right_score + Ct * cross_cnt + Ct * have
                candidate = left_set | right_set
                if total_score < best_score:
                    best_score = total_score
                    best_set = frozenset(candidate)
                    best_choice = f"dpl EXCLUDE+split@{k}(both)"
                    best_details = f"k={k}, cross={cross_cnt}, have={have}"

        if trace:
            trace_log.append({
                'region': f"dpl [{i},{j}] ({a},{b})",
                'choice': best_choice,
                'intervals': set(best_set),
                'score': best_score,
                'details': best_details
            })
        dpl_done[i][j] = True
        dpl_set[i][j] = best_set
        dpl_score[i][j] = best_score
        dpl_choice[i][j] = best_choice
        return dpl_set[i][j], dpl_score[i][j], dpl_choice[i][j]

    def dpr(i, j):
        # dpr: right boundary pj is already "created" outside (do NOT charge pj inside)
        if i >= j:
            return (frozenset(), inf, None)
        if dpr_done[i][j]:
            return (dpr_set[i][j], dpr_score[i][j], dpr_choice[i][j])

        a = positions[i]
        b = positions[j]
        if a not in start_boundaries or b not in end_boundaries:
            dpr_done[i][j] = True
            dpr_set[i][j] = frozenset()
            dpr_score[i][j] = inf
            dpr_choice[i][j] = None
            return (dpr_set[i][j], dpr_score[i][j], dpr_choice[i][j])

        # base case
        if j == i + 1:
            # include
            include_set = frozenset({(a,b)})
            miss = count_children_missing_interval(a,b)
            # only charge start boundary a (not end b)
            bcost = boundary_cost_include_start(a)
            include_score = Ct * miss + bcost
            # exclude
            exclude_set = frozenset()
            have = count_children_with_interval(a, b)
            bcost = boundary_cost_exclude_start(a)
            exclude_score = Ct * have + bcost
            if include_score <= exclude_score:
                if trace:
                    trace_log.append({'region': f"dpr [{i},{j}] ({a},{b})", 'choice': 'INCLUDE', 'score': include_score})
                dpr_done[i][j] = True
                dpr_set[i][j] = include_set
                dpr_score[i][j] = include_score
                dpr_choice[i][j] = "INCLUDE"
                return dpr_set[i][j], dpr_score[i][j], dpr_choice[i][j]
            else:
                if trace:
                    trace_log.append({'region': f"dpr [{i},{j}] ({a},{b})", 'choice': 'EXCLUDE', 'score': exclude_score})
                dpr_done[i][j] = True
                dpr_set[i][j] = exclude_set
                dpr_score[i][j] = exclude_score
                dpr_choice[i][j] = "EXCLUDE"
                return dpr_set[i][j], dpr_score[i][j], dpr_choice[i][j]

        best_score = float("inf")
        best_set = frozenset()
        best_choice = None
        best_details = ""

        # 1. INCLUDE [a,b]
        # 1.1 call dpl(i, j-1)
        sub_set, sub_score, sub_choice = dpl(i, j-1)
        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i, j)
        miss = count_children_missing_interval(a,b)
        # only charge start boundary a (not end b)
        c=positions[j-1]
        bcost = boundary_cost_include_start(a)+boundary_cost_exclude_start(c)
        total_score = sub_score + Ct * extra_cnt + Ct * miss + bcost
        candidate = frozenset({(a,b)}) | sub_set
        if total_score < best_score:
            best_score = total_score
            best_set = candidate
            best_choice = f"dpr INCLUDE+dpl({i},{j-1})"
            best_details = f"extra_cnt={extra_cnt}, miss={miss}, bcost={bcost:.2f}"

        # 1.2 call dpr(i+1, j)
        sub_set, sub_score, sub_choice = dpr(i+1, j)
        extra_cnt = count_child_intervals_start_eq_i_end_le_j(i, j)
        miss = count_children_missing_interval(a,b)
        c=positions[i+1]
        bcost = boundary_cost_include_start(a)+boundary_cost_exclude_end(c)
        total_score = sub_score + Ct * extra_cnt + Ct * miss + bcost
        candidate = frozenset({(a,b)}) | sub_set
        if total_score < best_score:
            best_score = total_score
            best_set = candidate
            best_choice = f"dpr INCLUDE+dpr({i+1},{j})"
            best_details = f"extra_cnt={extra_cnt}, miss={miss}, bcost={bcost:.2f}"

        # 1.3 splits: dpl(i,k) + dpr(k,j)
        for k in range(i+1, j):
            pk = positions[k]
            if (pk in start_boundaries) and (pk in end_boundaries):
                left_set, left_score, _ = dpl(i, k)
                right_set, right_score, _ = dpr(k, j)
                cross_cnt = count_child_intervals_crossing_k(i, k, j)
                miss = count_children_missing_interval(a,b)
                bcost = boundary_cost_include_start(a)
                total_score = left_score + right_score + Ct * cross_cnt + Ct * miss + bcost
                candidate = frozenset({(a,b)}) | left_set | right_set
                if total_score < best_score:
                    best_score = total_score
                    best_set = candidate
                    best_choice = f"dpr INCLUDE+split@{k}(both)"
                    best_details = f"k={k}, cross={cross_cnt}, miss={miss}"


        # 2. EXCLUDE [a,b]
        # 2.1 call dp0(i, j-1)
        sub_set, sub_score, sub_choice = dp0(i, j-1)
        extra_cnt = count_child_intervals_start_ge_i_end_eq_j(i, j)
        have = count_children_with_interval(a, b)
        c=positions[j-1]
        bcost = boundary_cost_exclude_start(c)
        total_score = sub_score + Ct * extra_cnt + Ct * have + bcost
        candidate = sub_set
        if total_score < best_score:
            best_score = total_score
            best_set = frozenset(candidate)
            best_choice = f"dpr EXCLUDE+dpl({i},{j-1})"
            best_details = f"extra_cnt={extra_cnt}, have={have}, bcost={bcost:.2f}"

        # 2.2 call dp0(i+1, j)
        sub_set, sub_score, sub_choice = dp0(i+1, j)
        extra_cnt = count_child_intervals_start_eq_i_end_le_j(i, j)
        have = count_children_with_interval(a, b)
        c=positions[i+1]
        bcost = boundary_cost_exclude_start(a)+boundary_cost_exclude_end(c)
        total_score = sub_score + Ct * extra_cnt + Ct * have + bcost
        candidate = sub_set
        if total_score < best_score:
            best_score = total_score
            best_set = frozenset(candidate)
            best_choice = f"dpr EXCLUDE+dp0({i+1},{j})"
            best_details = f"extra_cnt={extra_cnt}, have={have}, bcost={bcost:.2f}"

        # 2.3 split: dp0(i,k) and dpr(k,j)
        for k in range(i+1, j):
            pk = positions[k]
            if (pk in start_boundaries) and (pk in end_boundaries):
                left_set, left_score, _ = dp0(i, k)
                right_set, right_score, _ = dpr(k, j)
                cross_cnt = count_child_intervals_crossing_k(i, k, j)
                have = count_children_with_interval(a, b)
                total_score = left_score + right_score + Ct * cross_cnt + Ct * have
                candidate = left_set | right_set
                if total_score < best_score:
                    best_score = total_score
                    best_set = frozenset(candidate)
                    best_choice = f"dpr EXCLUDE+split@{k}(both)"
                    best_details = f"k={k}, cross={cross_cnt}, have={have}"

        if trace:
            trace_log.append({
                'region': f"dpr [{i},{j}] ({a},{b})",
                'choice': best_choice,
                'intervals': set(best_set),
                'score': best_score,
                'details': best_details
            })
        dpr_done[i][j] = True
        dpr_set[i][j] = best_set
        dpr_score[i][j] = best_score
        dpr_choice[i][j] = best_choice
        return dpr_set[i][j], dpr_score[i][j], dpr_choice[i][j]

    # -------------------------
    # Trace: only solution path from root to leaves
    # -------------------------
    def build_solution_trace():
        path = []
        visited = set()

        def region_label(func, i, j):
            return f"{func} [{i},{j}] ({positions[i]},{positions[j]})"

        def next_calls(func, i, j, choice):
            if not choice:
                return []

            if "split@" in choice:
                split_match = re.search(r"split@(\d+)\((both|adj)\)", choice)
                if not split_match:
                    return []
                k = int(split_match.group(1))
                is_adj = split_match.group(2) == "adj"
                is_exclude = "EXCLUDE" in choice

                if func == "dp0":
                    if is_exclude:
                        left = ("dp0", i, k)
                        right = ("dp0", k + 1 if is_adj else k, j)
                    else:
                        left = ("dpl", i, k)
                        right = ("dpr", k + 1 if is_adj else k, j)
                elif func == "dpl":
                    if is_exclude:
                        left = ("dpl", i, k)
                        right = ("dp0", k + 1 if is_adj else k, j)
                    else:
                        left = ("dpl", i, k)
                        right = ("dpr", k + 1 if is_adj else k, j)
                else:  # dpr
                    if is_exclude:
                        left = ("dp0", i, k)
                        right = ("dpr", k + 1 if is_adj else k, j)
                    else:
                        left = ("dpl", i, k)
                        right = ("dpr", k + 1 if is_adj else k, j)
                return [left, right]

            call_match = re.search(r"(dp0|dpl|dpr)\((\d+),(\d+)\)", choice)
            if call_match:
                return [(call_match.group(1), int(call_match.group(2)), int(call_match.group(3)))]
            return []

        def visit(func, i, j):
            key = (func, i, j)
            if key in visited:
                return
            visited.add(key)

            if func == "dp0":
                intervals, score, choice = dp0(i, j)
            elif func == "dpl":
                intervals, score, choice = dpl(i, j)
            else:
                intervals, score, choice = dpr(i, j)

            path.append({
                "region": region_label(func, i, j),
                "choice": choice,
                "score": score,
                "intervals": set(intervals),
                "details": ""
            })

            for next_func, ni, nj in next_calls(func, i, j, choice):
                visit(next_func, ni, nj)

        visit("dp0", 0, n - 1)
        return path

    # -------------------------
    # Solve and return
    # -------------------------
    intervals, score, _ = dp0(0, n-1)
    result = (set(intervals), score)

    # Return trace if requested
    if trace_requested:
        trace_log = build_solution_trace()
        result = result + (trace_log,)

    return result



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
            end_boundaries.add(b)
    
    return start_boundaries, end_boundaries


def calculate_parsimony_score(root_intervals, edges, leaf_intervals, Ct=1.0, Cb=1.0):
    """
    Calculate the parsimony (RF distance) score for a given root interval assignment.
    
    Parameters
    ----------
    root_intervals : set of (a, b) tuples
        Intervals assigned to the root node (node 0)
    edges : list of (parent, child) tuples
        Tree edges
    leaf_intervals : dict
        Mapping leaf node -> set of interval strings like "[a,b]"
    Ct : float
        Cost for interval differences
    Cb : float
        Cost for boundary differences
    
    Returns
    -------
    score : float
        Total parsimony score
    """
    def parse_interval(s):
        a, b = map(int, re.findall(r"\d+", s))
        return (a, b)
    
    # Get children of root (node 0)
    children = [v for (u, v) in edges if u == 0]
    
    # Parse child intervals
    child_intervals = {}
    for c in children:
        child_intervals[c] = set(parse_interval(x) for x in leaf_intervals.get(c, []))
    
    # Calculate RF distance
    total_score = 0.0
    
    root_starts = {a for (a, b) in root_intervals}
    root_ends = {b for (a, b) in root_intervals}
    
    for c in children:
        child_set = child_intervals[c]
        child_starts = {a for (a, b) in child_set}
        child_ends = {b for (a, b) in child_set}
        
        # RF on intervals (symmetric difference)
        total_score += Ct * (len(root_intervals - child_set) + len(child_set - root_intervals))
        
        # RF on start boundaries (symmetric difference)
        total_score += Cb * (len(root_starts - child_starts) + len(child_starts - root_starts))
        
        # RF on end boundaries (symmetric difference)
        total_score += Cb * (len(root_ends - child_ends) + len(child_ends - root_ends))
    
    return total_score


def compare_dp_vs_tad_distance(edges, leaf_intervals, inferred_root_intervals, 
                               dp_score, Ct=1.0, Cb=1.0):
    """
    Compare DP score with TAD distance for 2-leaf trees.
    
    For a tree with exactly 2 leaves, this function checks if the DP score
    matches the direct TAD (Robinson-Foulds) distance calculation.
    
    Parameters
    ----------
    edges : list of (parent, child) tuples
        Tree edges
    leaf_intervals : dict
        Mapping leaf node -> set of interval strings like "[a,b]"
    inferred_root_intervals : set of (a, b) tuples
        Root intervals inferred by DP algorithm
    dp_score : float
        Score returned by DP algorithm
    Ct : float
        Cost for interval differences
    Cb : float
        Cost for boundary differences
    
    Returns
    -------
    dict with keys:
        - 'is_two_leaf': bool, True if tree has exactly 2 leaves
        - 'dp_score': float, the DP score
        - 'tad_distance': float, the TAD distance (only if is_two_leaf)
        - 'match': bool, whether they match (only if is_two_leaf)
    """
    children = [v for (u, v) in edges if u == 0]
    
    result = {
        'is_two_leaf': len(children) == 2,
        'dp_score': dp_score,
    }
    
    if result['is_two_leaf']:
        tad_distance = calculate_parsimony_score(
            inferred_root_intervals, edges, leaf_intervals, Ct=Ct, Cb=Cb
        )
        result['tad_distance'] = tad_distance
        result['match'] = abs(dp_score - tad_distance) < 1e-6
        
        print(f"\n[2-LEAF TREE CHECK]")
        print(f"  DP score:     {dp_score:.6f}")
        print(f"  TAD distance: {tad_distance:.6f}")
        print(f"  Match:        {result['match']}")
    
    return result


def print_trace(trace_log):
    """
    Pretty print the DP algorithm trace.
    
    Parameters
    ----------
    trace_log : list of dicts
        Trace log from DP algorithm
    """
    print(f"\n{'='*80}")
    print(f"DP Algorithm Trace")
    print(f"{'='*80}")
    
    for i, entry in enumerate(trace_log, 1):
        region = entry.get('region', '(unknown)')
        choice = entry.get('choice', '(unknown)')
        score = entry.get('score', None)
        intervals = entry.get('intervals', None)
        details = entry.get('details', '')

        print(f"\n[Step {i}] Region {region}")
        print(f"  Decision: {choice}")
        if score is not None:
            print(f"  Score: {score:.6f}")
        else:
            print("  Score: (n/a)")
        if intervals:
            print(f"  Intervals: {sorted(intervals)}")
        else:
            print(f"  Intervals: (empty)")
        print(f"  Details: {details}")
    
    print(f"\n{'='*80}\n")


def save_trace(trace_log, filename):
    """
    Save the DP algorithm trace to a text file.
    
    Parameters
    ----------
    trace_log : list of dicts
        Trace log from DP algorithm
    filename : str
        Output filename
    """
    with open(filename, 'w') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"DP Algorithm Trace\n")
        f.write(f"{'='*80}\n")
        
        for i, entry in enumerate(trace_log, 1):
            region = entry.get('region', '(unknown)')
            choice = entry.get('choice', '(unknown)')
            score = entry.get('score', None)
            intervals = entry.get('intervals', None)
            details = entry.get('details', '')

            f.write(f"\n[Step {i}] Region {region}\n")
            f.write(f"  Decision: {choice}\n")
            if score is not None:
                f.write(f"  Score: {score:.6f}\n")
            else:
                f.write("  Score: (n/a)\n")
            if intervals:
                f.write(f"  Intervals: {sorted(intervals)}\n")
            else:
                f.write(f"  Intervals: (empty)\n")
            f.write(f"  Details: {details}\n")
        
        f.write(f"\n{'='*80}\n")
    
    print(f"Trace saved to {filename}")


def test_dp_vs_ilp(num_tests=10, min_children=2, max_children=5, 
                   min_val=1, max_val=20, num_intervals=10, 
                   max_per_leaf=5, Ct=1.0, Cb=1.0, verbose=True, show_trace=False):
    """
    Test DP algorithm against ILP solver using random star topology trees.
    
    Parameters
    ----------
    num_tests : int
        Number of random test cases to run
    min_children : int
        Minimum number of children in star topology
    max_children : int
        Maximum number of children in star topology
    min_val : int
        Minimum value for interval boundaries
    max_val : int
        Maximum value for interval boundaries
    num_intervals : int
        Number of random intervals to generate
    max_per_leaf : int
        Maximum intervals per leaf node
    Ct : float
        Cost for interval differences
    Cb : float
        Cost for boundary differences
    verbose : bool
        Print detailed output
    show_trace : bool
        Print DP algorithm trace for each test
    """

    
    print(f"\n{'='*70}")
    print(f"Testing DP vs ILP on Star Topology Trees")
    print(f"{'='*70}")
    print(f"Number of tests: {num_tests}")
    print(f"Children per tree: {min_children}-{max_children}")
    print(f"Interval range: [{min_val}, {max_val}]")
    print(f"Intervals pool size: {num_intervals}")
    print(f"Max intervals per leaf: {max_per_leaf}")
    print(f"Costs: Ct={Ct}, Cb={Cb}\n")
    
    matches = 0
    mismatches = 0
    failures = 0
    
    for test_num in range(1, num_tests + 1):
        # Generate random star topology
        num_children = random.randint(min_children, max_children)
        edges, leaf_nodes = generate_star_topology(num_children)
        
        # Generate random intervals
        available_intervals = generate_random_intervals(min_val, max_val, num_intervals)
        
        # Assign random laminar intervals to leaves
        leaf_intervals = generate_random_laminar_intervals_from_list(
            leaf_nodes, available_intervals, max_per_leaf=max_per_leaf
        )
        
        # Extract boundaries
    # across all leaf intervals find the min start and max end without using function
        # min_start = min(int(re.findall(r"\d+", interval)[0]) for intervals in leaf_intervals.values() for interval in intervals)
        # max_end = max(int(re.findall(r"\d+", interval)[1]) for intervals in leaf_intervals.values() for interval in intervals)
        # start_boundaries = range(min_start, max_end + 1)
        # end_boundaries = range(min_start, max_end + 1)
        
        # Extract and combine all start and end boundaries from leaf intervals
        all_boundaries = set()
        for intervals in leaf_intervals.values():
            for interval in intervals:
                nums = re.findall(r"\d+", interval)
                if len(nums) >= 2:
                    all_boundaries.add(int(nums[0]))  # start boundary
                    all_boundaries.add(int(nums[1]))  # end boundary
        
        # Sort and assign combined boundaries to both start and end
        start_boundaries = sorted(all_boundaries)
        end_boundaries = sorted(all_boundaries)



        if verbose:
            print(f"\nTest {test_num}:")
            print(f"  Tree: Star topology with {num_children} children")
            print(f"  Edges: {edges}")
            print(f"  Leaf intervals:")
            for leaf in sorted(leaf_nodes):
                print(f"    Node {leaf}: {sorted(leaf_intervals[leaf])}")
            # print(f"  Start boundaries: {sorted(start_boundaries)}")
            # print(f"  End boundaries: {sorted(end_boundaries)}")
        
        try:
            # Run DP solver with trace if requested
            result = infer_tads_with_dp(
                edges, leaf_intervals, start_boundaries, end_boundaries,
                Ct=Ct, Cb=Cb, trace=show_trace
            )
            
            if show_trace:
                dp_intervals, dp_score, trace_log = result
            else:
                dp_intervals, dp_score = result
                trace_log = None
            
            # Run ILP solver
            ilp_cost, is_integral, node_assignments, x_vars, nodes, intervals = \
                solve_lp_from_edges_and_leaves(
                    edges, leaf_intervals, Ct=Ct, Cb=Cb, mode="convex",
                    require_laminar_internal=True, use_fitch_hartigan=False,
                    verbose=False
                )
            
            if ilp_cost is None:
                print(f"  ❌ Test {test_num}: ILP solver failed")
                failures += 1
                continue
            
            # Get ILP intervals at root (node 0)
            ilp_root_intervals = node_assignments.get(0, set())
            
            # Calculate parsimony score for ILP solution
            ilp_parsimony_score = calculate_parsimony_score(
                ilp_root_intervals, edges, leaf_intervals, Ct=Ct, Cb=Cb
            )
            
            # Compare scores
            score_diff = abs(dp_score - ilp_parsimony_score)
            match = score_diff < 0.01  # Allow small numerical differences
            
            if verbose:
                print(f"  DP score:          {dp_score:.6f}")
                print(f"  ILP objective:     {ilp_cost:.6f}")
                print(f"  ILP parsimony:     {ilp_parsimony_score:.6f}")
                print(f"  Difference (DP vs ILP parsimony): {score_diff:.6f}")
                print(f"  DP intervals:  {sorted(dp_intervals)}")
                print(f"  ILP intervals: {sorted(ilp_root_intervals)}")
            
            if match:
                print(f"  ✓ Test {test_num}: MATCH (diff={score_diff:.6f})")
                matches += 1
                if show_trace and trace_log:
                    print_trace(trace_log)
            else:
                print(f"  ✗ Test {test_num}: MISMATCH (diff={score_diff:.6f})")
                print(f"    DP score:       {dp_score:.6f}")
                print(f"    ILP parsimony:  {ilp_parsimony_score:.6f}")
                print(f"    DP intervals:   {sorted(dp_intervals)}")
                print(f"    ILP intervals:  {sorted(ilp_root_intervals)}")
                print("    Leaf TADs:")
                for leaf in sorted(leaf_nodes):
                    print(f"      Node {leaf}: {sorted(leaf_intervals[leaf])}")
                if show_trace and trace_log:
                    print_trace(trace_log)
                mismatches += 1
                break
                
        except Exception as e:
            print(f"  ❌ Test {test_num}: EXCEPTION - {str(e)}")
            import traceback
            if verbose:
                traceback.print_exc()
            failures += 1
    
    print(f"\n{'='*70}")
    print(f"Test Results Summary")
    print(f"{'='*70}")
    print(f"Total tests: {num_tests}")
    print(f"Matches: {matches}")
    print(f"Mismatches: {mismatches}")
    print(f"Failures: {failures}")
    print(f"Success rate: {matches}/{num_tests - failures} ({100*matches/max(1, num_tests-failures):.1f}%)")
    print(f"{'='*70}\n")


def test_2leaf_tree(num_tests=10, min_val=1, max_val=20, num_intervals=10, 
                    max_per_leaf=5, Ct=1.0, Cb=1.0, verbose=True):
    """
    Test DP score vs TAD distance for 2-leaf trees.
    
    For each test, generates a random 2-leaf tree with random intervals,
    runs the DP algorithm, and compares the DP score with the direct
    TAD (Robinson-Foulds) distance calculation.
    
    Parameters
    ----------
    num_tests : int
        Number of random 2-leaf test cases to run
    min_val : int
        Minimum value for interval boundaries
    max_val : int
        Maximum value for interval boundaries
    num_intervals : int
        Number of random intervals to generate
    max_per_leaf : int
        Maximum intervals per leaf node
    Ct : float
        Cost for interval differences
    Cb : float
        Cost for boundary differences
    verbose : bool
        Print detailed output
    """
    
    print(f"\n{'='*70}")
    print(f"Testing DP vs TAD Distance on 2-Leaf Trees")
    print(f"{'='*70}")
    print(f"Number of tests: {num_tests}")
    print(f"Interval range: [{min_val}, {max_val}]")
    print(f"Intervals pool size: {num_intervals}")
    print(f"Max intervals per leaf: {max_per_leaf}")
    print(f"Costs: Ct={Ct}, Cb={Cb}\n")
    
    matches = 0
    mismatches = 0
    failures = 0
    
    for test_num in range(1, num_tests + 1):
        # Generate 2-leaf star topology
        edges = [(0, 1), (0, 2)]
        leaf_nodes = [1, 2]
        
        # Generate random intervals
        available_intervals = generate_random_intervals(min_val, max_val, num_intervals)
        
        # Assign random laminar intervals to leaves
        leaf_intervals = generate_random_laminar_intervals_from_list(
            leaf_nodes, available_intervals, max_per_leaf=max_per_leaf
        )
        
        # Extract all boundaries from leaf intervals
        all_boundaries = set()
        for intervals in leaf_intervals.values():
            for interval in intervals:
                nums = re.findall(r"\d+", interval)
                if len(nums) >= 2:
                    all_boundaries.add(int(nums[0]))  # start boundary
                    all_boundaries.add(int(nums[1]))  # end boundary
        
        # Sort and assign combined boundaries
        start_boundaries = sorted(all_boundaries)
        end_boundaries = sorted(all_boundaries)
        
        if verbose:
            print(f"\nTest {test_num}:")
            print(f"  Edges: {edges}")
            print(f"  Leaf intervals:")
            for leaf in sorted(leaf_nodes):
                print(f"    Node {leaf}: {sorted(leaf_intervals[leaf])}")
        
        try:
            # Run DP solver
            result = infer_tads_with_dp(
                edges, leaf_intervals, start_boundaries, end_boundaries,
                Ct=Ct, Cb=Cb, trace=False
            )
            
            dp_intervals, dp_score = result
            
            # Compare DP score with TAD distance
            comparison = compare_dp_vs_tad_distance(
                edges, leaf_intervals, dp_intervals, dp_score, Ct=Ct, Cb=Cb
            )
            
            if comparison['is_two_leaf']:
                if comparison['match']:
                    print(f"  ✓ Test {test_num}: MATCH")
                    matches += 1
                else:
                    print(f"  ✗ Test {test_num}: MISMATCH")
                    print(f"    DP intervals: {sorted(dp_intervals)}")
                    mismatches += 1
            else:
                print(f"  ❌ Test {test_num}: NOT a 2-leaf tree")
                failures += 1
                
        except Exception as e:
            print(f"  ❌ Test {test_num}: EXCEPTION - {str(e)}")
            import traceback
            if verbose:
                traceback.print_exc()
            failures += 1
    
    print(f"\n{'='*70}")
    print(f"2-Leaf Test Results")
    print(f"{'='*70}")
    print(f"Total tests: {num_tests}")
    print(f"Matches: {matches}")
    print(f"Mismatches: {mismatches}")
    print(f"Failures: {failures}")
    print(f"Success rate: {matches}/{num_tests - failures} ({100*matches/max(1, num_tests-failures):.1f}%)")
    print(f"{'='*70}\n")


# -------------------------
# Main Entry Point
# -------------------------
if __name__ == "__main__":
    # Seed for reproducibility (optional)
    # random.seed(42)
    
    # Run 2-leaf tree tests
    # test_2leaf_tree(
    #     num_tests=10000,
    #     min_val=1,
    #     max_val=10,
    #     num_intervals=40,
    #     max_per_leaf=5,
    #     Ct=1.0,
    #     Cb=1.0,
    #     verbose=True
    # )
    test_dp_vs_ilp(
        num_tests=100,
        min_children=2,
        max_children=5,
        min_val=1,
        max_val=20,
        num_intervals=10,
        max_per_leaf=5,
        Ct=1.0,
        Cb=1.0,
        verbose=True)

    # edges = [(0, 1), (0, 2)]
    # leaf_intervals = { 
    #     1: ['[1,5]','[6,10]'],
    #     2: ['[11,15]','[16,20]'],

    # }
    # start_boundaries = [1,11,5,6,15,10,16,20]
    # end_boundaries = [1,11,5,6,15,10,16,20]
    
    # # # Run with trace enabled
    # result = infer_tads_with_dp(
    #     edges, leaf_intervals, start_boundaries, end_boundaries,
    #     Ct=1.0, Cb=1.0, trace=True
    # )
    # dp_intervals, dp_score, trace_log = result
    
    # print(f"DP intervals: {sorted(dp_intervals)}")
    # print(f"DP score: {dp_score:.6f}")

    # ilp_cost, is_integral, node_assignments, x_vars, nodes, intervals = \
    #             solve_lp_from_edges_and_leaves(
    #                 edges, leaf_intervals, Ct=1.0, Cb=1.0, mode="convex",
    #                 require_laminar_internal=True, use_fitch_hartigan=False,
    #                 verbose=False
    #             )
            
    # if ilp_cost is None:
    #     print(f"  ❌ ILP solver failed")
    # else:
            
    #     # Get ILP intervals at root (node 0)
    #     ilp_root_intervals = node_assignments.get(0, set())
        
    #     # Calculate parsimony score for ILP solution
    #     ilp_parsimony_score = calculate_parsimony_score(
    #         ilp_root_intervals, edges, leaf_intervals, Ct=1, Cb=1
    #     )
    # # print results
    #     print(f"ILP parsimony:     {ilp_parsimony_score:.6f}")
    #     print(f"ILP intervals: {sorted(ilp_root_intervals)}")
    
    # Print the trace
    # print_trace(trace_log)
    
    # Save the trace to a file
    # save_trace(trace_log, "dp_trace.txt")