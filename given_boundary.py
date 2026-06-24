from functools import lru_cache

def laminar_dp(positions, boundary_type):
    """
    positions: sorted list of boundary coordinates
    boundary_type: dict {position: 'start' | 'end' | 'both'}
    """

    n = len(positions)

    def crosses(a, b, s, e):
        """Return True if intervals [a,b] and [s,e] cross."""
        return (a < s < b < e) or (s < a < e < b)

    @lru_cache(None)
    def dp(i, j, s_idx, e_idx, status):
        """
        Returns True if there exists a laminar assignment
        inside [positions[i], positions[j]] satisfying:
        interval [s,e] is present/absent depending on status.
        """

        a = positions[i]
        b = positions[j]
        s = positions[s_idx]
        e = positions[e_idx]
        if a>=b:
            return False

        # -----------------------
        # Base case
        # -----------------------
        if j == i + 1:
            if status == "present":
                if crosses(a, b, s, e):
                    return False
                if s >= a and e <= b and not (a == s and b == e):
                    # [s,e] strictly inside base interval
                    return False
                return True
            else:  # status == "absent"
                if a == s and b == e:
                    return False
                return True

        # -----------------------
        # Try creating [a,b]
        # -----------------------
        can_create = True

        if status == "present":
            if crosses(a, b, s, e):
                can_create = False

        elif status == "absent":
            if a == s and b == e:
                can_create = False

        if can_create:

            # 1.1 shrink left
            if boundary_type.get(positions[i+1]) in ("start"):
                if dp(i+1, j, s_idx, e_idx, status):
                    return True

            # 1.2 shrink right
            if boundary_type.get(positions[j-1]) in ("end"):
                if dp(i, j-1, s_idx, e_idx, status):
                    return True

            # 1.3 split cases
            for k in range(i+1, j):

                pk = positions[k]

                # case: pk is both
                if boundary_type.get(pk) == "both":
                    if dp(i, k, s_idx, e_idx, status) and \
                       dp(k, j, s_idx, e_idx, status):
                        return True

                # case: pk is end and p(k+1) is start
                if boundary_type.get(pk) == "end":
                    if k+1 < j:
                        pk1 = positions[k+1]
                        if boundary_type.get(pk1) == "start":
                            if dp(i, k, s_idx, e_idx, status) and \
                               dp(k+1, j, s_idx, e_idx, status):
                                return True

        # -----------------------
        # Do NOT create [a,b]
        # -----------------------
        if status == "present" and a == s and b == e:
            return False

        # Only splitting allowed
        for k in range(i+1, j):

            pk = positions[k]

            if boundary_type.get(pk) == "both":
                if dp(i, k, s_idx, e_idx, status) and \
                   dp(k, j, s_idx, e_idx, status):
                    return True

            if boundary_type.get(pk) == "end":
                if k+1 < j:
                    pk1 = positions[k+1]
                    if boundary_type.get(pk1) == "start":
                        if dp(i, k, s_idx, e_idx, status) and \
                           dp(k+1, j, s_idx, e_idx, status):
                            return True

        return False

    return dp


start_boundaries = [1,2,4,5]
end_boundaries = [2,6,7]
boundary_type = {}
for b in start_boundaries:
    boundary_type[b] = "start"
for b in end_boundaries:
    if b in boundary_type:
        boundary_type[b] = "both"
    else:
        boundary_type[b] = "end"
positions = sorted(set(start_boundaries + end_boundaries))

dp = laminar_dp(positions, boundary_type)

# check if interval [5,20] must be present
s_idx = positions.index(4)
e_idx = positions.index(7)

print(dp(0, len(positions)-1, s_idx, e_idx, "absent"))  # True