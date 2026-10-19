"""Exact TAD-hierarchy median under the paper's distance, compiled with numba.

Same problem as nussinov.infer_tads_with_dp on a star (one parent, m leaves):

    minimize over laminar M   sum_c  Ct |T(M) delta T_c| + Cb (|S(M) delta S_c| + |E(M) delta E_c|)

Up to a constant this is maximizing
    sum_{I in M} Ct (2 f(I) - m) + sum_{s in S(M)} Cb (2 g(s) - m) + sum_{e in E(M)} Cb (2 h(e) - m)
with f, g, h the number of leaves holding interval I, start s, end e. A start (end) is paid
once however many intervals of M share it, which is why the recursion carries two flags:

  G(i, j, l, r)  best family inside [p_i, p_j]; l = start p_i already paid by an enclosing
                 interval, r = end p_j already paid
  H(i, j, l, r)  the same, but without the interval [p_i, p_j] itself

  G = max(H(i,j,l,r),  w(i,j) + [not l] a_i + [not r] b_j + H(i,j,1,1))
  H = max(G(i+1,j,0,r),  G(i,j-1,l,0),
          max_k  w(i,k) + [not l] a_i + b_k + H(i,k,1,1) + G(k,j,0,r))

The last case takes [p_i, p_k] as the largest interval starting at p_i; no interval can cross
p_k, so the family splits there. O(n^3) time, O(n^2) memory, no sets in the tables.
Checked against nussinov.py in check_fast_median.py.
"""

import numpy as np
from numba import njit

NEG = -1e300


@njit(cache=True)
def _solve(w, a, b):
    n = len(a)
    G = np.full((n, n, 2, 2), 0.0)
    H = np.full((n, n, 2, 2), 0.0)
    # choice codes for traceback
    gc = np.zeros((n, n, 2, 2), np.int8)      # 0 = H, 1 = include [i, j]
    hc = np.zeros((n, n, 2, 2), np.int64)     # -1 = drop i, -2 = drop j, k >= 0 = split at k
    for span in range(1, n):
        for i in range(n - span):
            j = i + span
            for l in range(2):
                for r in range(2):
                    best = G[i + 1, j, 0, r]
                    choice = -1
                    v = G[i, j - 1, l, 0]
                    if v > best:
                        best = v
                        choice = -2
                    for k in range(i + 1, j):
                        if w[i, k] <= NEG:
                            continue
                        v = w[i, k] + (0.0 if l else a[i]) + b[k] + H[i, k, 1, 1] + G[k, j, 0, r]
                        if v > best:
                            best = v
                            choice = k
                    H[i, j, l, r] = best
                    hc[i, j, l, r] = choice
            for l in range(2):
                for r in range(2):
                    best = H[i, j, l, r]
                    c = 0
                    if w[i, j] > NEG:
                        v = w[i, j] + (0.0 if l else a[i]) + (0.0 if r else b[j]) + H[i, j, 1, 1]
                        if v > best:
                            best = v
                            c = 1
                    G[i, j, l, r] = best
                    gc[i, j, l, r] = c
    return G, gc, hc


@njit(cache=True)
def _traceback(gc, hc, n):
    out = np.zeros((2 * n, 2), np.int64)
    used = 0
    # explicit stack of (kind, i, j, l, r); kind 0 = G, 1 = H
    stack = np.zeros((4 * n * n + 4, 5), np.int64)
    top = 0
    stack[0] = (0, 0, n - 1, 0, 0)
    top = 1
    while top > 0:
        top -= 1
        kind, i, j, l, r = stack[top]
        if i >= j:
            continue
        if kind == 0:
            if gc[i, j, l, r] == 1:
                out[used] = (i, j)
                used += 1
                stack[top] = (1, i, j, 1, 1)
            else:
                stack[top] = (1, i, j, l, r)
            top += 1
        else:
            c = hc[i, j, l, r]
            if c == -1:
                stack[top] = (0, i + 1, j, 0, r)
                top += 1
            elif c == -2:
                stack[top] = (0, i, j - 1, l, 0)
                top += 1
            else:
                out[used] = (i, c)
                used += 1
                stack[top] = (1, i, c, 1, 1)
                stack[top + 1] = (0, c, j, 0, r)
                top += 2
    return out[:used]


def median(member_sets, ct=1.0, cb=1.0):
    """Exact median of a list of laminar interval sets ((a, b) tuples); returns (set, cost)."""
    m = len(member_sets)
    pos = sorted({p for s in member_sets for iv in s for p in iv})
    if len(pos) < 2:
        return set(), float(sum(len(s) + 2 * cb * len(s) for s in member_sets))
    index = {p: i for i, p in enumerate(pos)}
    n = len(pos)
    f = np.zeros((n, n))
    g = np.zeros(n)
    h = np.zeros(n)
    for s in member_sets:
        for a_, b_ in s:
            f[index[a_], index[b_]] += 1
        for p in {a_ for a_, _ in s}:
            g[index[p]] += 1
        for p in {b_ for _, b_ in s}:
            h[index[p]] += 1
    w = ct * (2 * f - m)
    w[np.tril_indices(n)] = NEG          # intervals need start < end
    a = cb * (2 * g - m)
    b = cb * (2 * h - m)
    G, gc, hc = _solve(w, a, b)
    picked = _traceback(gc, hc, n)
    result = {(pos[i], pos[j]) for i, j in picked}
    const = sum(ct * len(s) + cb * (len({x for x, _ in s}) + len({y for _, y in s})) for s in member_sets)
    return result, float(const - G[0, n - 1, 0, 0])
