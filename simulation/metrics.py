"""Boundary-level precision / recall / F1 with a bin tolerance.

Both the ground-truth ``.tad`` files (written by hierarchy.write_tad) and the
OnTAD ``.tad`` calls share the same layout, so a single parser serves both.
"""

from __future__ import annotations


def parse_tad_boundaries(path) -> set[int]:
    """Read a ``.tad`` file and return the set of 0-indexed boundary bins.

    The first line (whole region, level 0) is skipped. Each domain line
    ``start end ...`` is 1-indexed inclusive; its boundaries in 0-indexed
    coordinates are ``start-1`` (left) and ``end`` (right, half-open).
    """
    bounds: set[int] = set()
    with open(path) as fh:
        for i, line in enumerate(fh):
            parts = line.split()
            if i == 0 or len(parts) < 2:
                continue
            start, end = int(parts[0]), int(parts[1])
            bounds.add(start - 1)
            bounds.add(end)
    return bounds


def boundary_prf(pred: set[int], true: set[int], tol: int) -> tuple[float, float, float]:
    """Greedy one-to-one nearest match within ``tol`` bins -> (P, R, F1)."""
    pred_sorted = sorted(pred)
    true_sorted = sorted(true)
    used: set[int] = set()
    tp = 0
    for t in true_sorted:
        best, best_dist = None, tol + 1
        for idx, p in enumerate(pred_sorted):
            if idx in used:
                continue
            dist = abs(p - t)
            if dist <= tol and dist < best_dist:
                best_dist, best = dist, idx
        if best is not None:
            used.add(best)
            tp += 1
    precision = tp / len(pred_sorted) if pred_sorted else 0.0
    recall = tp / len(true_sorted) if true_sorted else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0
