"""Parsimonious, laminarity-preserving evolution of a TAD hierarchy along an edge.

Each parent->child edge copies the parent hierarchy and applies a small number
of local events (the per-edge budget ``k``). Every operation preserves the
forest invariant by construction and reuses existing boundaries where possible,
so most structure is conserved from parent to child -- the parsimony the whole
experiment relies on.

Operations
----------
split  : [a,b] -> nested [a,x],[x,b]      (adds one interior boundary x)
merge  : two adjacent siblings -> one span (removes one interior boundary)
insert : new domain placed in a gap        (adds a domain)
delete : remove a domain, promote its kids (removes a domain)
shift  : move a shared boundary x -> x +/- d
"""

from __future__ import annotations

import random

from hierarchy import Domain, boundaries, is_laminar


# ---------------------------------------------------------------------------
# Structural iterators over the mutable forest
# ---------------------------------------------------------------------------
def _iter_slots(hier: list[Domain], lo: int, hi: int):
    """Yield ``(container_list, index, domain, lo, hi)`` for every domain.

    ``container_list`` is the actual list holding ``domain`` (a parent's
    ``children`` or the top-level forest); ``lo``/``hi`` are the bounds of the
    region that list tiles.
    """
    for idx, d in enumerate(hier):
        yield hier, idx, d, lo, hi
        yield from _iter_slots(d.children, d.start, d.end)


def _iter_containers(hier: list[Domain], lo: int, hi: int, level: int = 1):
    """Yield ``(container_list, lo, hi, level)`` for every domain-list.

    ``level`` is the nesting level of the domains held by that list (top-level
    list = 1). Used to cap how deep new domains may be inserted.
    """
    yield hier, lo, hi, level
    for d in hier:
        yield from _iter_containers(d.children, d.start, d.end, level + 1)


# ---------------------------------------------------------------------------
# Operations. Each mutates ``hier`` in place and returns True on success.
# ---------------------------------------------------------------------------
def op_split(hier: list[Domain], params, rng: random.Random) -> bool:
    """Insert one interior boundary: replace [a,b] with siblings [a,x],[x,b].

    This is the inverse of op_merge and keeps nesting depth bounded (it never
    deepens the hierarchy). Any children of the split domain are partitioned to
    the side that contains them; ``x`` is therefore never placed strictly inside
    a child, which also keeps the result laminar.
    """
    ms = params.min_size
    slots = [
        (container, idx, d)
        for container, idx, d, _, _ in _iter_slots(hier, 0, params.n_bins)
        if d.size >= 2 * ms
    ]
    rng.shuffle(slots)
    global_bounds = boundaries(hier)
    for container, idx, d in slots:
        # Boundary positions that leave both halves >= ms and don't cut a child.
        allowed = set(range(d.start + ms, d.end - ms + 1))
        for c in d.children:
            allowed -= set(range(c.start + 1, c.end))  # strictly interior to a child
        if not allowed:
            continue
        reuse = [x for x in allowed if x in global_bounds]
        x = rng.choice(reuse) if reuse and rng.random() < 0.5 else rng.choice(sorted(allowed))
        left_kids = [c for c in d.children if c.end <= x]
        right_kids = [c for c in d.children if c.start >= x]
        container[idx : idx + 1] = [
            Domain(d.start, x, left_kids),
            Domain(x, d.end, right_kids),
        ]
        return True
    return False


def op_merge(hier: list[Domain], params, rng: random.Random) -> bool:
    # Collect adjacent sibling pairs (end_i == start_{i+1}) anywhere in the forest.
    pairs = []
    for container, _, _, _ in _iter_containers(hier, 0, params.n_bins):
        for i in range(len(container) - 1):
            if container[i].end == container[i + 1].start:
                pairs.append((container, i))
    if not pairs:
        return False
    container, i = rng.choice(pairs)
    left, right = container[i], container[i + 1]
    merged = Domain(left.start, right.end, left.children + right.children)
    container[i : i + 2] = [merged]
    return True


def op_insert(hier: list[Domain], params, rng: random.Random) -> bool:
    ms = params.min_size
    max_level = params.nesting_depth
    # Find gaps large enough for a new domain, across every container, without
    # inserting deeper than the configured nesting depth.
    candidates = []  # (container, insert_index, gap_lo, gap_hi)
    for container, lo, hi, level in _iter_containers(hier, 0, params.n_bins):
        if level > max_level:
            continue
        cursor = lo
        for idx, d in enumerate(container):
            if d.start - cursor >= ms:
                candidates.append((container, idx, cursor, d.start))
            cursor = d.end
        if hi - cursor >= ms:
            candidates.append((container, len(container), cursor, hi))
    if not candidates:
        return False
    container, idx, glo, ghi = rng.choice(candidates)
    max_size = ghi - glo
    size = rng.randint(ms, max_size)
    start = rng.randint(glo, ghi - size)
    container.insert(idx, Domain(start, start + size))
    return True


def op_delete(hier: list[Domain], params, rng: random.Random) -> bool:
    slots = list(_iter_slots(hier, 0, params.n_bins))
    if not slots:
        return False
    container, idx, d, _, _ = rng.choice(slots)
    # Promote children into the domain's position (keeps them, drops d's span).
    container[idx : idx + 1] = d.children
    return True


def op_shift(hier: list[Domain], params, rng: random.Random) -> bool:
    ms = params.min_size
    delta = params.shift_delta
    pairs = []
    for container, _, _, _ in _iter_containers(hier, 0, params.n_bins):
        for i in range(len(container) - 1):
            if container[i].end == container[i + 1].start:
                pairs.append((container, i))
    if not pairs:
        return False
    rng.shuffle(pairs)
    for container, i in pairs:
        left, right = container[i], container[i + 1]
        x = left.end
        left_child_max = max((c.end for c in left.children), default=left.start)
        right_child_min = min((c.start for c in right.children), default=right.end)
        # Feasible window for the new boundary.
        new_lo = max(left.start + ms, left_child_max, x - delta)
        new_hi = min(right.end - ms, right_child_min, x + delta)
        choices = [v for v in range(new_lo, new_hi + 1) if v != x]
        if not choices:
            continue
        new_x = rng.choice(choices)
        left.end = new_x
        right.start = new_x
        return True
    return False


_OPS = [
    (op_split, 0.25),
    (op_merge, 0.20),
    (op_insert, 0.15),
    (op_delete, 0.15),
    (op_shift, 0.25),
]


def evolve(parent_hier: list[Domain], k: int, params, rng: random.Random) -> list[Domain]:
    """Return a child hierarchy: a copy of ``parent_hier`` after ``k`` events."""
    hier = [d.copy() for d in parent_hier]
    ops, weights = zip(*_OPS)
    applied = 0
    attempts = 0
    max_attempts = max(50, k * 50)
    while applied < k and attempts < max_attempts:
        op = rng.choices(ops, weights=weights, k=1)[0]
        if op(hier, params, rng):
            applied += 1
        attempts += 1
    assert is_laminar(hier, params.n_bins), "evolution produced a non-laminar hierarchy"
    return hier
