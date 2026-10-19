"""Laminar TAD hierarchy: data structure, random root generation, and I/O.

A hierarchy is a forest of domains over the bins ``0 .. N-1``. Each domain
covers a half-open bin interval ``[start, end)`` (0-indexed). Children of a
domain are sorted, mutually disjoint, and strictly nested inside the parent.
That forest invariant is exactly the laminarity constraint the reconstruction
requires: any two domains are either nested or disjoint, never crossing.

OnTAD reports TADs as 1-indexed inclusive intervals with a nesting level. The
helpers here convert to that convention on write so ground-truth and OnTAD
calls live in the same coordinate system.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field


@dataclass
class Domain:
    """A single TAD: covers 0-indexed bins ``start .. end-1`` (``end`` exclusive)."""

    start: int
    end: int
    children: list["Domain"] = field(default_factory=list)

    @property
    def size(self) -> int:
        return self.end - self.start

    def copy(self) -> "Domain":
        return Domain(self.start, self.end, [c.copy() for c in self.children])

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "children": [c.to_dict() for c in self.children],
        }

    @staticmethod
    def from_dict(d: dict) -> "Domain":
        return Domain(
            int(d["start"]),
            int(d["end"]),
            [Domain.from_dict(c) for c in d.get("children", [])],
        )


# A hierarchy is just a list of top-level Domains (a forest).
Hierarchy = list


# ---------------------------------------------------------------------------
# Traversal / queries
# ---------------------------------------------------------------------------
def iter_domains(hier: list[Domain]):
    """Yield every domain in the forest, parents before children (pre-order)."""
    for d in hier:
        yield d
        yield from iter_domains(d.children)


def iter_with_level(hier: list[Domain], level: int = 1):
    """Yield ``(domain, level)`` with top-level domains at ``level`` (default 1)."""
    for d in hier:
        yield d, level
        yield from iter_with_level(d.children, level + 1)


def all_domains(hier: list[Domain]) -> list[Domain]:
    return list(iter_domains(hier))


def boundaries(hier: list[Domain]) -> set[int]:
    """Set of boundary bin positions (every domain start and end)."""
    b: set[int] = set()
    for d in iter_domains(hier):
        b.add(d.start)
        b.add(d.end)
    return b


def is_laminar(hier: list[Domain], n_bins: int) -> bool:
    """Validate the forest invariant over the whole hierarchy."""

    def check(domains: list[Domain], lo: int, hi: int) -> bool:
        prev = lo
        for d in domains:
            if not (lo <= d.start < d.end <= hi):
                return False
            if d.start < prev:  # overlap or out of order
                return False
            prev = d.end
            if not check(d.children, d.start, d.end):
                return False
        return True

    return check(hier, 0, n_bins)


# ---------------------------------------------------------------------------
# Random root-hierarchy generation
# ---------------------------------------------------------------------------
def _tile_region(
    lo: int,
    hi: int,
    target_n: int,
    min_size: int,
    rng: random.Random,
    gap_prob: float,
) -> list[Domain]:
    """Lay down up to ``target_n`` disjoint domains inside ``[lo, hi)``.

    Domains are separated by optional gaps (inter-TAD regions). Sizes are
    random but always ``>= min_size``, and enough room is reserved so the
    remaining domains can still fit.
    """
    domains: list[Domain] = []
    cursor = lo
    while cursor + min_size <= hi and len(domains) < target_n:
        space = hi - cursor
        slots_left = target_n - len(domains)

        # Maybe leave a gap before this domain.
        if rng.random() < gap_prob and space > 2 * min_size:
            cursor += rng.randint(1, min_size)
            space = hi - cursor
            if space < min_size:
                break

        # Largest size that still leaves min_size for every remaining slot.
        reserved = (slots_left - 1) * min_size
        max_dom = space - reserved if slots_left > 1 else space
        max_dom = max(min_size, min(max_dom, space))
        size = rng.randint(min_size, max_dom)
        domains.append(Domain(cursor, cursor + size))
        cursor += size
    return domains


def _subdivide(
    d: Domain,
    depth_left: int,
    min_size: int,
    rng: random.Random,
    split_prob: float,
    gap_prob: float,
) -> None:
    """Recursively nest child domains inside ``d`` up to ``depth_left`` levels."""
    if depth_left <= 0 or d.size < 2 * min_size:
        return
    if rng.random() > split_prob:
        return
    n_children = rng.randint(2, 3)
    kids = _tile_region(d.start, d.end, n_children, min_size, rng, gap_prob=gap_prob)
    if len(kids) < 2:
        return
    d.children = kids
    for k in kids:
        _subdivide(k, depth_left - 1, min_size, rng, split_prob * 0.8, gap_prob)


def generate_root(
    n_bins: int,
    n_top_tads: int,
    nesting_depth: int,
    min_size: int,
    rng: random.Random,
    split_prob: float = 0.6,
    gap_prob: float = 0.15,
) -> list[Domain]:
    """Generate a random laminar root hierarchy over ``0 .. n_bins-1``.

    ``gap_prob`` controls how often an inter-TAD gap is left between domains;
    keep it low so domains mostly tile the region, as real TADs do (that also
    keeps OnTAD calls clean and stable across the tree).
    """
    top = _tile_region(0, n_bins, n_top_tads, min_size, rng, gap_prob=gap_prob)
    for d in top:
        _subdivide(d, nesting_depth - 1, min_size, rng, split_prob, gap_prob * 0.5)
    assert is_laminar(top, n_bins), "root generation produced a non-laminar hierarchy"
    return top


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def hierarchy_to_json(hier: list[Domain]) -> list[dict]:
    return [d.to_dict() for d in hier]


def hierarchy_from_json(obj: list[dict]) -> list[Domain]:
    return [Domain.from_dict(d) for d in obj]


def write_tad(hier: list[Domain], n_bins: int, path) -> None:
    """Write the hierarchy in OnTAD ``.tad`` layout (1-indexed inclusive).

    Line 0 is the whole region at level 0 (as OnTAD emits), followed by every
    domain as ``start<TAB>end<TAB>level``. A 0-indexed half-open ``[s, e)``
    domain becomes 1-indexed inclusive ``[s+1, e]``.
    """
    lines = [f"1\t{n_bins}\t0"]
    for d, level in iter_with_level(hier, level=1):
        lines.append(f"{d.start + 1}\t{d.end}\t{level}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def save_hierarchy(hier: list[Domain], path) -> None:
    with open(path, "w") as fh:
        json.dump(hierarchy_to_json(hier), fh, indent=2)


def load_hierarchy(path) -> list[Domain]:
    with open(path) as fh:
        return hierarchy_from_json(json.load(fh))
