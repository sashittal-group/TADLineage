"""Randomly generated phylogenetic trees for the data-quality sweep.

You fix a **total node count** ``n_nodes``; the generator produces distinct
random rooted trees of that size, each following the same generative pattern as
the fixed experiment (start at the root, evolve down). The topology constraint
is: **every internal node except the root has at least 2 children** (the root is
a single-child stem -- ``root -> n0`` -- exactly as in the original tree, so the
root stays a pure observed anchor and ``n0`` is the top scored node).

Node roles are derived from the topology alone, so every generated tree works:

  * given   = root + leaves           (reconstruction inputs)
  * scored  = internal non-root nodes (reconstructed and evaluated)

Trees are just a replication axis: results are pooled across them. A tree's
topology is a deterministic function of ``(name, n_nodes, seed)``, so the CALL
and SCORE stages (separate processes) reconstruct the identical tree.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

MAX_ARITY = 4          # cap children per internal node, for readable trees
MIN_NODES = 4          # fewest nodes that still yield one scored internal node


@dataclass(frozen=True)
class Tree:
    """A rooted tree plus the role derivations the sweep needs."""

    name: str
    root: str
    edges: dict[str, tuple[str, ...]]

    @property
    def nodes(self) -> list[str]:
        """Top-down (BFS) node order; parents always precede their children."""
        order = [self.root]
        i = 0
        while i < len(order):
            order.extend(self.edges.get(order[i], ()))
            i += 1
        return order

    @property
    def parent(self) -> dict[str, str]:
        p: dict[str, str] = {}
        for u, kids in self.edges.items():
            for k in kids:
                p[k] = u
        return p

    @property
    def leaves(self) -> list[str]:
        return [n for n in self.nodes if not self.edges.get(n)]

    @property
    def internal(self) -> list[str]:
        leaves = set(self.leaves)
        return [n for n in self.nodes if n != self.root and n not in leaves]

    @property
    def given(self) -> list[str]:
        """Nodes supplied to reconstruction (observed): root + leaves."""
        return [self.root] + self.leaves

    @property
    def scored(self) -> list[str]:
        """Nodes reconstructed and scored: every internal non-root node."""
        return self.internal

    @property
    def node_id(self) -> dict[str, int]:
        """Stable name -> integer id map (the solvers work in integer ids)."""
        return {n: i for i, n in enumerate(self.nodes)}

    @property
    def edge_ids(self) -> list[tuple[int, int]]:
        nid = self.node_id
        return [(nid[u], nid[v]) for u, kids in self.edges.items() for v in kids]


def random_tree(name: str, n_nodes: int, rng: random.Random, max_arity: int = MAX_ARITY) -> Tree:
    """Grow a random rooted tree with exactly ``n_nodes`` nodes.

    Invariant maintained throughout: every internal node has >= 2 children. The
    root is a single-child stem (``root -> n0``), the one node exempt from the
    >= 2 rule -- so ``n0`` is the first branch point and the top scored node.

    Growth: start from ``root -> n0`` (n0 a leaf), then repeatedly turn a random
    leaf into an internal node with ``j`` (2..max_arity) fresh leaf children,
    picking ``j`` so the remaining budget can always be finished (never leaving a
    remainder of 1, which is impossible to place under the >= 2 rule).
    """
    if n_nodes < MIN_NODES:
        raise ValueError(f"n_nodes must be >= {MIN_NODES} (got {n_nodes})")

    edges: dict[str, tuple[str, ...]] = {"root": ("n0",)}
    leaves = ["n0"]
    count = 2                      # root + n0
    counter = 1                    # next free "n{counter}"

    while count < n_nodes:
        remaining = n_nodes - count
        leaf = leaves.pop(rng.randrange(len(leaves)))
        # Feasible child counts: j == remaining (finishes exactly) or j <= remaining-2
        # (leaves a placeable >= 2 remainder). Cap at max_arity.
        hi = min(max_arity, remaining)
        choices = [j for j in range(2, hi + 1) if j == remaining or j <= remaining - 2]
        if not choices:            # only when remaining == 2 and 2 > max_arity cap; keep safe
            choices = [remaining]
        j = rng.choice(choices)
        kids = tuple(f"n{counter + k}" for k in range(j))
        counter += j
        edges[leaf] = kids
        leaves.extend(kids)
        count += j

    return Tree(name, "root", edges)


def get_tree(name: str, n_nodes: int, seed: int) -> Tree:
    """Deterministically rebuild tree ``name`` (== ``tree<index>``) of ``n_nodes``.

    The index parsed from the name seeds the RNG, so the same ``name`` always
    yields the same topology across processes (CALL and SCORE must agree).
    """
    index = int("".join(ch for ch in name if ch.isdigit()) or "0")
    rng = random.Random(seed * 100_003 + index)
    return random_tree(name, n_nodes, rng)


def tree_names(n_trees: int) -> list[str]:
    return [f"tree{i:02d}" for i in range(n_trees)]
