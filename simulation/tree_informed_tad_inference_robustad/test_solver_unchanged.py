#!/usr/bin/env python3
"""Regression: the edited weighted solver must be IDENTICAL when impute=False.

This experiment added ``impute_cost``, ``candidate_intervals`` and an
impute-only branch in the laminarity cuts to the SHARED
``check_lp_constrained_weighted_updated.py``, which ``hemato_cells/`` and
``simulation/using_robustad/`` also solve with. Those pipelines must be
bit-identical afterwards, so this asserts it rather than assuming it.

Run (gurobi env), from anywhere:
    python test_solver_unchanged.py
Exits non-zero on any mismatch. Compares against ``HEAD``, so run it before
committing the solver change -- once committed, HEAD is the edited version and
the test compares it against itself.

Loads the committed (pre-edit) version of check_lp_constrained_weighted_updated
from git alongside the working-tree version, and solves the same problems with
both -- with and without leaf_importance, with and without a fixed root -- then
compares cost and every node's assignment exactly.
"""
from __future__ import annotations

import importlib.util
import random
import subprocess
import sys
from pathlib import Path

REPO = Path("/Users/PSLabMonisha/Desktop/tads")
SCRATCH = Path(__file__).resolve().parent / ".regress"
SCRATCH.mkdir(exist_ok=True)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


old_src = subprocess.run(
    ["git", "-C", str(REPO), "show", "HEAD:check_lp_constrained_weighted_updated.py"],
    capture_output=True, text=True, check=True).stdout
old_path = SCRATCH / "_old_weighted.py"
old_path.write_text(old_src)

OLD = load("old_weighted", old_path)
NEW = load("new_weighted", REPO / "check_lp_constrained_weighted_updated.py")


def make_case(seed: int):
    """A random 7-node tree with random leaf interval sets."""
    rng = random.Random(seed)
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (2, 5), (3, 6)]
    leaves = [4, 5, 6]
    leaf_intervals, importance = {}, {}
    for v in leaves:
        ivs = set()
        while len(ivs) < rng.randint(3, 6):
            a = rng.randrange(0, 40)
            b = a + rng.randint(3, 20)
            ivs.add((a, b))
        leaf_intervals[v] = {f"[{a},{b}]" for (a, b) in ivs}
        importance[v] = {
            (a, b): {"x": rng.random(), "s": rng.random(), "e": rng.random()}
            for (a, b) in ivs
        }
    root_ivs = sorted(leaf_intervals[leaves[0]])[:2]
    return edges, leaf_intervals, importance, root_ivs


def norm(result):
    cost, integral, assign, _x, nodes, intervals = result
    cost = None if cost is None else round(float(cost), 9)
    assign = {k: sorted(map(tuple, v)) for k, v in (assign or {}).items()}
    return cost, integral, assign, list(nodes), sorted(map(tuple, intervals or []))


fails = 0
for seed in range(12):
    edges, leaf_intervals, importance, root_ivs = make_case(seed)
    variants = [
        ("plain", dict()),
        ("importance", dict(leaf_importance=importance, leaf_importance_weight=0.3)),
        ("root-fixed", dict(root_intervals=root_ivs)),
        ("importance+root", dict(leaf_importance=importance, leaf_importance_weight=0.1,
                                 root_intervals=root_ivs)),
        ("fitch-hartigan", dict(use_fitch_hartigan=True)),
        ("lex_tad_first", dict(mode="lex_tad_first")),
        ("lex_boundary_first", dict(mode="lex_boundary_first")),
    ]
    for label, kwargs in variants:
        a = norm(OLD.solve_lp_from_edges_and_leaves(edges, leaf_intervals, **kwargs))
        b = norm(NEW.solve_lp_from_edges_and_leaves(edges, leaf_intervals, **kwargs))
        if a != b:
            fails += 1
            print(f"MISMATCH seed={seed} variant={label}")
            print("  old cost/assign:", a[0], a[2])
            print("  new cost/assign:", b[0], b[2])

print(f"\n{'FAILED' if fails else 'PASS'}: {12 * 7} non-imputation solves compared, "
      f"{fails} mismatch(es)")

# And the new-only path actually does something different when asked.
edges, leaf_intervals, importance, _ = make_case(0)
base = norm(NEW.solve_lp_from_edges_and_leaves(edges, leaf_intervals))
imp = norm(NEW.solve_lp_from_edges_and_leaves(
    edges, leaf_intervals, impute=True, impute_cost=1.0))
print("impute=True changes the leaf assignments:",
      any(base[2][v] != imp[2][v] for v in (4, 5, 6)))
sys.exit(1 if fails else 0)
