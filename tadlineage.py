#!/usr/bin/env python3
"""TADLineage command-line interface.

One entry point for the three problems of the paper, on any input:

  small   Small parsimony: given a cell differentiation tree and the TAD hierarchies
          of its observed cell states, infer the TAD hierarchies of the unobserved
          ones (integer linear program, Gurobi).
  median  Median: one TAD hierarchy minimizing the total TAD distance to a set of
          hierarchies (polynomial-time dynamic program).
  large   Large parsimony: given only the observed TAD hierarchies, infer the tree
          and the hierarchies of its internal vertices (integer linear program,
          Gurobi).

TAD files have one TAD per line, "start end ..." in 1-based bin indices (OnTAD .tad
output works as-is; extra columns are ignored). Every input must use the same
chromosome and bin size. Outputs are .tad files ("start<TAB>end") plus summary.json.

Examples:
  python tadlineage.py small --tree tree.txt --tads LT=LT.tad CLP=CLP.tad MK=MK.tad G=G.tad --tolerance 1 --out out_small
  python tadlineage.py median --tads cell1.tad cell2.tad cell3.tad --out out_median
  python tadlineage.py large --tads LT=LT.tad CLP=CLP.tad MK=MK.tad G=G.tad --tolerance 1 --out out_large
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "hemato_cells" / "large_parsimony"))

from filtering import harmonize_leaf_boundaries  # noqa: E402
from tad_utilities import parse_interval, read_tad_intervals, write_tads_to_file  # noqa: E402


# ----------------------------------------------------------------------------- input

def default_names(paths):
    """Names for bare paths: the file stem if those are unique, otherwise the last
    path component that differs (MEP/chr19/1mb/tads.tad -> MEP)."""
    parts = [Path(p).with_suffix("").parts for p in paths]
    if len({p[-1] for p in parts}) == len(parts):
        return [p[-1] for p in parts]
    common = 0
    while all(len(p) > common + 1 for p in parts) and len({p[-1 - common] for p in parts}) == 1:
        common += 1
    names = [p[-1 - common] for p in parts]
    return names if len(set(names)) == len(names) else ["_".join(p) for p in parts]


def parse_named_tads(items, require_names=True):
    """['NAME=path', ...] (or bare paths) -> {name: path}, keeping order."""
    bare = [item for item in items if "=" not in item]
    if bare and require_names:
        raise SystemExit(f"--tads entries must be NAME=path, got {bare[0]!r}")
    auto = iter(default_names(bare))
    named = {}
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path, name = item, next(auto)
        if not Path(path).is_file():
            raise SystemExit(f"TAD file not found: {path}")
        if name in named:
            raise SystemExit(f"Duplicate name {name!r} in --tads")
        named[name] = path
    return named


def load_hierarchies(named_paths, tolerance, workdir):
    """Read every TAD file; with a tolerance, first harmonize the boundaries across
    all of them (filtering.harmonize_leaf_boundaries) on copies inside workdir."""
    if tolerance is None:
        return {name: read_tad_intervals(path) for name, path in named_paths.items()}
    hdir = Path(workdir) / "harmonized"
    hdir.mkdir(parents=True, exist_ok=True)
    copies = {}
    for name, path in named_paths.items():
        dst = hdir / f"{name}.tad"
        shutil.copyfile(path, dst)
        copies[name] = dst
    harmonize_leaf_boundaries([str(p) for p in copies.values()], tolerance)
    return {name: read_tad_intervals(str(p.with_name(p.stem + "_filtered.tad")))
            for name, p in copies.items()}


def read_tree(path):
    """Edge list, one 'parent child' pair per line (# comments allowed)."""
    edges = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 2:
            raise SystemExit(f"Tree line must be 'parent child': {line!r}")
        edges.append((parts[0], parts[1]))
    children = {v for _, v in edges}
    roots = {u for u, _ in edges} - children
    if len(roots) != 1:
        raise SystemExit(f"Tree must have exactly one root, found {sorted(roots)}")
    if len(children) != len(edges):
        raise SystemExit("Every vertex must have at most one parent.")
    return edges, roots.pop()


# ----------------------------------------------------------------------------- output

def as_tuples(tads):
    return {parse_interval(t) if isinstance(t, str) else tuple(t) for t in tads}


def distance_terms(tads_u, tads_v):
    """(TAD changes, boundary changes) between two hierarchies."""
    tu, tv = as_tuples(tads_u), as_tuples(tads_v)
    starts = {a for a, _ in tu} ^ {a for a, _ in tv}
    ends = {b for _, b in tu} ^ {b for _, b in tv}
    return len(tu ^ tv), len(starts) + len(ends)


def edge_report(edges, tads):
    report = []
    for u, v in edges:
        t, b = distance_terms(tads[u], tads[v])
        report.append({"parent": u, "child": v, "tad_changes": t, "boundary_changes": b})
    return report


def write_outputs(out, tads, inferred, summary):
    out.mkdir(parents=True, exist_ok=True)
    for name in inferred:
        write_tads_to_file(as_tuples(tads[name]), str(out / f"{name}.tad"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {len(inferred)} inferred hierarchies and summary.json to {out}/")


def to_newick(edges, root):
    children = defaultdict(list)
    for u, v in edges:
        children[u].append(v)

    def rec(v):
        if not children[v]:
            return str(v)
        return "(" + ",".join(rec(c) for c in children[v]) + ")" + str(v)

    return rec(root) + ";"


# ----------------------------------------------------------------------------- commands

def cmd_small(args):
    from check_lp_constrained import solve_lp_from_edges_and_leaves

    edges, root = read_tree(args.tree)
    vertices = {u for e in edges for u in e}
    named = parse_named_tads(args.tads)
    unknown = set(named) - vertices
    if unknown:
        raise SystemExit(f"--tads names not in the tree: {sorted(unknown)}")
    leaves = vertices - {u for u, _ in edges}
    missing_leaves = leaves - set(named)
    if missing_leaves:
        raise SystemExit(f"Every leaf needs a TAD file; missing: {sorted(missing_leaves)}")

    out = Path(args.out)
    observed = load_hierarchies(named, args.tolerance, out)
    ids = {name: i for i, name in enumerate(sorted(vertices, key=lambda x: (x != root, x)))}
    names = {i: name for name, i in ids.items()}
    id_edges = [(ids[u], ids[v]) for u, v in edges]
    fixed = {ids[n]: tads for n, tads in observed.items() if n != root}
    root_tads = observed.get(root)

    print(f"Small parsimony: {len(vertices)} vertices, observed {sorted(observed)}, "
          f"inferring {sorted(vertices - set(observed))}")
    cost, integral, assign, *_ = solve_lp_from_edges_and_leaves(
        id_edges, fixed, Ct=args.Ct, Cb=args.Cb, mode=args.mode,
        require_laminar_internal=True, root_intervals=root_tads, verbose=args.verbose)
    if cost is None:
        raise SystemExit("The solver found no solution.")

    tads = {n: as_tuples(t) for n, t in observed.items()}
    for i, t in assign.items():
        if names[i] not in observed:
            tads[names[i]] = as_tuples(t)
    inferred = sorted(vertices - set(observed))
    summary = {
        "problem": "small_parsimony", "cost": cost, "integral": bool(integral),
        "Ct": args.Ct, "Cb": args.Cb, "mode": args.mode, "tolerance": args.tolerance,
        "root": root, "observed": sorted(observed), "inferred": inferred,
        "n_tads": {n: len(tads[n]) for n in sorted(tads)},
        "edges": edge_report(edges, tads),
    }
    print(f"Cost {cost:g}")
    write_outputs(out, tads, inferred, summary)


def cmd_median(args):
    from nussinov import infer_tads_with_dp

    named = parse_named_tads(args.tads, require_names=False)
    if len(named) < 2:
        raise SystemExit("The median needs at least 2 hierarchies.")
    out = Path(args.out)
    observed = load_hierarchies(named, args.tolerance, out)
    order = list(observed)
    edges = [(0, i + 1) for i in range(len(order))]
    leaf = {i + 1: observed[n] for i, n in enumerate(order)}
    # as in the clustering pipelines: every observed boundary position may be a
    # start or an end of a median TAD (the DP needs both to span the chromosome)
    positions = {p for t in leaf.values() for s in t for p in parse_interval(s)}
    starts = ends = positions

    print(f"Median of {len(order)} hierarchies")
    median, score = infer_tads_with_dp(edges, leaf, starts, ends, Ct=args.Ct, Cb=args.Cb)[:2]
    tads = {n: as_tuples(t) for n, t in observed.items()}
    tads[args.name] = as_tuples(median)
    star = [(args.name, n) for n in order]
    summary = {
        "problem": "median", "cost": score, "Ct": args.Ct, "Cb": args.Cb,
        "tolerance": args.tolerance, "inputs": order, "n_tads_median": len(tads[args.name]),
        "distances_to_median": edge_report(star, tads),
    }
    print(f"Cost {score:g}; median has {len(tads[args.name])} TADs")
    write_outputs(out, tads, [args.name], summary)


def cmd_large(args):
    from large_parsimony_ilp import LargeParsimonyILP

    named = parse_named_tads(args.tads)
    if len(named) < 3:
        raise SystemExit("Large parsimony needs at least 3 hierarchies.")
    out = Path(args.out)
    observed = load_hierarchies(named, args.tolerance, out)
    order = list(observed)
    k = len(order)
    root_leaf = order.index(args.outgroup) + 1 if args.outgroup else 1
    if args.outgroup and args.outgroup not in observed:
        raise SystemExit(f"--outgroup {args.outgroup!r} is not one of the --tads names")

    print(f"Large parsimony on {k} hierarchies ({order}), rooted on the edge to {order[root_leaf - 1]}")
    ilp = LargeParsimonyILP([as_tuples(observed[n]) for n in order], Ct=args.Ct, Cb=args.Cb,
                            mode=args.mode, threads=args.threads, time_limit=args.time_limit,
                            root_leaf=root_leaf, verbose=args.verbose)
    res = ilp.solve()
    if res["status"] != "optimal":
        raise SystemExit(f"Solver stopped: {res['status']}")

    def vname(v):
        return order[v - 1] if v <= k else ("root" if v == 2 * k - 1 else f"N{v}")

    edges = [(vname(u), vname(v)) for u, v in sorted(res["arcs"], reverse=True)]
    tads = {n: as_tuples(observed[n]) for n in order}
    for v, t in res["tads"].items():
        if v > k:
            tads[vname(v)] = as_tuples(t)
    inferred = [vname(v) for v in range(2 * k - 1, k, -1)]
    newick = to_newick(edges, "root")
    summary = {
        "problem": "large_parsimony", "cost": res["cost"], "Ct": args.Ct, "Cb": args.Cb,
        "mode": args.mode, "tolerance": args.tolerance, "leaves": order,
        "outgroup": order[root_leaf - 1], "newick": newick, "inferred": inferred,
        "n_tads": {n: len(tads[n]) for n in tads}, "edges": edge_report(edges, tads),
        "seconds": res["seconds"],
    }
    (out / "tree.nwk").parent.mkdir(parents=True, exist_ok=True)
    (out / "tree.nwk").write_text(newick + "\n")
    print(f"Cost {res['cost']:g}; tree {newick}")
    write_outputs(out, tads, inferred, summary)


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, names_required=True):
        sp.add_argument("--tads", nargs="+", required=True,
                        help="TAD files, as NAME=path" + ("" if names_required else " or bare paths"))
        sp.add_argument("--out", required=True, help="output directory")
        sp.add_argument("--tolerance", type=int, default=None,
                        help="harmonize boundaries within this many bins across inputs first "
                             "(recommended 1-2; default: use the calls as given)")
        sp.add_argument("--Ct", type=float, default=1.0, help="cost of inserting/deleting a TAD")
        sp.add_argument("--Cb", type=float, default=1.0, help="cost of inserting/deleting a boundary")
        sp.add_argument("--verbose", action="store_true")

    s = sub.add_parser("small", help="small parsimony on a given tree")
    s.add_argument("--tree", required=True, help="edge list file, one 'parent child' per line")
    common(s)
    s.add_argument("--mode", default="convex", choices=["convex", "lex_boundary_first", "lex_tad_first"])
    s.set_defaults(func=cmd_small)

    m = sub.add_parser("median", help="median TAD hierarchy of a set of hierarchies")
    common(m, names_required=False)
    m.add_argument("--name", default="median", help="name of the output hierarchy")
    m.set_defaults(func=cmd_median)

    g = sub.add_parser("large", help="large parsimony: infer the tree and internal hierarchies")
    common(g)
    g.add_argument("--outgroup", help="leaf whose edge carries the root (does not change the cost)")
    g.add_argument("--mode", default="convex", choices=["convex", "lex_boundary_first", "lex_tad_first"])
    g.add_argument("--threads", type=int, default=None)
    g.add_argument("--time-limit", type=float, default=None, help="seconds")
    g.set_defaults(func=cmd_large)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
