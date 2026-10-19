#!/usr/bin/env python3
"""The TADs that reorganise along each edge of the haematopoietic tree.

One rule, applied to every edge: a TAD is **reorganised** if it is called at the parent but
not at the child, or at the child but not at the parent.  Everything the two nodes have in
common drops out.  Two calls count as the same TAD when both of their endpoints agree to
within ``--tolerance`` bins, which is what ``--tolerance`` means everywhere else in this
repository.

Nothing is typed as a split, a merge, a gain or a loss, and no boundary bins are carried
separately -- the symmetric difference is the whole definition.

Two callers can be read, and each brings its own tree:

    ontad      hemato_cells/<NODE>/chr<C>/<RES>/tads.tad                 10 nodes, 9 edges
    robustad   hemato_cells/using_robustad/<NODE>/chr<C>/<RES>/tads.tad   9 nodes, 8 edges

RobusTAD was never run on ST, so its tree goes LT -> MPP directly.  The two callers agree
on roughly 70% of their calls at 250 kb, so they are a genuine cross-check, not a copy.

Output, one file per chromosome:

    results/reorg/<caller>/<res>/tol_<N>/<chrom>.tsv     edge, chrom, side, start_bin, end_bin
    results/reorg/<caller>/<res>/tol_<N>/covered.tsv     chrom, start_bin, end_bin

``side`` is ``parent_only`` or ``child_only``; ``covered`` is the span any node calls at
all, which becomes the universe of the enrichment test.

Usage:
    python reorganized_tads.py --caller ontad --resolution 250kb --tolerance 3
    python reorganized_tads.py                        # both callers, the default grid
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEMATO = HERE.parent
REPO = HEMATO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tad_utilities import match_tads_within_tolerance  # noqa: E402

# parent -> child, down the stem and then each branch
EDGES_ONTAD = [
    ("LT", "ST"), ("ST", "MPP"), ("MPP", "CMP"), ("MPP", "CLP"),
    ("CMP", "MEP"), ("CMP", "GMP"), ("MEP", "MKP"), ("MKP", "MK"), ("GMP", "G"),
]
# RobusTAD has no ST call, so the stem is one edge shorter
EDGES_ROBUSTAD = [("LT", "MPP")] + EDGES_ONTAD[2:]

CALLERS = {
    "ontad": (lambda node: HEMATO / node, EDGES_ONTAD),
    "robustad": (lambda node: HEMATO / "using_robustad" / node, EDGES_ROBUSTAD),
}
CHROMS = [f"chr{i}" for i in range(1, 20)]
RESULTS = HERE / "results"
FIELDS = ["edge", "chrom", "side", "start_bin", "end_bin"]


def edges_for(caller: str):
    return CALLERS[caller][1]


def nodes_for(caller: str):
    return sorted({n for edge in edges_for(caller) for n in edge})


def load(caller: str, node: str, chrom: str, res: str) -> list[tuple[int, int]]:
    path = CALLERS[caller][0](node) / chrom / res / "tads.tad"
    if not path.exists():
        raise FileNotFoundError(f"Missing {caller} calls: {path}")
    with path.open() as fh:
        return sorted({(int(p[0]), int(p[1]))
                       for p in (line.split() for line in fh) if len(p) >= 2})


def reorganised(parent, child, tolerance):
    """The TADs of either node that the other does not also call, within tolerance."""
    _matched, parent_only, child_only = match_tads_within_tolerance(
        parent, child, tolerance=tolerance)
    return ([("parent_only",) + tuple(iv) for iv in parent_only]
            + [("child_only",) + tuple(iv) for iv in child_only])


def run_config(caller: str, resolution: str, tolerance: int, chroms) -> Counter:
    out_dir = RESULTS / "reorg" / caller / resolution / f"tol_{tolerance}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tally = Counter()
    covered = []

    for chrom in chroms:
        sets = {node: load(caller, node, chrom, resolution) for node in nodes_for(caller)}
        rows = []
        for parent, child in edges_for(caller):
            label = f"{parent}->{child}"
            for side, a, b in reorganised(sets[parent], sets[child], tolerance):
                rows.append({"edge": label, "chrom": chrom, "side": side,
                             "start_bin": a, "end_bin": b})
                tally[label] += 1
                tally[(label, "bins")] += b - a + 1
        with (out_dir / f"{chrom}.tsv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

        every = [iv for call in sets.values() for iv in call]
        covered.append({"chrom": chrom,
                        "start_bin": min(a for a, _ in every),
                        "end_bin": max(b for _, b in every)})

    with (out_dir / "covered.tsv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["chrom", "start_bin", "end_bin"],
                                delimiter="\t")
        writer.writeheader()
        writer.writerows(covered)
    return tally


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caller", action="append", choices=sorted(CALLERS))
    ap.add_argument("--resolution", action="append")
    ap.add_argument("--tolerance", action="append", type=int)
    ap.add_argument("--chrom", action="append")
    args = ap.parse_args()

    callers = args.caller or sorted(CALLERS)
    resolutions = args.resolution or ["100kb", "250kb", "500kb", "1mb"]
    tolerances = args.tolerance or [1, 2, 3, 5, 8, 10]
    chroms = args.chrom or CHROMS

    for caller in callers:
        for resolution in resolutions:
            for tolerance in tolerances:
                tally = run_config(caller, resolution, tolerance, chroms)
                print(f"\n=== {caller} {resolution} tol={tolerance} ===")
                print(f"{'edge':<12}{'reorganised TADs':>18}{'bins spanned':>14}")
                for parent, child in edges_for(caller):
                    label = f"{parent}->{child}"
                    print(f"{label:<12}{tally[label]:>18}{tally[(label, 'bins')]:>14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
