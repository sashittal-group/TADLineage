#!/usr/bin/env python3
"""Reading the per-cell TAD calls the single-cell clusterings were built from.

One helper, kept apart from any analysis so that the analyses can come and go without
taking it with them. A clustering run at one resolution and chromosome uses a few dozen
cells, but the per-cell call tree holds several hundred, so everything a run did not touch
is available as a held-out set.

Layout is ``<root>/<stage>/<cell>/<chrom>/<resolution>/tads_filtered.tad``, and a domain
``[a, b]`` contributes both ``a`` and ``b`` as boundary bins, which is the convention the
rest of the folder uses.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

STAGE_ORDER = ["1-cell", "2-cell", "4-cell", "8-cell", "64-cell"]
CHROMS = [f"chr{i}" for i in range(1, 20)] + ["chrX"]

# the per-cell calls, and the clustering runs that drew from them. The two callers are
# separate pipelines end to end: different TAD calls, different clusterings, different cell
# pools. Nothing here compares their positions -- each is validated against its own
# held-out cells, which is the whole point of running both.
SOURCES = {
    "RobusTAD": {
        "events": Path(__file__).resolve().parent / "results" / "events",
        "runs": REPO_ROOT / "scptr_robustad" / "results_updated" / "scptr_runs_all_cells",
        "pool": REPO_ROOT / "scptr_robustad" / "cell_differentiation_robustad_ontad",
    },
    "OnTAD": {
        "events": Path(__file__).resolve().parent / "results" / "events_ontad",
        "runs": REPO_ROOT / "scptr_ontad" / "results_updated" / "scptr_runs_all_cells",
        "pool": REPO_ROOT / "scptr" / "cell_differentiation",
    },
}


def read_cells(root, stage, chrom, resolution):
    """``{cell: sorted boundary bins}`` from the per-cell TAD calls of one stage."""
    base = Path(root) / stage
    if not base.is_dir():
        return {}
    cells = {}
    for entry in sorted(base.iterdir()):
        path = entry / chrom / str(resolution) / "tads_filtered.tad"
        if not path.is_file():
            continue
        bins = set()
        for line in open(path):
            fields = line.split()
            if len(fields) < 2:
                continue
            try:
                a, b = int(fields[0]), int(fields[1])
            except ValueError:
                continue
            bins.add(a)
            bins.add(b)
        if bins:
            cells[entry.name] = sorted(bins)
    return cells
