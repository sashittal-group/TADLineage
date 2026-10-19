#!/usr/bin/env python3
"""Collect the scHiCPTR runs into ``results/figure/ari_summary.tsv`` and draw the ARI heatmap.

The TSV carries the schema the other baselines write, so
``scptr_robustad/plot_ari_comparison.py`` reads all three the same way:

    kind    resolution    chromosome    ari

``kind`` is the clustering that produced the row. Three of them are exactly the three that
Fig. 2b of the scHiCPTR paper scores its own embedding under: ``louvain`` at the shipped
resolution, which is what Section 3.4 uses on these cells, and ``kmeans`` and ``spectral``, both
given the true number of stages. ``louvain5`` is the fourth: Louvain again, at the resolution
whose partition has exactly five communities, which is the count the accuracy the scHiCPTR paper
reports must have come from. The comparison figure takes ``cluster``, an alias for whichever of
the four scored highest on that run, so scHiCPTR is compared at its best.

Usage:
    python plot_ari_summaries.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESOLUTIONS = ["250kb", "500kb", "1000kb"]
RESOLUTION_LABEL = {"250kb": "250 kb", "500kb": "500 kb", "1000kb": "1 Mb"}
CHROMS = [f"chr{i}" for i in range(1, 20)] + ["chrX"]
METHODS = ["louvain", "louvain5", "kmeans", "spectral"]


def collect() -> list[dict]:
    """One row per (clustering, bin size, chromosome), plus the ``cluster`` alias rows."""
    rows: list[dict] = []
    best: dict[tuple[str, str], dict] = {}
    for resolution in RESOLUTIONS:
        for chrom in ["all"] + CHROMS:
            for method in METHODS:
                path = RESULTS / resolution / chrom / method / "schicptr_results.json"
                if not path.exists():
                    continue
                summary = json.loads(path.read_text())
                row = {"kind": method, "resolution": resolution, "chromosome": chrom,
                       "ari": summary["ARI"], "nmi": summary["NMI"], "acc": summary["ACC"],
                       "n_clusters": summary["n_clusters"]}
                rows.append(row)
                key = (resolution, chrom)
                if key not in best or row["ari"] > best[key]["ari"]:
                    best[key] = row
    for (resolution, chrom), row in best.items():
        rows.append({**row, "kind": "cluster"})
    return rows


def write_tsv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    order = {c: i for i, c in enumerate(["all"] + CHROMS)}
    rows = sorted(rows, key=lambda r: (r["kind"], RESOLUTIONS.index(r["resolution"]),
                                       order[r["chromosome"]]))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t",
                                fieldnames=["kind", "resolution", "chromosome", "ari", "nmi",
                                            "acc", "n_clusters"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "ari": f"{row['ari']:.6f}", "nmi": f"{row['nmi']:.6f}",
                             "acc": f"{row['acc']:.6f}"})


def plot(rows: list[dict], path: Path) -> None:
    """ARI of every run as a bin size by chromosome grid, one panel per clustering."""
    columns = ["all"] + CHROMS
    fig, axes = plt.subplots(len(METHODS), 1, figsize=(13, 2.0 * len(METHODS) + 1.4),
                             constrained_layout=True)
    for ax, method in zip(np.atleast_1d(axes), METHODS):
        grid = np.full((len(RESOLUTIONS), len(columns)), np.nan)
        for row in rows:
            if row["kind"] != method:
                continue
            grid[RESOLUTIONS.index(row["resolution"]), columns.index(row["chromosome"])] = row["ari"]
        image = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(["all"] + [c.replace("chr", "") for c in CHROMS], fontsize=8)
        ax.set_yticks(range(len(RESOLUTIONS)))
        ax.set_yticklabels([RESOLUTION_LABEL[r] for r in RESOLUTIONS], fontsize=8)
        ax.set_title(f"scHiCPTR embedding, {method} clustering", fontsize=10)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if grid[i, j] < 0.6 else "black")
        fig.colorbar(image, ax=ax, label="ARI", fraction=0.02)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    rows = collect()
    if not rows:
        raise SystemExit(f"No scHiCPTR results found under {RESULTS}")
    out = RESULTS / "figure"
    write_tsv(rows, out / "ari_summary.tsv")
    plot(rows, out / "ari_summary.png")
    for method in METHODS + ["cluster"]:
        for resolution in RESOLUTIONS:
            pooled = [r for r in rows if r["kind"] == method
                      and r["resolution"] == resolution and r["chromosome"] == "all"]
            per_chrom = [r["ari"] for r in rows if r["kind"] == method
                         and r["resolution"] == resolution and r["chromosome"] != "all"]
            if not pooled and not per_chrom:
                continue
            genome = f"{pooled[0]['ari']:.3f}" if pooled else "-"
            mean = f"{np.mean(per_chrom):.3f} over {len(per_chrom)}" if per_chrom else "-"
            print(f"{method:8s} {resolution:7s} genome-wide ARI {genome}, "
                  f"mean single-chromosome ARI {mean}")
    print(f"\nwrote {out / 'ari_summary.tsv'} and {out / 'ari_summary.png'}")


if __name__ == "__main__":
    main()
