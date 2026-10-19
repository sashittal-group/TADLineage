#!/usr/bin/env python3
"""Whether positional tolerance rescues the boundary signal, in three panels.

a. CTCF density as a function of distance from the call. This is the panel that decides the
   question: a boundary blurred by sparse coverage would show a hump at the centre, or at
   worst a flat line over the width of the blur. A sharp minimum at offset 0 means the
   calls are positionally precise and land on CTCF-poor territory on purpose.
b. What a tolerance does to the CTCF fold. Every curve climbs toward 1 and none crosses it,
   which is what dilution looks like: a wider window is a window closer to the genome
   average.
c. What a tolerance does to reproducibility across bin sizes, the one place it helps.
d. The other way sparsity could act, and the one that leaves a mark: CTCF density against
   bin size. Contacts per bin rise fourfold from 250 kb to 1 Mb, and the depletion goes
   away along that axis, so it is about how reliably a boundary can be placed rather than
   about the genome.

Usage:
    python plot_tolerance.py
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from summarise_results import read_tsv  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SETS = ["boundary_gained", "boundary_lost", "boundary_stable", "split_boundary",
        "merge_boundary"]
COLORS = {"boundary_gained": "#4c72b0", "boundary_lost": "#dd8452",
          "boundary_stable": "#55a868", "split_boundary": "#c44e52",
          "merge_boundary": "#8172b3"}
LABEL = {n: n.replace("boundary_", "").replace("_boundary", "") for n in SETS}


def panel_profile(ax):
    rows = read_tsv(RESULTS / "profile" / "boundary_profile.tsv")
    for name in SETS:
        by_offset = collections.defaultdict(list)
        for row in rows:
            if row["feature_set"] == name and row["ctcf_fold"]:
                by_offset[int(row["offset_bins"])].append(float(row["ctcf_fold"]))
        offsets = sorted(by_offset)
        ax.plot(offsets, [np.mean(by_offset[o]) for o in offsets], marker="o",
                markersize=3, linewidth=1.4, color=COLORS[name], label=LABEL[name])
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("bins from the called boundary")
    ax.set_ylabel("CTCF density vs\ncallable-region mean")
    ax.set_title("a  The dip is centred on the call,\nso the calls are not jittered",
                 loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False)


def panel_flank(ax):
    rows = read_tsv(RESULTS / "tolerance" / "ctcf_flank_sweep.tsv")
    for name in SETS:
        by_flank = collections.defaultdict(list)
        for row in rows:
            if row["feature_set"] == name and row["fold"]:
                by_flank[int(row["flank_bins"])].append(float(row["fold"]))
        flanks = sorted(by_flank)
        ax.plot(flanks, [np.mean(by_flank[f]) for f in flanks], marker="o",
                markersize=4, color=COLORS[name], label=LABEL[name])
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("tolerance applied (± bins)")
    ax.set_ylabel("CTCF density vs\nrelocated windows")
    ax.set_ylim(0.7, 1.05)
    ax.set_title("b  Tolerance dilutes toward 1,\nit does not cross it", loc="left",
                 fontsize=10)


def panel_slop(ax):
    rows = read_tsv(RESULTS / "tolerance" / "consensus_slop_sweep.tsv")
    rows.sort(key=lambda r: int(r["slop_bp"]))
    slops = [int(r["slop_bp"]) / 1e6 for r in rows]
    two = [int(r["two_resolutions"]) for r in rows]
    three = [int(r["three_resolutions"]) for r in rows]
    positions = np.arange(len(slops))
    ax.bar(positions - 0.19, two, 0.38, label="2 bin sizes", color="#4c72b0")
    ax.bar(positions + 0.19, three, 0.38, label="3 bin sizes", color="#55a868")
    for index, value in enumerate(two):
        ax.text(index - 0.19, value, f"{value}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{s:g}" for s in slops])
    ax.set_xlabel("matching tolerance (Mb)")
    ax.set_ylabel("loci recovered at more\nthan one bin size")
    ax.set_title("c  Tolerance does buy\nreproducible loci", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False)


def panel_gradient(ax):
    rows = read_tsv(RESULTS / "tolerance" / "ctcf_resolution_gradient.tsv")
    if not rows:
        return
    columns = [c for c in rows[0] if c != "feature_set"]
    columns.sort(key=lambda c: int(c.replace("kb", "")))
    positions = np.arange(len(columns))
    for row in rows:
        name = row["feature_set"]
        values = [float(row[c]) if row[c] else np.nan for c in columns]
        ax.plot(positions, values, marker="o", markersize=4, color=COLORS[name],
                label=LABEL[name])
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(columns)
    ax.set_xlabel("bin size (contacts per bin rise to the right)")
    ax.set_ylabel("CTCF density vs\nrelocated windows")
    ax.set_ylim(0.55, 1.1)
    ax.set_title("d  The depletion is a sparsity effect:\nit is gone at 1 Mb", loc="left",
                 fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower right")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(RESULTS / "figures" / "tolerance.png"))
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))
    for panel, ax in zip((panel_profile, panel_flank, panel_slop, panel_gradient),
                         axes.ravel()):
        panel(ax)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
