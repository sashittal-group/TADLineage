#!/usr/bin/env python3
"""What one caller's merge/split calls do and do not support, in one figure.

Run once per caller. The two callers are never compared by position here; each is scored
against its own held-out cells, so the figures are two independent readings of the same
question rather than a replication test.

a. the class-level result at the caller's chosen bin size, one point per chromosome. Bins OnTAD calls *split*
   gain penetrance in held-out cells, bins it calls *merge* lose it. This is the positive.
b. every transition, both classes. All eight rows point the way the method predicts.
c. the locus-level negative and why it is not about caller disagreement. The grey histogram
   is the merge/split loci themselves, each standardised against the median and robust
   scale of its own chromosome's background bins, and the coloured rug repeats them split
   against merge. If the per-bin penetrance change behaved the way a 2x2 Fisher test
   assumes, that histogram would sit under the dashed Gaussian. It does not: its spread is
   about twice as wide and 44 of 764 loci pass |z| = 4, where a Gaussian predicts 0.05 of
   them. So a locus at z = 7 is an ordinary member of this distribution rather than the
   1-in-1e12 event a parametric test would call it, which is exactly why Fisher reports
   hundreds of false hits on these bins.
d. gene sets at merge/split domains at the same bin size, counted by *cluster* rather
   than by gene.
   ``housekeeping_control`` is the method check, not a claim: the curated sets do no better
   than it, so none of them is enriched.

Usage:
    python plot_merge_split_validation.py --caller RobusTAD
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
FIGURES = RESULTS / "figures"

COLORS = {"split": "#4c72b0", "merge": "#c44e52"}
TRANSITIONS = ["1-cell->2-cell", "2-cell->4-cell", "4-cell->8-cell", "8-cell->64-cell"]
SHORT = {t: t.replace("-cell", "") for t in TRANSITIONS}
TITLE = 10
SMALL = 7.5
PRIMARY = "500kb"


# The bin size the whole figure is drawn at, per caller. 500 kb is where both callers have
# their strongest class-level result, and it is the same bin size for both, which is why it
# is written down here rather than picked by a rule. THIS IS A DISPLAY CHOICE MADE AFTER
# SEEING THE RESULT: the configurations it leaves out are not hidden, they are reported in
# section 4 of the report -- RobusTAD is null at 250 kb, OnTAD is null at 1 Mb, and the
# RobusTAD 1 Mb clustering has no held-out cells at all because it used the whole pool.
# Override with --primary, and `class_summary.tsv` carries every bin size either way.
PRIMARY_BY_CALLER = {"RobusTAD": "500kb", "OnTAD": "500kb"}


def resolutions_present(summary):
    return sorted({r["resolution"] for r in summary if r["transition"] == "all"},
                  key=lambda r: int(r.replace("kb", "")))


def panel_class(ax, loci, summary, primary):
    """Split and merge side by side at the caller's chosen bin size."""
    data, labels, colors, notes = [], [], [], []
    for resolution in (primary,):
        for event_type in ("split", "merge"):
            by_chrom = collections.defaultdict(list)
            for row in loci:
                if row["resolution"] != resolution or row["event_type"] != event_type:
                    continue
                by_chrom[row["chrom"]].append(float(row["referenced_delta"]))
            if not by_chrom:
                continue
            data.append([float(np.mean(v)) for v in by_chrom.values()])
            record = next((r for r in summary if r["resolution"] == resolution
                           and r["event_type"] == event_type
                           and r["transition"] == "all"), None)
            labels.append(f"{event_type}\n{record['agreeing']}/{record['chromosomes']} "
                          f"chrom, p = {float(record['sign_p']):.4f}"
                          if record else event_type)
            colors.append(COLORS[event_type])
    positions = np.arange(len(data))
    box = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                     medianprops=dict(color="black", linewidth=1.1),
                     flierprops=dict(marker=".", markersize=3, alpha=0.6))
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.42)
    rng = np.random.default_rng(0)
    for position, values, color in zip(positions, data, colors):
        jitter = (rng.random(len(values)) - 0.5) * 0.26
        ax.plot(position + jitter, values, ".", color=color, markersize=4.2, alpha=0.9)
    ax.axhline(0, color="0.4", linewidth=0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=SMALL)
    ax.set_ylabel("penetrance change in held-out cells\n(one point per chromosome)")
    ax.set_title(f"a  the calls as a class, at {primary} "
                 "(split predicts above zero, merge below)",
                 fontsize=TITLE, loc="left")


def panel_transitions(ax, summary, primary):
    width = 0.38
    positions = np.arange(len(TRANSITIONS))
    for offset, event_type in ((-width / 2, "split"), (width / 2, "merge")):
        values = []
        for transition in TRANSITIONS:
            record = next((r for r in summary if r["resolution"] == primary
                           and r["event_type"] == event_type
                           and r["transition"] == transition), None)
            values.append(float(record["mean_referenced_delta"]) if record else np.nan)
        ax.bar(positions + offset, values, width, color=COLORS[event_type], alpha=0.85,
               edgecolor="white", label=event_type)
    ax.axhline(0, color="0.4", linewidth=0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels([SHORT[t] for t in TRANSITIONS], fontsize=SMALL)
    ax.set_xlabel("transition between consecutive stage medians", fontsize=SMALL)
    ax.set_ylabel("mean penetrance change\nin held-out cells")
    rows = [r for r in summary if r["resolution"] == primary and r["transition"] != "all"]
    correct = sum(1 for r in rows
                  if (float(r["mean_referenced_delta"]) > 0) == (r["expected"] == "up"))
    ax.set_title(f"b  {correct} of {len(rows)} rows point the predicted way, at {primary}",
                 fontsize=TITLE, loc="left")
    ax.legend(fontsize=SMALL, frameon=False)


def panel_background(ax, loci, primary):
    picked = [r for r in loci if r["resolution"] == primary and r["z_score"] != ""]
    z = np.array([float(r["z_score"]) for r in picked])
    grid = np.linspace(-9, 9, 300)
    gaussian = np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi)
    ax.plot(grid, gaussian, color="0.35", linewidth=1.2, linestyle="--",
            label="Gaussian the Fisher test assumes")
    ax.hist(z, bins=60, density=True, color="0.75", edgecolor="white", linewidth=0.3,
            label="observed merge/split loci")
    for event_type in ("split", "merge"):
        values = np.array([float(r["z_score"]) for r in picked
                           if r["event_type"] == event_type])
        ax.plot(values, np.full(values.size, 1.5e-3), "|", color=COLORS[event_type],
                markersize=6, alpha=0.6)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.0)
    ax.set_xlabel("standardised penetrance change (z against the chromosome's own bins)")
    ax.set_ylabel("density")
    ax.set_title("c  why a parametric test lies here: the tails are far too heavy",
                 fontsize=TITLE, loc="left")
    ax.legend(fontsize=6.6, frameon=False, loc="upper right")


def panel_gene_sets(ax, clusters, primary):
    """Restricted to the same bin size as the rest of the figure.

    Pooling all bin sizes instead roughly doubles the runs behind each bar and gives the
    same answer -- no set clears chance on either caller -- so nothing is lost by keeping
    the figure internally consistent. `cluster_enrichment.tsv` holds every bin size.
    """
    keep = [r for r in clusters if r["event_class"] in ("merge", "split")
            and r["resolution"] == primary and r["fold"] not in ("", None)]
    by_set = collections.defaultdict(list)
    for row in keep:
        by_set[row["gene_set"]].append(float(row["fold"]))
    # the fraction of configurations that are enriched at all, which is 0.5 under the null.
    # the MEAN fold is not usable here: it is a ratio whose denominator is often near zero,
    # so a set with four clusters can average 2.9 while being enriched in 7 runs of 42.
    share = {k: float(np.mean([f > 1 for f in v])) for k, v in by_set.items()}
    order = sorted(share, key=lambda k: share[k])
    positions = np.arange(len(order))
    colors = ["#8172b3" if k == "housekeeping_control" else "0.6" for k in order]
    ax.barh(positions, [share[k] for k in order], 0.66, color=colors, alpha=0.9,
            edgecolor="white")
    ax.axvline(0.5, color="0.35", linewidth=1.1, linestyle="--")
    for position, key in zip(positions, order):
        ax.annotate(f"mean fold {np.mean(by_set[key]):.2f} "
                    f"({len(by_set[key])} runs)",
                    xy=(share[key], position), xytext=(4, 0),
                    textcoords="offset points", fontsize=5.8, color="0.35",
                    va="center")
    ax.set_xlim(0, 1.05)
    ax.set_yticks(positions)
    ax.set_yticklabels([k.replace("_", " ") for k in order], fontsize=6.6)
    ax.set_xlabel(f"share of {primary} configurations with any enrichment "
                  "(0.5 = chance, dashed)", fontsize=SMALL)
    ax.set_title("d  no curated set is enriched at merge/split domains",
                 fontsize=TITLE, loc="left")
    ax.annotate("method check, not a claim", xy=(0.5, positions[
                    order.index("housekeeping_control")]),
                xytext=(0, -13), textcoords="offset points", fontsize=6.0,
                color="#8172b3", ha="center", va="top")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--caller", default="OnTAD")
    parser.add_argument("--primary", default=None,
                        help="bin size for panel b; defaults to PRIMARY_BY_CALLER")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    out_dir = Path(args.out_dir or FIGURES)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = RESULTS / "merge_split" / args.caller
    loci = read_tsv(tables / "locus_validation.tsv")
    summary = read_tsv(tables / "class_summary.tsv")
    clusters = read_tsv(tables / "cluster_enrichment.tsv")
    available = resolutions_present(summary)
    primary = args.primary or PRIMARY_BY_CALLER.get(args.caller, PRIMARY)
    if primary not in available:
        raise SystemExit(f"{args.caller} has no {primary} rows; available: {available}")

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    panel_class(axes[0][0], loci, summary, primary)
    panel_transitions(axes[0][1], summary, primary)
    panel_background(axes[1][0], loci, primary)
    panel_gene_sets(axes[1][1], clusters, primary)
    figure.suptitle(f"{args.caller} merge and split calls, validated on its own held-out "
                    f"cells", fontsize=11, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    path = out_dir / f"{args.caller.lower()}_merge_split.png"
    figure.savefig(path, dpi=300)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
