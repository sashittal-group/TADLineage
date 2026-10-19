#!/usr/bin/env python3
"""
Analyze internal-node inferred TADs that are not present in children.

For each (chromosome, resolution, threshold, leaf1, leaf2):
1) CMP: infer novelty in inferred CMP vs (leaf1 union leaf2)
2) MPP: infer novelty in inferred MPP vs (inferred CMP union CLP)
3) For each node, count how many novel inferred TADs appear in the real node set.

Two modes:
* single case (default, used by check_intermediate_snakemake_given_root_weighted_updated):
  one (chrom, res, tol, leaf1, leaf2) instance -> one JSON summary.
* --sweep: scan the whole grid of instances under an inferred directory
  (default inferred_given_root_weighted_updated) and, per resolution, plot the
  distribution of "how many novel TADs were inferred" over all instances.
"""

import os
import json
import argparse
import itertools
from collections import Counter

import sys
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath(".."))

from tad_utilities import read_tad_intervals, parse_interval, match_tads_within_tolerance


DEFAULT_OUTDIR = "inferred_given_root_weighted_updated"
LINE1 = ["MEP", "MKP", "MK"]
LINE2 = ["GMP", "G"]
CHRS = [str(i) for i in range(1, 20)]
RESOLUTIONS = ["100kb", "250kb", "500kb", "1mb", "2mb"]
RESOLUTION_ORDER = ["100kb", "250kb", "500kb", "1mb", "2mb"]
TOLS = [str(t) for t in range(5, 11)]

_TAD_CACHE = {}


def load_tads(path):
    """Read a .tad file into a set of (a, b) bin intervals (cached by path)."""
    if path not in _TAD_CACHE:
        intervals = read_tad_intervals(path)
        _TAD_CACHE[path] = {parse_interval(s) for s in intervals}
    return _TAD_CACHE[path]


def unmatched_with_tolerance(source_set, reference_set, tolerance):
    _, unmatched_source, _ = match_tads_within_tolerance(source_set, reference_set, tolerance)
    return set(unmatched_source)


def count_matches_with_tolerance(source_set, reference_set, tolerance):
    matched_pairs, _, _ = match_tads_within_tolerance(source_set, reference_set, tolerance)
    return len(matched_pairs)


def to_interval_strings(tads):
    return [f"[{a},{b}]" for (a, b) in sorted(tads)]


def resolve_leaf_name(leaf):
    # Some notes use GR while the folder is G.
    if leaf == "GR":
        return "G"
    return leaf


def run_analysis(leaf1, leaf2, chrom, res, tolerance, outdir=DEFAULT_OUTDIR):
    leaf1_dir = resolve_leaf_name(leaf1)
    leaf2_dir = resolve_leaf_name(leaf2)

    cmp_inferred_path = os.path.join(
        outdir, "CMP", f"chr{chrom}", res, f"inferred_{leaf1}_{leaf2}_{tolerance}.tad"
    )
    mpp_inferred_path = os.path.join(
        outdir, "MPP", f"chr{chrom}", res, f"inferred_{leaf1}_{leaf2}_{tolerance}.tad"
    )

    leaf1_path = os.path.join(leaf1_dir, f"chr{chrom}", res, "tads.tad")
    leaf2_path = os.path.join(leaf2_dir, f"chr{chrom}", res, "tads.tad")
    clp_path = os.path.join("CLP", f"chr{chrom}", res, "tads.tad")
    cmp_real_path = os.path.join("CMP", f"chr{chrom}", res, "tads.tad")
    mpp_real_path = os.path.join("MPP", f"chr{chrom}", res, "tads.tad")

    needed_paths = [
        cmp_inferred_path,
        mpp_inferred_path,
        leaf1_path,
        leaf2_path,
        clp_path,
        cmp_real_path,
        mpp_real_path,
    ]
    missing = [p for p in needed_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    cmp_inferred = load_tads(cmp_inferred_path)
    mpp_inferred = load_tads(mpp_inferred_path)
    leaf1_tads = load_tads(leaf1_path)
    leaf2_tads = load_tads(leaf2_path)
    clp_tads = load_tads(clp_path)
    cmp_real = load_tads(cmp_real_path)
    mpp_real = load_tads(mpp_real_path)

    # CMP novelty: inferred CMP not found in (leaf1 union leaf2).
    cmp_children_union = leaf1_tads | leaf2_tads
    cmp_novel = unmatched_with_tolerance(cmp_inferred, cmp_children_union, tolerance)
    cmp_novel_in_real = count_matches_with_tolerance(cmp_novel, cmp_real, tolerance)

    # MPP novelty: inferred MPP not found in (inferred CMP union CLP).
    mpp_children_union = cmp_inferred | clp_tads
    mpp_novel = unmatched_with_tolerance(mpp_inferred, mpp_children_union, tolerance)
    mpp_novel_in_real = count_matches_with_tolerance(mpp_novel, mpp_real, tolerance)

    cmp_novel_fraction_in_real = (cmp_novel_in_real / len(cmp_novel)) if cmp_novel else 0.0
    mpp_novel_fraction_in_real = (mpp_novel_in_real / len(mpp_novel)) if mpp_novel else 0.0

    return {
        "chrom": str(chrom),
        "resolution": res,
        "tolerance": int(tolerance),
        "leaf1": leaf1,
        "leaf2": leaf2,
        "paths": {
            "cmp_inferred": cmp_inferred_path,
            "mpp_inferred": mpp_inferred_path,
            "leaf1": leaf1_path,
            "leaf2": leaf2_path,
            "clp": clp_path,
            "cmp_real": cmp_real_path,
            "mpp_real": mpp_real_path,
        },
        "CMP": {
            "n_inferred": len(cmp_inferred),
            "n_children_union": len(cmp_children_union),
            "n_novel_vs_children": len(cmp_novel),
            "n_novel_present_in_real": cmp_novel_in_real,
            "novel_fraction_present_in_real": cmp_novel_fraction_in_real,
            "novel_tads": to_interval_strings(cmp_novel),
        },
        "MPP": {
            "n_inferred": len(mpp_inferred),
            "n_children_union": len(mpp_children_union),
            "n_novel_vs_children": len(mpp_novel),
            "n_novel_present_in_real": mpp_novel_in_real,
            "novel_fraction_present_in_real": mpp_novel_fraction_in_real,
            "novel_tads": to_interval_strings(mpp_novel),
        },
    }


def resolution_sort_key(res):
    if res in RESOLUTION_ORDER:
        return (0, RESOLUTION_ORDER.index(res))
    return (1, res)


def sweep_novelty(outdir, chroms, resolutions, tolerances, leaf1_list, leaf2_list, verbose=True):
    """
    Run the novelty analysis over the whole instance grid.

    Returns (counts, skipped) where
      counts[res][node] = list of novel-TAD counts, one entry per instance
      skipped           = list of (instance-key, reason) for instances whose files are missing
    """
    counts = {res: {"CMP": [], "MPP": []} for res in resolutions}
    skipped = []

    grid = list(itertools.product(resolutions, chroms, leaf1_list, leaf2_list, tolerances))
    for res, chrom, leaf1, leaf2, tol in grid:
        key = f"chr{chrom}_{res}_{leaf1}_{leaf2}_{tol}"
        try:
            summary = run_analysis(
                leaf1=leaf1,
                leaf2=leaf2,
                chrom=chrom,
                res=res,
                tolerance=int(tol),
                outdir=outdir,
            )
        except FileNotFoundError as exc:
            skipped.append((key, str(exc).splitlines()[-1]))
            continue
        counts[res]["CMP"].append(summary["CMP"]["n_novel_vs_children"])
        counts[res]["MPP"].append(summary["MPP"]["n_novel_vs_children"])

    if verbose:
        n_done = sum(len(counts[res]["CMP"]) for res in resolutions)
        print(f"Analyzed {n_done}/{len(grid)} instances ({len(skipped)} skipped for missing files)")
        for key, reason in skipped[:5]:
            print(f"  skipped {key}: {reason}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")

    return counts, skipped


def distribution_stats(values):
    """Frequency and fraction of instances per novel-TAD count."""
    import numpy as np

    dist = Counter(values)
    total = len(values)
    return {
        "n_instances": total,
        "mean": float(np.mean(values)) if total else 0.0,
        "median": float(np.median(values)) if total else 0.0,
        "min": int(min(values)) if total else 0,
        "max": int(max(values)) if total else 0,
        "frac_zero": (dist[0] / total) if total else 0.0,
        "counts": {str(k): int(dist[k]) for k in sorted(dist)},
        "fractions": {str(k): dist[k] / total for k in sorted(dist)},
    }


def plot_resolution_distribution(res, res_counts, out_path):
    """
    One figure per resolution: x = number of novel TADs inferred at an internal
    node, y = fraction of instances with exactly that many, CMP vs MPP.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nodes = [n for n in ("CMP", "MPP") if res_counts.get(n)]
    if not nodes:
        print(f"No instances for resolution {res}; skipping plot")
        return None

    max_count = max(max(res_counts[n]) for n in nodes)
    x_vals = list(range(0, max_count + 1))
    x_idx = np.arange(len(x_vals))
    width = 0.8 / len(nodes)
    colors = {"CMP": "#4C72B0", "MPP": "#DD8452"}

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(x_vals) + 4), 6))

    stats = {}
    for i, node in enumerate(nodes):
        values = res_counts[node]
        stats[node] = distribution_stats(values)
        total = len(values)
        dist = Counter(values)
        fracs = [dist[x] / total for x in x_vals]
        offset = (i - (len(nodes) - 1) / 2) * width
        bars = ax.bar(
            x_idx + offset,
            fracs,
            width=width,
            label=f"{node} (n={total}, mean={stats[node]['mean']:.2f})",
            color=colors.get(node),
            edgecolor="black",
            alpha=0.85,
        )
        for bar, x in zip(bars, x_vals):
            if dist[x] == 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(dist[x]),
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_xticks(x_idx)
    ax.set_xticklabels([str(x) for x in x_vals])
    ax.set_xlabel("Number of novel inferred TADs (not present in children)",
                  fontsize=12, fontweight="bold")
    ax.set_ylabel("Fraction of instances", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{res}: distribution of novel TADs inferred at internal nodes",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
    return stats


def plot_mean_by_resolution(counts, resolutions, out_path):
    """Summary figure: mean number of novel TADs per resolution, CMP vs MPP."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resolutions = [r for r in resolutions if counts[r]["CMP"] or counts[r]["MPP"]]
    if not resolutions:
        return

    x_idx = np.arange(len(resolutions))
    width = 0.38
    colors = {"CMP": "#4C72B0", "MPP": "#DD8452"}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, node in enumerate(("CMP", "MPP")):
        means = [float(np.mean(counts[r][node])) if counts[r][node] else 0.0 for r in resolutions]
        sems = [
            float(np.std(counts[r][node]) / np.sqrt(len(counts[r][node]))) if counts[r][node] else 0.0
            for r in resolutions
        ]
        offset = (i - 0.5) * width
        ax.bar(x_idx + offset, means, width=width, yerr=sems, capsize=3,
               label=node, color=colors[node], edgecolor="black", alpha=0.85)
        for x, m in zip(x_idx + offset, means):
            ax.text(x, m, f"{m:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x_idx)
    ax.set_xticklabels(resolutions)
    ax.set_xlabel("Resolution", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean # novel inferred TADs", fontsize=12, fontweight="bold")
    ax.set_title("Novel TADs at internal nodes by resolution", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def run_sweep(args):
    resolutions = sorted(args.resolutions, key=resolution_sort_key)
    counts, skipped = sweep_novelty(
        outdir=args.outdir,
        chroms=args.chroms,
        resolutions=resolutions,
        tolerances=args.tolerances,
        leaf1_list=args.leaf1_list,
        leaf2_list=args.leaf2_list,
    )

    fig_dir = args.fig_dir or os.path.join(args.outdir, "figures", "novel")
    os.makedirs(fig_dir, exist_ok=True)

    summary = {"outdir": args.outdir, "n_skipped": len(skipped), "by_resolution": {}}
    for res in resolutions:
        out_path = os.path.join(fig_dir, f"novel_tads_distribution_{res}.png")
        stats = plot_resolution_distribution(res, counts[res], out_path)
        if stats:
            summary["by_resolution"][res] = stats

    plot_mean_by_resolution(counts, resolutions, os.path.join(fig_dir, "novel_tads_mean_by_resolution.png"))

    summary_path = os.path.join(fig_dir, "novel_tads_distribution_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved {summary_path}")

    for res in resolutions:
        for node in ("CMP", "MPP"):
            st = summary["by_resolution"].get(res, {}).get(node)
            if st:
                print(f"{res:>6} {node}: n={st['n_instances']:4d} mean={st['mean']:.2f} "
                      f"zero-novel={100 * st['frac_zero']:.1f}% max={st['max']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze internal-only inferred TADs for CMP/MPP")
    parser.add_argument("--sweep", action="store_true",
                        help="scan every instance under --outdir and plot per-resolution distributions")
    parser.add_argument("--leaf1", help="one of MEP/MKP/MK")
    parser.add_argument("--leaf2", help="one of GMP/G (or GR alias)")
    parser.add_argument("--chrom", help="chromosome without chr prefix")
    parser.add_argument("--resolution", help="e.g. 250kb")
    parser.add_argument("--tolerance", type=int, help="matching tolerance in bins")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="base inferred directory")
    parser.add_argument("--output", help="output json path (single-case mode)")
    # sweep-mode grid
    parser.add_argument("--chroms", nargs="+", default=CHRS)
    parser.add_argument("--resolutions", nargs="+", default=RESOLUTIONS)
    parser.add_argument("--tolerances", nargs="+", default=TOLS)
    parser.add_argument("--leaf1-list", nargs="+", default=LINE1)
    parser.add_argument("--leaf2-list", nargs="+", default=LINE2)
    parser.add_argument("--fig-dir", default=None,
                        help="figure output dir (default <outdir>/figures/novel)")
    args = parser.parse_args()

    if args.sweep:
        run_sweep(args)
        return

    missing_args = [
        name for name in ("leaf1", "leaf2", "chrom", "resolution", "tolerance", "output")
        if getattr(args, name) is None
    ]
    if missing_args:
        parser.error(
            "single-case mode requires " + ", ".join(f"--{n}" for n in missing_args) + " (or pass --sweep)"
        )

    summary = run_analysis(
        leaf1=args.leaf1,
        leaf2=args.leaf2,
        chrom=args.chrom,
        res=args.resolution,
        tolerance=args.tolerance,
        outdir=args.outdir,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Saved internal novelty summary: {args.output}")


if __name__ == "__main__":
    main()
