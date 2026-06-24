#!/usr/bin/env python3
"""
Plot distribution of internal-only novel TADs across all instances.

For each node (CMP, MPP):
- Histogram showing: x-axis = number of novel TADs, y-axis = count of instances with that many novel TADs
- Aggregated across all chromosomes, resolutions, tolerances, and leaf combinations
"""

import os
import json
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


RESOLUTION_ORDER = ["2mb", "1mb", "500kb", "250kb", "100kb"]


def load_novelty_jsons(novelty_dir="inferred/jsons"):
    """
    Load all internal_novelty_*.json files and extract novel TAD counts.
    Returns:
      - cmp_counts: list of CMP counts across all instances
      - mpp_counts: list of MPP counts across all instances
      - resolution_counts: nested dict mapping node -> resolution -> list of counts
    """
    cmp_counts = []
    mpp_counts = []
    resolution_counts = {
        "CMP": defaultdict(list),
        "MPP": defaultdict(list),
    }
    
    pattern = os.path.join(novelty_dir, "internal_novelty_chr*.json")
    json_files = sorted(glob.glob(pattern))
    
    print(f"Found {len(json_files)} internal_novelty JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, "r") as fh:
                data = json.load(fh)
                resolution = data.get("resolution", "unknown")
                cmp_count = data.get("CMP", {}).get("n_novel_vs_children", 0)
                mpp_count = data.get("MPP", {}).get("n_novel_vs_children", 0)
                cmp_counts.append(cmp_count)
                mpp_counts.append(mpp_count)
                resolution_counts["CMP"][resolution].append(cmp_count)
                resolution_counts["MPP"][resolution].append(mpp_count)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    
    return cmp_counts, mpp_counts, resolution_counts


def load_missing_jsons(missing_dir="inferred/jsons"):
    """
    Load all missing_tads_*.json files and extract score percentiles of
    ground-truth TADs missed by inference.
    Returns nested dict mapping node -> resolution -> list of percentiles.
    """
    resolution_percentiles = {
        "CMP": defaultdict(list),
        "MPP": defaultdict(list),
    }

    pattern = os.path.join(missing_dir, "missing_tads_chr*.json")
    json_files = sorted(glob.glob(pattern))

    print(f"Found {len(json_files)} missing_tads JSON files")

    for json_file in json_files:
        try:
            with open(json_file, "r") as fh:
                data = json.load(fh)
                resolution = data.get("resolution", "unknown")
                cmp_percentiles = data.get("CMP", {}).get("missing_ground_truth", {}).get("score_percentiles", [])
                mpp_percentiles = data.get("MPP", {}).get("missing_ground_truth", {}).get("score_percentiles", [])
                resolution_percentiles["CMP"][resolution].extend(cmp_percentiles)
                resolution_percentiles["MPP"][resolution].extend(mpp_percentiles)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return resolution_percentiles


def resolution_sort_key(resolution):
    if resolution in RESOLUTION_ORDER:
        return (0, RESOLUTION_ORDER.index(resolution))
    return (1, resolution)


def create_histogram(counts, node_name, output_path):
    """
    Create a histogram of novel TAD counts.
    x-axis: number of novel TADs
    y-axis: frequency (count of instances with that many novel TADs)
    """
    if not counts:
        print(f"No counts available for {node_name}")
        return
    
    # Create histogram data
    count_dist = defaultdict(int)
    for c in counts:
        count_dist[c] += 1
    
    x_vals = sorted(count_dist.keys())
    y_vals = [count_dist[x] for x in x_vals]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar plot
    bars = ax.bar(x_vals, y_vals, width=0.8, edgecolor='black', alpha=0.7)
    
    # Color bars by gradient
    colors_list = plt.cm.Blues(np.linspace(0.4, 0.9, len(bars)))
    for bar, color in zip(bars, colors_list):
        bar.set_color(color)
    
    ax.set_xlabel("Number of Novel TADs (not in children)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Frequency (# instances)", fontsize=12, fontweight='bold')
    ax.set_title(f"{node_name}: Distribution of Novel Inferred TADs", fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Set x-axis to show all values
    if len(x_vals) > 1:
        ax.set_xticks(x_vals)
    
    # Add value labels on bars
    for x, y in zip(x_vals, y_vals):
        ax.text(x, y + 0.5, str(y), ha='center', va='bottom', fontweight='bold')
    
    # Summary statistics
    total_instances = len(counts)
    mean_novel = np.mean(counts)
    median_novel = np.median(counts)
    max_novel = max(counts)
    min_novel = min(counts)
    
    # Add text box with stats
    stats_text = (
        f"Total instances: {total_instances}\n"
        f"Mean: {mean_novel:.2f}\n"
        f"Median: {median_novel:.1f}\n"
        f"Min: {min_novel}, Max: {max_novel}"
    )
    ax.text(
        0.98, 0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {node_name} plot: {output_path}")
    plt.close()
    
    return {
        "node": node_name,
        "total_instances": total_instances,
        "mean": mean_novel,
        "median": median_novel,
        "min": min_novel,
        "max": max_novel,
        "distribution": dict(count_dist),
    }


def create_resolution_histograms(resolution_counts, node_name, out_dir):
    """
    Create one histogram per resolution for the supplied node.
    Returns dict mapping resolution -> summary statistics.
    """
    resolution_stats = {}

    for resolution in sorted(resolution_counts.keys(), key=resolution_sort_key):
        counts = resolution_counts[resolution]
        output_name = f"{node_name.lower()}_novel_tads_distribution_{resolution}.png"
        output_path = os.path.join(out_dir, output_name)
        stats = create_histogram(
            counts,
            f"{node_name} ({resolution})",
            output_path,
        )
        if stats:
            resolution_stats[resolution] = stats

    return resolution_stats


def create_percentile_histogram(percentiles, node_name, output_path, resolution):
    """
    Create a histogram of score percentiles for missed ground-truth TADs.
    x-axis: percentile in the ground-truth file
    y-axis: frequency of missed TAD instances
    """
    if not percentiles:
        print(f"No missing-ground-truth percentile data available for {node_name} ({resolution})")
        return None

    bins = np.arange(0, 110, 10)
    fig, ax = plt.subplots(figsize=(10, 6))
    counts, bin_edges, patches = ax.hist(
        percentiles,
        bins=bins,
        edgecolor="black",
        alpha=0.75,
    )

    colors_list = plt.cm.Oranges(np.linspace(0.35, 0.9, len(patches)))
    for patch, color in zip(patches, colors_list):
        patch.set_facecolor(color)

    ax.set_xlabel("Ground-Truth TAD Score Percentile", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency (# missed TAD instances)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{node_name} ({resolution}): Distribution of Missed Ground-Truth TAD Percentiles",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(0, 100)
    ax.set_xticks(bins)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    total_instances = len(percentiles)
    mean_percentile = float(np.mean(percentiles))
    median_percentile = float(np.median(percentiles))
    min_percentile = float(np.min(percentiles))
    max_percentile = float(np.max(percentiles))

    stats_text = (
        f"Missed TAD instances: {total_instances}\n"
        f"Mean percentile: {mean_percentile:.2f}\n"
        f"Median percentile: {median_percentile:.2f}\n"
        f"Min: {min_percentile:.2f}, Max: {max_percentile:.2f}"
    )
    ax.text(
        0.98, 0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved missing TAD percentile plot: {output_path}")
    plt.close()

    bin_labels = [
        f"{int(bin_edges[i])}-{int(bin_edges[i + 1])}"
        for i in range(len(bin_edges) - 1)
    ]
    bin_distribution = {
        label: int(count)
        for label, count in zip(bin_labels, counts)
    }

    return {
        "node": node_name,
        "resolution": resolution,
        "total_missed_tads": total_instances,
        "mean_percentile": mean_percentile,
        "median_percentile": median_percentile,
        "min_percentile": min_percentile,
        "max_percentile": max_percentile,
        "bin_distribution": bin_distribution,
    }


def create_missing_resolution_histograms(resolution_percentiles, node_name, out_dir):
    """
    Create one missed-ground-truth percentile histogram per resolution.
    """
    resolution_stats = {}

    for resolution in sorted(resolution_percentiles.keys(), key=resolution_sort_key):
        percentiles = resolution_percentiles[resolution]
        output_name = f"{node_name.lower()}_missing_tads_distribution_{resolution}.png"
        output_path = os.path.join(out_dir, output_name)
        stats = create_percentile_histogram(percentiles, node_name, output_path, resolution)
        if stats:
            resolution_stats[resolution] = stats

    return resolution_stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot internal novelty distributions")
    parser.add_argument("--novelty-dir", default="inferred/jsons", help="directory with internal_novelty JSONs")
    parser.add_argument("--out-dir", default="inferred/figures", help="output directory for plots")
    args = parser.parse_args()
    
    # Load novelty counts
    cmp_counts, mpp_counts, resolution_counts = load_novelty_jsons(args.novelty_dir)
    missing_resolution_percentiles = load_missing_jsons(args.novelty_dir)
    
    if not cmp_counts and not mpp_counts:
        print("No novelty JSON files found. Run snakemake first to generate them.")
        return
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    novel_out_dir = os.path.join(args.out_dir, "novel")
    missing_out_dir = os.path.join(args.out_dir, "missing")
    os.makedirs(novel_out_dir, exist_ok=True)
    os.makedirs(missing_out_dir, exist_ok=True)
    
    # Create plots
    cmp_stats = create_histogram(cmp_counts, "CMP", os.path.join(novel_out_dir, "cmp_novel_tads_distribution.png"))
    mpp_stats = create_histogram(mpp_counts, "MPP", os.path.join(novel_out_dir, "mpp_novel_tads_distribution.png"))
    cmp_resolution_stats = create_resolution_histograms(resolution_counts["CMP"], "CMP", novel_out_dir)
    mpp_resolution_stats = create_resolution_histograms(resolution_counts["MPP"], "MPP", novel_out_dir)
    cmp_missing_resolution_stats = create_missing_resolution_histograms(
        missing_resolution_percentiles["CMP"],
        "CMP",
        missing_out_dir,
    )
    mpp_missing_resolution_stats = create_missing_resolution_histograms(
        missing_resolution_percentiles["MPP"],
        "MPP",
        missing_out_dir,
    )
    
    # Save summary statistics
    if cmp_stats or mpp_stats:
        summary = {
            "CMP": cmp_stats,
            "MPP": mpp_stats,
            "CMP_by_resolution": cmp_resolution_stats,
            "MPP_by_resolution": mpp_resolution_stats,
            "CMP_missing_tads_by_resolution": cmp_missing_resolution_stats,
            "MPP_missing_tads_by_resolution": mpp_missing_resolution_stats,
        }
        summary_path = os.path.join(args.out_dir, "novelty_statistics.json")
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"Saved summary statistics: {summary_path}")


if __name__ == "__main__":
    main()
