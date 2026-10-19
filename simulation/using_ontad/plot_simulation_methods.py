#!/usr/bin/env python3
"""Aggregate simulation method summaries and draw one A/B/C F1 boxplot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SIM_ROOT = SCRIPT_DIR.parent           # simulation/ (shared config.py)
sys.path.insert(0, str(SIM_ROOT))
MPL_CACHE = SCRIPT_DIR / ".matplotlib_cache"
XDG_CACHE = SCRIPT_DIR / ".cache"
MPL_CACHE.mkdir(exist_ok=True)
XDG_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config


METHOD_LABELS = {
    "inferred_given_root": "given_root",
    "inferred_given_root_weighted": "given_root_weighted",
    "inferred_intersection": "intersection",
    "inferred_rf": "FH",
    "inferred_rf_weighted": "Sankoff",
    "inferred_union": "union",
}
METHOD_ORDER = ["given_root", "given_root_weighted", "intersection", "FH", "Sankoff", "union"]
METRICS = ["precision", "recall", "f1"]


def load_rows(results_dir: Path) -> list[dict]:
    rows = []
    for method_dir in sorted(results_dir.glob("inferred_*")):
        json_dir = method_dir / "jsons"
        if not json_dir.exists():
            continue
        method = method_dir.name
        label = METHOD_LABELS.get(method, method)
        for json_path in sorted(json_dir.glob("region_*.json")):
            with json_path.open() as handle:
                summary = json.load(handle)
            for node in config.SCORE_NODES:
                node_summary = summary[node]
                for metric in METRICS:
                    rows.append(
                        {
                            "region": summary["region"],
                            "method": method,
                            "method_label": label,
                            "node": node,
                            "metric": metric,
                            "value": float(node_summary[metric]),
                            "n_inferred": int(node_summary["n_inferred"]),
                            "n_true": int(node_summary["n_true"]),
                            "n_matched": int(node_summary["n_matched"]),
                        }
                    )
    return rows


def plot_f1_by_node(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df[df["metric"] == "f1"].copy()
    plot_df["node"] = pd.Categorical(plot_df["node"], categories=config.SCORE_NODES, ordered=True)
    method_order = [m for m in METHOD_ORDER if m in set(plot_df["method_label"])]
    plot_df["method_label"] = pd.Categorical(
        plot_df["method_label"],
        categories=method_order,
        ordered=True,
    )

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(
        data=plot_df,
        x="node",
        y="value",
        hue="method_label",
        hue_order=method_order,
        ax=ax,
        linewidth=1.0,
        fliersize=2.5,
    )
    sns.stripplot(
        data=plot_df,
        x="node",
        y="value",
        hue="method_label",
        hue_order=method_order,
        ax=ax,
        dodge=True,
        size=1.8,
        alpha=0.35,
        linewidth=0,
        legend=False,
    )

    ax.set_title("Internal-node reconstruction F1")
    ax.set_xlabel("Internal node")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.02)
    ax.legend(title="Method", loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=6, frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "method_results")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent / "method_plots")
    args = parser.parse_args()

    rows = load_rows(args.results_dir)
    if not rows:
        raise SystemExit(f"No method summary JSONs found under {args.results_dir}")

    df = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.outdir / "method_metrics_long.csv", index=False)

    summary = (
        df.groupby(["node", "method_label", "metric"], observed=True)["value"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    summary.to_csv(args.outdir / "method_metrics_summary.csv", index=False)

    plot_f1_by_node(df, args.outdir / "internal_nodes_f1_boxplot.png")

    print(f"Wrote plots and metric tables under {args.outdir}")


if __name__ == "__main__":
    main()
