#!/usr/bin/env python3
"""Alluvial ("Sankey") view of a clustering, for comparing clustering methods.

The left column is the known developmental stage of each cell and the right column is
the cluster the method returned. A ribbon carries the cells that share a stage and a
cluster, and takes the colour of the stage it leaves. A method that recovers the stages
draws five straight ribbons of five colours; a method that mixes them draws crossing
ribbons and clusters that receive several colours at once, so the failure is visible
without reading the ARI off a bar chart.

The clusters on the right are ordered by the stage they match best, under the assignment
that maximizes the number of cells on the diagonal (the Hungarian assignment). That
ordering is what makes a good clustering look flat: it removes the arbitrariness of the
cluster numbering, which carries no meaning, without moving any cell.

Every folder that clusters these cells (``scptr_robustad``, ``schiccluster_scptr``,
``higashi_scptr``) has a ``plot_cluster_sankey.py`` that reads its own result format and
calls into this module, so the three panels are drawn by one piece of code.
"""

from collections import Counter

import numpy as np

STAGE_ORDER = ["1-cell", "2-cell", "4-cell", "8-cell", "64-cell"]

# The method colours of the paper take blue, orange, green and red (TADLineage #4c72b0,
# scHiCluster #dd8452, Higashi #55a868, scHiCPTR #c44e52), so the stages take the hues those
# leave free: purple, magenta, brown, mustard and teal. Adjacent stages, which sit next to each
# other in every alluvial panel, stay apart under colour-vision deficiency (protan dE >= 9).
STAGE_COLORS = {
    "1-cell": "#7b3fa0",
    "2-cell": "#e7298a",
    "4-cell": "#8f5a1a",
    "8-cell": "#d1b000",
    "64-cell": "#17a3b8",
}


def stage_palette(stage_order=None):
    """The stage colours used by every embryo figure in the repository."""
    import matplotlib.pyplot as plt

    stage_order = list(stage_order or STAGE_ORDER)
    if all(stage in STAGE_COLORS for stage in stage_order):
        return {stage: STAGE_COLORS[stage] for stage in stage_order}
    cmap = plt.get_cmap("tab10", len(stage_order))
    return {stage: cmap(i) for i, stage in enumerate(stage_order)}


def contingency_table(true_labels, predicted, stage_order=None, cluster_order=None):
    """Counts of cells per (stage, cluster), as an array plus the two axis orders."""
    true_labels = list(true_labels)
    predicted = list(predicted)
    if len(true_labels) != len(predicted):
        raise ValueError("true_labels and predicted must have the same length")

    present = set(true_labels)
    if stage_order is None:
        stage_order = [s for s in STAGE_ORDER if s in present]
        stage_order += sorted(present - set(stage_order), key=str)
    else:
        stage_order = [s for s in stage_order if s in present]

    if cluster_order is None:
        cluster_order = sorted(set(predicted), key=lambda c: (isinstance(c, str), c))

    stage_index = {stage: i for i, stage in enumerate(stage_order)}
    cluster_index = {cluster: j for j, cluster in enumerate(cluster_order)}

    table = np.zeros((len(stage_order), len(cluster_order)), dtype=int)
    for stage, cluster in zip(true_labels, predicted):
        table[stage_index[stage], cluster_index[cluster]] += 1
    return table, stage_order, cluster_order


def match_clusters_to_stages(table):
    """Assign each cluster to a stage so that the diagonal holds as many cells as possible.

    Returns the column order that lines the clusters up with the stages, and the stage
    row each cluster was matched to (``None`` for a cluster left over when there are more
    clusters than stages).
    """
    n_stages, n_clusters = table.shape
    matched_stage = {}
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-table)
        matched_stage = dict(zip(cols.tolist(), rows.tolist()))
    except ImportError:
        # greedy fallback: repeatedly take the largest remaining cell of the table
        remaining_rows = set(range(n_stages))
        remaining_cols = set(range(n_clusters))
        while remaining_rows and remaining_cols:
            best = max(
                ((r, c) for r in remaining_rows for c in remaining_cols),
                key=lambda rc: table[rc],
            )
            if table[best] == 0:
                break
            matched_stage[best[1]] = best[0]
            remaining_rows.discard(best[0])
            remaining_cols.discard(best[1])

    def sort_key(cluster_col):
        stage_row = matched_stage.get(cluster_col)
        if stage_row is None:
            # an unmatched cluster sits next to the stage it draws most of its cells from
            column = table[:, cluster_col]
            stage_row = int(np.argmax(column)) if column.sum() else n_stages
            return (stage_row, 1, -int(column.sum()))
        return (stage_row, 0, 0)

    column_order = sorted(range(n_clusters), key=sort_key)
    return column_order, matched_stage


def matched_fraction(table, matched_stage):
    """Fraction of cells sitting on the matched diagonal: the accuracy the figure shows."""
    total = int(table.sum())
    if total == 0:
        return float("nan")
    on_diagonal = sum(int(table[row, col]) for col, row in matched_stage.items())
    return on_diagonal / total


def diagonal_fraction(true_labels, predicted, stage_order=None):
    """The matched-diagonal fraction of an assignment, without drawing it.

    Use this to quote the accuracy of the cells a method actually clustered when the
    figure draws a rescaled version of them.
    """
    table, _, _ = contingency_table(true_labels, predicted, stage_order=stage_order)
    _, matched_stage = match_clusters_to_stages(table)
    return matched_fraction(table, matched_stage)


def resolution_label(resolution):
    """Name a bin size the way the paper does: 250000 or "250kb" -> "250 kb"."""
    text = str(resolution)
    if text.isdigit():
        value = int(text)
        return (f"{value // 1_000_000} Mb" if value % 1_000_000 == 0
                else f"{value // 1000} kb")
    if text.endswith("kb"):
        value = int(text[:-2])
        return f"{value // 1000} Mb" if value % 1000 == 0 else f"{value} kb"
    return text


def panel_heading(method, resolution, chrom):
    """The heading of a panel: the method, the bin size and what it was run on."""
    genome_wide = chrom in (None, "all", "chrall", "all_higashi", "genome-wide")
    where = "genome-wide" if genome_wide else str(chrom)
    return f"{method}, {resolution_label(resolution)}, {where}"


def _ribbon(ax, x0, x1, top_start, bottom_start, top_end, bottom_end, color, alpha,
            samples=120, zorder=1):
    """One alluvial band, easing smoothly from the left column to the right column."""
    t = np.linspace(0.0, 1.0, samples)
    ease = 0.5 - 0.5 * np.cos(np.pi * t)
    x = x0 + (x1 - x0) * t
    top = top_start + (top_end - top_start) * ease
    bottom = bottom_start + (bottom_end - bottom_start) * ease
    ax.fill_between(x, bottom, top, facecolor=color, alpha=alpha, linewidth=0,
                    zorder=zorder)


def draw_alluvial(
    ax,
    true_labels,
    predicted,
    stage_order=None,
    stage_colors=None,
    title=None,
    subtitle=None,
    gap_fraction=0.015,
    bar_width=0.055,
    ribbon_alpha=0.62,
    label_fontsize=9,
    show_cluster_labels=True,
    show_counts=True,
):
    """Draw the stage-to-cluster alluvial diagram on ``ax``.

    Returns a summary dict with the contingency table, the matching, and the fraction of
    cells on the matched diagonal.
    """
    table, stage_order, cluster_order = contingency_table(
        true_labels, predicted, stage_order=stage_order
    )
    column_order, matched_stage = match_clusters_to_stages(table)
    table = table[:, column_order]
    cluster_order = [cluster_order[c] for c in column_order]
    matched_stage = {
        new_col: matched_stage[old_col]
        for new_col, old_col in enumerate(column_order)
        if old_col in matched_stage
    }

    colors = stage_colors or stage_palette(stage_order)
    total = int(table.sum())
    gap = gap_fraction * total

    stage_sizes = table.sum(axis=1)
    cluster_sizes = table.sum(axis=0)

    # stack both columns from the top, so stage order reads top to bottom
    def stack(sizes):
        tops = []
        cursor = 0.0
        for size in sizes:
            tops.append(cursor)
            cursor += size + gap
        return tops, cursor - gap if len(sizes) else 0.0

    stage_tops, left_extent = stack(stage_sizes)
    cluster_tops, right_extent = stack(cluster_sizes)

    # centre the shorter column against the taller one
    extent = max(left_extent, right_extent)
    stage_tops = [t + (extent - left_extent) / 2.0 for t in stage_tops]
    cluster_tops = [t + (extent - right_extent) / 2.0 for t in cluster_tops]

    x_left, x_right = 0.0, 1.0
    ribbon_x0 = x_left + bar_width
    ribbon_x1 = x_right - bar_width

    # the blocks: stages on the left in their own colour, clusters on the right in the
    # colour of the stage they were matched to, so a correct cluster echoes its stage
    for i, stage in enumerate(stage_order):
        if stage_sizes[i] == 0:
            continue
        ax.fill_between(
            [x_left, ribbon_x0], stage_tops[i], stage_tops[i] + stage_sizes[i],
            facecolor=colors[stage], edgecolor="none", zorder=3,
        )
        label = f"{stage}\n{stage_sizes[i]} cells" if show_counts else stage
        ax.text(
            x_left - 0.015, stage_tops[i] + stage_sizes[i] / 2.0, label,
            ha="right", va="center", fontsize=label_fontsize, linespacing=1.35,
        )

    for j, cluster in enumerate(cluster_order):
        if cluster_sizes[j] == 0:
            continue
        stage_row = matched_stage.get(j)
        face = colors[stage_order[stage_row]] if stage_row is not None else "0.6"
        ax.fill_between(
            [ribbon_x1, x_right], cluster_tops[j], cluster_tops[j] + cluster_sizes[j],
            facecolor=face, edgecolor="none", zorder=3,
        )
        if show_cluster_labels:
            label = (f"cluster {cluster}\n{cluster_sizes[j]} cells" if show_counts
                     else f"cluster {cluster}")
            ax.text(
                x_right + 0.015, cluster_tops[j] + cluster_sizes[j] / 2.0, label,
                ha="left", va="center", fontsize=label_fontsize, linespacing=1.35,
            )

    # the ribbons, ordered within each block so that they cross as little as possible,
    # and drawn widest first so a thin ribbon stays visible on top of a thick one
    left_cursor = list(stage_tops)
    right_cursor = list(cluster_tops)
    bands = []
    for i in range(len(stage_order)):
        for j in range(len(cluster_order)):
            count = int(table[i, j])
            if count == 0:
                continue
            top_start = left_cursor[i]
            left_cursor[i] += count
            top_end = right_cursor[j]
            right_cursor[j] += count
            bands.append((count, i, top_start, top_end))

    for count, i, top_start, top_end in sorted(bands, key=lambda b: -b[0]):
        _ribbon(
            ax, ribbon_x0, ribbon_x1,
            top_start, top_start + count,
            top_end, top_end + count,
            colors[stage_order[i]], ribbon_alpha,
        )

    ax.set_xlim(x_left - 0.30, x_right + 0.30)
    ax.set_ylim(-0.04 * extent, extent * 1.04)
    ax.invert_yaxis()
    ax.set_axis_off()

    accuracy = matched_fraction(table, matched_stage)
    if title:
        ax.set_title(title, fontsize=13, pad=16)
    if subtitle:
        ax.text(0.5, -0.035 * extent, subtitle, ha="center", va="bottom",
                fontsize=10, transform=ax.transData)

    return {
        "table": table,
        "stage_order": stage_order,
        "cluster_order": cluster_order,
        "matched_stage": matched_stage,
        "matched_fraction": accuracy,
        "n_cells": total,
    }


def summary_line(stats, ari=None, nmi=None):
    """The one-line caption under a panel: ARI plus the accuracy the ribbons show."""
    parts = []
    if ari is not None:
        parts.append(f"ARI = {ari:.3f}")
    if nmi is not None:
        parts.append(f"NMI = {nmi:.3f}")
    parts.append(f"{stats['matched_fraction'] * 100:.0f}% of cells on the matched diagonal")
    return "     ".join(parts)


def print_table(stats):
    """Print the contingency table, so a run can be checked without opening the figure."""
    table = stats["table"]
    header = "".join(f"{f'C{c}':>10s}" for c in stats["cluster_order"])
    print(f"{'stage':>10s}{header}{'total':>10s}")
    for i, stage in enumerate(stats["stage_order"]):
        row = "".join(f"{int(v):>10d}" for v in table[i])
        print(f"{stage:>10s}{row}{int(table[i].sum()):>10d}")
    totals = "".join(f"{int(v):>10d}" for v in table.sum(axis=0))
    print(f"{'total':>10s}{totals}{int(table.sum()):>10d}")
    print(f"cells on the matched diagonal: {stats['matched_fraction'] * 100:.1f}%")


def figure_for_single_method(
    true_labels, predicted, title, subtitle=None, figsize=(6.4, 5.2), **kwargs
):  # kwargs pass straight to draw_alluvial, e.g. show_counts=False
    """A one-panel figure, for the per-method scripts."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    stats = draw_alluvial(ax, true_labels, predicted, title=title, subtitle=subtitle,
                          **kwargs)
    fig.tight_layout()
    return fig, stats


def majority_stage_by_cluster(true_labels, predicted):
    """The stage most of the cells of each cluster actually belong to."""
    by_cluster = {}
    for stage, cluster in zip(true_labels, predicted):
        by_cluster.setdefault(cluster, Counter())[stage] += 1
    return {cluster: counts.most_common(1)[0][0] for cluster, counts in by_cluster.items()}
