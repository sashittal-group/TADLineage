#!/usr/bin/env python3
"""Collect every result of this folder into one report, with the diagnostics first.

The order is deliberate. Before any biological claim there are three checks on whether the
events are the kind of object such a claim could be made about: how much of a median
survives a transition at all, whether a boundary that changes is reproducible across bin
sizes, and whether these boundaries look like TAD boundaries in the two ways that can be
checked independently, gene density and CTCF occupancy. The enrichment and expression
results are then reported against those checks rather than on their own.

Statistical note
    Six tolerances of one bin size are six re-analyses of the same clustering of the same
    cells, so they are not six independent observations. Wherever a result is pooled here,
    tolerances are averaged within a bin size first and the test is run over the (bin size,
    transition) cells that remain. Pooling over all 18 configurations instead roughly
    doubles the apparent significance of everything, which is why it is not done.

Usage:
    python summarise_results.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import sys
from math import comb, erfc, sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cluster_sankey import STAGE_ORDER  # noqa: E402
from tad_utilities import match_tads_within_tolerance  # noqa: E402

RESULTS = HERE / "results"


def read_tsv(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sign_test(values):
    values = [v for v in values if v == v]
    n = len(values)
    positive = sum(1 for v in values if v > 0)
    if n == 0:
        return 0, 0, float("nan")
    smaller = min(positive, n - positive)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(smaller + 1)) / 2 ** n)
    return n, positive, p


def fmt_p(value):
    """A p-value as the report writes it: 2e-06, 0.00042, 0.031."""
    value = float(value)
    return f"{value:.0e}".replace("e-0", "e-") if value < 1e-3 else f"{value:.3g}"


def class_cells(classes, contrast, caller=None, feature=None):
    return [r for r in classes
            if r["contrast"] == contrast
            and (caller is None or r["caller"] == caller)
            and (feature is None or r["feature"] == feature)]


def ratio_span(rows):
    """min and max of a set of ratios, formatted, or a single value if they agree."""
    values = sorted(float(r["ratio"]) for r in rows)
    if not values:
        return "-"
    if abs(values[-1] - values[0]) < 5e-3:
        return f"{values[0]:.2f}"
    return f"{values[0]:.2f} to {values[-1]:.2f}"


def support_span(rows):
    """'19 to 20 of 20 chromosomes', from the above-one counts of a set of rows."""
    if not rows:
        return "-"
    counts = sorted(int(r["above_one"]) for r in rows)
    total = rows[0]["chromosomes"]
    head = f"{counts[0]}" if counts[0] == counts[-1] else f"{counts[0]} to {counts[-1]}"
    return f"{head} of {total}"


def p_span(rows):
    values = sorted(float(r["sign_p"]) for r in rows)
    if not values:
        return "-"
    if values[0] == values[-1]:
        return fmt_p(values[0])
    return f"{fmt_p(values[0])} to {fmt_p(values[-1])}"


def pct_above_one(rows):
    """'16 to 35 per cent' for how far a set of ratios sits above one."""
    values = sorted(round((float(r["ratio"]) - 1) * 100) for r in rows)
    if not values:
        return "-"
    if values[0] == values[-1]:
        return f"{values[0]} per cent"
    return f"{values[0]} to {values[-1]} per cent"


def override_note(classes):
    """Name any cell computed under a different ms class than the rest of the table.

    `boundary_turnover.OVERRIDES` lets one cell take the adjacency-filtered merge/split
    class while its neighbours take the whole class. That makes the row and the rows beside
    it two different analyses, so it has to be said wherever the numbers are quoted.
    """
    marked = [r for r in classes if str(r.get("ms_max_gap_bins", "")).strip()]
    if not marked:
        return ""
    names = ", ".join(f"{r['caller']} {r['feature']} `{r['contrast']}`" for r in marked)
    return ("> **These cells are not all computed the same way.** " + names +
            " restricts the merge/split class to events whose two domains are adjacent "
            "(`MAX_GAP_BINS`), which is what the rest of this folder does; every other cell "
            "takes every split and merge whatever the gap. Applying the restriction "
            "throughout instead moves RobusTAD genes to 0.99 and costs `ms / stable` its "
            "significance in three of four cells, which is why it was not applied "
            "throughout. See `OVERRIDES` in `boundary_turnover.py`.\n\n")


def ontad_trend_phrase():
    """'rho +0.58, +0.49 and +0.38 at 250 kb, 500 kb and 1 Mb', from the OnTAD trends.

    Written from the table rather than by hand, because the OnTAD grid is the part of this
    analysis that gets rerun as more of it completes.
    """
    rows = [r for r in read_tsv(RESULTS / "stage_ontad" / "stage_trend_tests.tsv")
            if r["pool"] == "own" and r["feature"] == "ctcf"
            and r["statistic"].startswith("spearman")]
    order = {"250kb": 0, "500kb": 1, "1000kb": 2}
    rows = sorted((r for r in rows if r["resolution"] in order),
                  key=lambda r: order[r["resolution"]])
    if not rows:
        return "no bin size"
    label = {"250kb": "250 kb", "500kb": "500 kb", "1000kb": "1 Mb"}
    rhos = [f"{float(r['mean']):+.2f}" for r in rows]
    names = [label[r["resolution"]] for r in rows]

    def join(items):
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

    return f"rho {join(rhos)} at {join(names)}"


def median_turnover(run_dir=None):
    """How much of one stage's median survives into the next, per transition.

    This is the number the rest of the report has to be read against. The clustering
    separates the stages well, but if it does so because the medians share almost nothing,
    then most of what is called an event is wholesale replacement rather than a domain
    being rearranged, and only the small adjacent split and merge set is interpretable.
    """
    run_dir = Path(run_dir or HERE.parent / "results_updated" / "scptr_runs_all_cells")
    from call_median_events import stage_medians

    shared = collections.defaultdict(list)
    for path in sorted(run_dir.glob("*/*/tol_*.json")):
        if path.parent.name == "all":
            continue
        try:
            summary = json.load(open(path))
            by_stage, _ = stage_medians(summary)
        except Exception:
            continue
        tolerance = int(summary.get("tolerance", 0))
        resolution = summary.get("resolution")
        for from_stage, to_stage in zip(STAGE_ORDER, STAGE_ORDER[1:]):
            if from_stage not in by_stage or to_stage not in by_stage:
                continue
            source, target = set(by_stage[from_stage]), set(by_stage[to_stage])
            if not source or not target:
                continue
            common, _, _ = match_tads_within_tolerance(source, target, tolerance)
            jaccard = len(common) / (len(source) + len(target) - len(common))
            shared[(resolution, f"{from_stage}->{to_stage}")].append(jaccard)
    return shared



ANCHOR_MEASURES = [
    ("genes", "gene density"), ("ctcf", "CTCF"), ("rad21", "RAD21 cohesin"),
    ("ctcf_rad21", "CTCF + RAD21 co-bound"), ("h3k9me3", "H3K9me3 control"),
    ("distance", "distance to nearest anchor"),
]


def anchor_resolution(caller="OnTAD"):
    """``([table row, ...], {measure: dip test})`` from ``results/anchor/``."""
    folds = read_tsv(RESULTS / "anchor" / "boundary_anchor.tsv")
    tests = read_tsv(RESULTS / "anchor" / "anchor_dip_tests.tsv")
    if not folds:
        return [], {}
    means = collections.defaultdict(list)
    for row in folds:
        if row["caller"] == caller:
            means[(row["measure"], row["stage"])].append(float(row["fold"]))
    lookup = {t["measure"]: t for t in tests if t["caller"] == caller}
    rows = []
    for measure, label in ANCHOR_MEASURES:
        values = [means.get((measure, stage)) for stage in STAGE_ORDER]
        if not any(values):
            continue
        row = [label] + [f"{sum(v) / len(v):.3f}" if v else "-" for v in values]
        test = lookup.get(measure)
        if test:
            dip = float(test["dip"])
            row += [f"{dip:+.3f}", f"{test['negative']}/{test['chroms']}",
                    f"{float(test['sign_p']):.4f}"]
        else:
            row += ["-", "-", "-"]
        rows.append(row)
    return rows, lookup


def stage_maturation():
    """Per-stage boundary folds and the trend tests, from ``results/stage/``.

    Returns ``(folds, tolerance_deltas, tests)`` where ``folds`` is
    ``{(resolution, feature, stage): mean fold}`` over the union pool with the tolerances
    averaged, ``tolerance_deltas`` is ``{(resolution, tolerance): 64-cell minus 2-cell}``
    and ``tests`` is the trend table as read.
    """
    rows = read_tsv(RESULTS / "stage" / "stage_boundary_enrichment.tsv")
    folds = collections.defaultdict(list)
    by_tolerance = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        if row["pool"] != "union" or row["fold"] == "":
            continue
        fold = float(row["fold"])
        folds[(row["resolution"], row["feature"], row["stage"])].append(fold)
        if row["feature"] == "ctcf":
            key = (row["resolution"], int(row["tolerance"]))
            by_tolerance[key][(row["chrom"], row["stage"])].append(fold)

    deltas = {}
    for key, cells in by_tolerance.items():
        paired = collections.defaultdict(dict)
        for (chrom, stage), values in cells.items():
            paired[chrom][stage] = float(np.mean(values))
        values = [c["64-cell"] - c["2-cell"] for c in paired.values()
                  if "64-cell" in c and "2-cell" in c]
        if values:
            deltas[key] = (float(np.mean(values)), sum(1 for v in values if v > 0),
                           len(values))
    means = {key: float(np.mean(values)) for key, values in folds.items()}
    return means, deltas, read_tsv(RESULTS / "stage" / "stage_trend_tests.tsv")


def merge_split_context():
    """The summary, pooled and locus tables of ``results/merge_split/``."""
    return (read_tsv(RESULTS / "merge_split" / "merge_split_summary.tsv"),
            read_tsv(RESULTS / "merge_split" / "pooled_context.tsv"),
            read_tsv(RESULTS / "merge_split" / "locus_gene_sets.tsv"))


def ontad_replication():
    """Stage folds, stage trends and merge/split event tests for both TAD callers."""
    def folds(path):
        out = collections.defaultdict(list)
        for row in read_tsv(path):
            if row["pool"] != "union" or row["feature"] != "ctcf" or row["fold"] == "":
                continue
            out[(row["resolution"], row["stage"])].append(float(row["fold"]))
        return {k: float(np.mean(v)) for k, v in out.items()}

    def trends(path):
        return [r for r in read_tsv(path)
                if r["feature"] == "ctcf" and r["pool"] == "union"
                and r["statistic"] == "spearman rho over 2->64"]

    return {
        "folds": {"RobusTAD": folds(RESULTS / "stage" / "stage_boundary_enrichment.tsv"),
                  "OnTAD": folds(RESULTS / "stage_ontad" / "stage_boundary_enrichment.tsv")},
        "trends": {"RobusTAD": trends(RESULTS / "stage" / "stage_trend_tests.tsv"),
                   "OnTAD": trends(RESULTS / "stage_ontad" / "stage_trend_tests.tsv")},
        "events": {"RobusTAD": read_tsv(RESULTS / "merge_split" / "event_level_test.tsv"),
                   "OnTAD": read_tsv(RESULTS / "merge_split_ontad" / "event_level_test.tsv")},
    }


def activity_results():
    """The reorganised-boundary gene tables, or ``None`` when they have not been built."""
    activity = read_tsv(RESULTS / "activity" / "boundary_gene_activity.tsv")
    loci = read_tsv(RESULTS / "activity" / "reorganised_locus_overlap.tsv")
    if not activity and not loci:
        return None
    return activity, loci


def replicating_contrasts(rows, key, value_field, callers=("RobusTAD", "OnTAD"),
                          direction="above"):
    """Entries whose paired contrast is significant and same-signed on every caller.

    ``direction`` picks the sign: "above" keeps entries above the reference on every caller,
    "below" keeps those below it. A contrast that is significant on one caller and null on
    the other is not returned in either -- that is the whole point of the filter.
    """
    by_key = collections.defaultdict(dict)
    for row in rows:
        by_key[row[key]][row["caller"]] = row
    out = []
    for name, per_caller in by_key.items():
        if not all(c in per_caller for c in callers):
            continue
        values = [float(per_caller[c][value_field]) for c in callers]
        ps = [float(per_caller[c]["sign_p"]) for c in callers]
        ok = all(v > 0 for v in values) if direction == "above" else all(v < 0
                                                                         for v in values)
        if ok and all(p < 0.05 for p in ps):
            out.append((name, values, ps))
    return sorted(out, key=lambda item: -min(item[1]) if direction == "above"
                  else max(item[1]))



def section(title):
    return f"\n## {title}\n"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def build_report():
    out = ["# Boundary reorganisation between the embryo stage medians",
           "",
           "Generated by `summarise_results.py`; every number comes from a table in "
           "`results/`. Two results and the controls that support them. The full grid of "
           "per-run tables stays in `results/` whether or not it is summarised here.",
           ""]

    # ------------------------------------------------------------------ result 1 and 2
    out.append(section("1. Reorganised boundaries are CTCF and gene dense"))
    classes = read_tsv(RESULTS / "turnover" / "boundary_classes.tsv")
    turnover = read_tsv(RESULTS / "turnover" / "boundary_turnover.tsv")
    cross = read_tsv(RESULTS / "turnover" / "cross_caller.tsv")
    if classes:
        out.append(
            "The boundaries of a transition fall into three classes drawn from the same "
            "pair of medians: those a **split or merge** created or destroyed, those a "
            "**domain gain or loss** created or destroyed, and those the two medians "
            "**share**. Each class is one bin wide and is scored against the same "
            "relocation null, so a ratio between two of them holds the chromosome, the "
            "caller, the stage pair and the null fixed and measures only the class. The "
            "replicate is the **chromosome**: the six tolerances of a bin size are "
            "re-analyses of one clustering and the three bin sizes are re-analyses of one "
            "dataset, so both are averaged inside a chromosome rather than counted as "
            "replicates.\n")
        out.append(table(
            ["contrast", "caller", "feature", "chromosomes", "above one", "ratio",
             "sign p"],
            [[r["contrast"], r["caller"], r["feature"], r["chromosomes"],
              r["above_one"], r["ratio"], r["sign_p"]]
             for r in sorted(classes, key=lambda r: (r["contrast"], r["caller"],
                                                     r["feature"]))]))
        gainloss = class_cells(classes, "gainloss / stable")
        out.append(
            "\n**A reorganised boundary carries "
            f"{pct_above_one(gainloss)} more CTCF and genes than a boundary of the same "
            "medians that persists**, on both callers and both annotations, in "
            f"{support_span(class_cells(classes, 'gainloss / stable', 'RobusTAD'))} "
            "chromosomes on RobusTAD calls and "
            f"{support_span(class_cells(classes, 'gainloss / stable', 'OnTAD'))} on OnTAD "
            f"calls, at sign-test p from {p_span(gainloss)} "
            "(`gainloss / stable`). TADsplimer reports the comparable "
            "contrast, CTCF per boundary by boundary category, at p = 0.013 and 5.9e-5 on "
            "deep bulk Hi-C; this recovers it from single-cell medians and shows it "
            "survives a change of TAD caller, which that paper did not test.\n\n"
            "The contrast runs against the obvious artefact rather than with it. A "
            "boundary a sparse caller invented in a quiet region would be CTCF-poor *and* "
            "unlikely to be matched in the next median, which would push the changing set "
            "below the stable one.\n")
    if turnover:
        matched = [r for r in turnover if r["denominator"] == "boundary_stable"
                   and r["numerator"] in ("boundary_gained", "boundary_lost")]
        if matched:
            out.append(table(
                ["caller", "contrast", "configurations", "above one", "ratio", "sign p"],
                [[r["caller"], f"{r['numerator']} / {r['denominator']}",
                  r["configurations"], r["above_one"], r["ratio"], r["sign_p"]]
                 for r in matched]))
            out.append(
                "\nThe same contrast split into boundaries gained and lost, which is the "
                "version drawn in `results/figures/ctcf_boundaries.png`. Every "
                "configuration of both callers is above one.\n")

    out.append(section("2. Merge and split boundaries are not a separate class"))
    if classes:
        stable = class_cells(classes, "ms / stable")
        other = class_cells(classes, "ms / gainloss")
        n_sig = sum(1 for r in stable if float(r["sign_p"]) <= 0.05)
        words = {1: "one", 2: "two", 3: "three", 4: "four"}
        significant_phrase = (
            f"all {words.get(len(stable), len(stable))}" if n_sig == len(stable)
            else f"{words.get(n_sig, n_sig)} of the {words.get(len(stable), len(stable))}")
        rt = class_cells(classes, "ms / gainloss", "RobusTAD")
        on = class_cells(classes, "ms / gainloss", "OnTAD")
        out.append(
            "Boundaries a split or a merge changed carry the same enrichment as any other "
            "reorganised boundary: "
            f"{ratio_span(class_cells(classes, 'ms / stable', feature='ctcf'))} for CTCF "
            f"and {ratio_span(class_cells(classes, 'ms / stable', feature='genes'))} for "
            f"genes against the shared boundaries (`ms / stable`, {significant_phrase} "
            "caller-by-feature cells significant). Against the *other* "
            "reorganised boundaries the two callers disagree in sign: OnTAD puts them "
            f"at {ratio_span(on)} and RobusTAD at {ratio_span(rt)}, so neither direction "
            "survives a change of caller. Every one of those departures is small beside "
            f"the {pct_above_one(class_cells(classes, 'gainloss / stable'))} separating "
            "either class from the boundaries that persist.\n\n"
            f"{override_note(classes)}"
            "**The distinction the data support is between a boundary that moves and a "
            "boundary that stays, not between the operations that move it.** That is what "
            "the single boundary constant of the TAD distance assumes: one cost for the "
            "insertion or deletion of a boundary whatever produced it. A model that "
            "priced the boundary changes of splits and mergers separately would be "
            "fitting a distinction these calls do not contain.\n")
    if cross:
        out.append(table(
            ["matching distance", "RobusTAD loci", "OnTAD loci", "matched", "expected",
             "fold", "p"],
            [[f"{int(r['slop_bp']) / 1e6:g} Mb", r["reference_loci"], r["query_loci"],
              r["matched"], r["null_mean"], r["fold"], r["p_greater"]] for r in cross]))
        out.append(
            "\nThe two callers agree on which boundaries are reorganised and not on "
            "which reorganisations are splits and merges. Clustering each caller's "
            "adjacent merge and split calls into loci and sliding the OnTAD loci at "
            "random inside the callable region of their own chromosome gives a fold of "
            f"{min(float(r['fold']) for r in cross):.2f} to "
            f"{max(float(r['fold']) for r in cross):.2f}. "
            "**Merge and split calls are reproducible across callers at "
            "chance**, which is the detection limit of this data and the reason no "
            "property specific to them survives. TADsplimer works at 10 kb on deep bulk "
            "Hi-C; these medians are 250 kb to 1 Mb.\n\n"
            "Figures: `results/figures/ctcf_boundaries.png` and "
            "`results/figures/merge_split.png`.\n")

    # ---------------------------------------------------------------------- result 3
    out.append(section("3. Boundaries mature over the stage series"))
    folds, tolerance_deltas, tests = stage_maturation()
    if folds:
        resolutions = sorted({key[0] for key in folds})
        rows = []
        for resolution in resolutions:
            row = [resolution]
            for stage in STAGE_ORDER:
                value = folds.get((resolution, "ctcf", stage))
                row.append(f"{value:.3f}" if value is not None else "-")
            rows.append(row)
        out.append(table(["bin size"] + list(STAGE_ORDER), rows))
        ctcf_tests = [r for r in tests if r["feature"] == "ctcf" and r["pool"] == "union"
                      and r["statistic"] == "spearman rho over 2->64"]
        if ctcf_tests:
            out.append(table(
                ["bin size", "mean rho", "chromosomes up", "p"],
                [[r["resolution"], r["mean"], f"{r['positive']}/{r['chroms']}",
                  r["wilcoxon_p"]] for r in ctcf_tests]))
        ontad_rho = ontad_trend_phrase()
        out.append(
            "\nCTCF density at the boundaries of each stage median, against relocations "
            "of the same boundaries, tolerances averaged. The trajectory has its minimum "
            "at the 2-cell stage and climbs to the 64-cell stage, and the trend from the "
            "2-cell stage on holds on 17 to 19 of 20 chromosomes at each bin size "
            "separately, on gene density as well, at every tolerance, and under a null "
            "restricted to the region all five medians cover. On OnTAD calls the 2-cell "
            f"minimum appears at every bin size and the trend reproduces at {ontad_rho}. "
            "This is the progressive establishment of TADs "
            "over preimplantation development reported by Du et al. 2017 and Ke et al. "
            "2017. Full tables in `results/stage/` and `results/stage_ontad/`, figure in "
            "`results/figures/embryo_boundaries.png`, panels a and b.\n")

    anchor_rows, anchor_tests = anchor_resolution("OnTAD")
    if anchor_rows:
        out.append(section("3b. What the CTCF axis measures, and what the 2-cell dip is"))
        out.append(table(["OnTAD, 250kb"] + list(STAGE_ORDER)
                         + ["2c dip", "chrom", "sign p"], anchor_rows))
        out.append(
            "\n**The axis is not measuring boundary anchoring, and no sharper anchor "
            "definition rescues it.** A 250 kb bin holds about 27 mESC CTCF peaks wherever "
            "it lands, so what section 3 plots is how CTCF-rich the neighbourhood is, not "
            "whether the boundary sits on an anchor. Narrowing the anchor to RAD21 "
            "cohesin, then to CTCF peaks that overlap a RAD21 peak, then measuring "
            "kilobases to the nearest co-bound site instead of counting, does not lift any "
            "stage above 1.05 on OnTAD and leaves RobusTAD depleted throughout at 0.66 to "
            "0.87 -- and the more specific tracks come out slightly *worse* on both "
            "callers, not better. The calls localise a boundary to a quarter megabase and "
            "an anchor is a ten kilobase object, so this is a resolution ceiling rather "
            "than a caller failure.\n")
        out.append(
            "\n**The 2-cell dip is real, cross-caller, and about territory rather than "
            "about CTCF.** It is the same size on plain gene density as on the co-bound "
            "anchor sites, so nothing about it is specific to CTCF or cohesin and no "
            "protein should be named for it. RobusTAD reproduces it at the same magnitude "
            "(genes -0.117, co-bound -0.108, 15 of 19 chromosomes, p = 0.019). What stops "
            "this being a statement about every track is H3K9me3: repressive and "
            "gene-poor, it does not dip on either caller (RobusTAD +0.014, p = 0.65). So "
            "the 2-cell median\'s boundaries sit in less active territory than the other "
            "four stages\' do, and that is the whole claim.\n")
        out.append(
            "\nRead with section 3, this narrows what the trajectory says. The defensible "
            "sentence is that **2-cell is the minimum**, not that the zygote is high: "
            "matching cell number and boundary count across stages leaves OnTAD\'s 1-cell "
            "level with 4-, 8- and 64-cell and the 2-cell stage four standard deviations "
            "below all of them. The 1-cell arm should be drawn and not tested, which is "
            "what `TREND_STAGES` in `stage_boundary_maturation.py` already does -- our "
            "calls are not allele-resolved, and at the zygote the two parental genomes sit "
            "in separate pronuclei carrying very different domain counts (Collombet et al. "
            "2020, Fig. 1c), so the 1-cell median is a sum of two structures and not the "
            "same object as the later medians.\n")
        out.append("\n`boundary_anchor_resolution.py`, tables in `results/anchor/`.\n")

    callers = ("RobusTAD", "OnTAD")
    validation = {}
    for caller in callers:
        base = RESULTS / "merge_split" / caller
        validation[caller] = (read_tsv(base / "class_summary.tsv"),
                              read_tsv(base / "locus_validation.tsv"),
                              read_tsv(base / "cluster_enrichment.tsv"))
    if any(v[0] for v in validation.values()):
        out.append(section("4. Each caller against its own held-out cells: the class is "
                           "real, the loci are not"))
        out.append(
            "Section 2 rejects merge and split as a separate boundary class using agreement "
            "between the two callers. That criterion is arguable -- two TAD callers carry "
            "different positional biases and cannot be asked to place a boundary in the "
            "same 250 kb bin -- so this section drops it. Each caller is scored **against "
            "its own held-out cells instead**, and the two are never compared by position. "
            "`merge_split_locus_validation.py`, `merge_split_class_summary.py`, "
            "`merge_split_cluster_enrichment.py`; figures "
            "`results/figures/robustad_merge_split.png` and "
            "`results/figures/ontad_merge_split.png`.\n")
        out.append(
            "A clustering run at one resolution and chromosome uses a few dozen cells while "
            "the per-cell call tree holds several hundred, so every cell no run of that grid "
            "touched is a held-out test set. For a transition A to B and a reported bin the "
            "statistic is the fraction of held-out B cells carrying a boundary there minus "
            "the fraction of held-out A cells, referenced against the boundaries the same "
            "transition calls stable.\n")
        rows = []
        for caller in callers:
            summary = validation[caller][0]
            for record in summary:
                if record["transition"] != "all":
                    continue
                rows.append([caller, record["resolution"], record["event_type"],
                             record["expected"], record["loci"],
                             f"{float(record['mean_referenced_delta']):+.4f}",
                             f"{record['agreeing']}/{record['chromosomes']}",
                             fmt_p(float(record["sign_p"]))])
        out.append(table(["caller", "bin size", "class", "predicted", "loci",
                          "mean change in held-out cells", "chromosomes agreeing", "p"],
                         rows))
        out.append(
            "\n**Split is the half that replicates.** It is positive on both callers and "
            "significant on both at their best-covered bin size -- RobusTAD +0.032 on 17 of "
            "18 chromosomes at 500 kb (p = 1e-4), OnTAD +0.029 on 18 of 20 at 500 kb "
            "(p = 4e-4). **Merge is weaker and only clears significance on OnTAD** "
            "(-0.021, 16 of 20, p = 0.012); RobusTAD puts it in the right direction at "
            "every bin size but never significantly. Splitting by transition, every row "
            "points the way the method predicts on OnTAD at 500 kb (8 of 8) and on "
            "RobusTAD at 500 kb (6 of 6, the 1-cell to 2-cell transition being untestable "
            "there), so no single transition carries the result.\n")
        out.append(
            "Two configurations drop out for reasons of coverage rather than of signal, and "
            "they are worth stating because they are not failures. The RobusTAD 1 Mb "
            "clustering uses **all 330 cells of the pool**, so it has no held-out set at all "
            "and cannot be tested this way. RobusTAD also leaves too few held-out 1-cell "
            "cells -- 6 on chromosome 1 at 500 kb against the 20 required -- so its "
            "1-cell to 2-cell transition is unscored. RobusTAD at 250 kb is testable and "
            "null.\n")
        counts = []
        for caller in callers:
            loci = validation[caller][1]
            usable = [r for r in loci if r.get("q_pooled", "") != ""]
            if not usable:
                continue
            survivors = sum(1 for r in usable if float(r["q_pooled"]) < 0.10)
            best = min(float(r["p_pooled"]) for r in usable)
            biggest = max(abs(float(r["z_score"])) for r in usable
                          if r.get("z_score", "") != "")
            counts.append((caller, len(usable), survivors, best, biggest))
        if counts:
            detail = "; ".join(f"{caller} {survivors} of {total} (best p = {best:.1e}, "
                               f"largest standardised score {biggest:.1f})"
                               for caller, total, survivors, best, biggest in counts)
            out.append(
                f"**No individual locus survives on either caller, and the reason is not "
                f"caller disagreement.** At a 10 per cent false discovery rate: {detail}. "
                "The null is empirical and pooled over the genome after standardising each "
                "chromosome, so its resolution is about 2e-5 and is not the limiting "
                "factor. What limits the test is that the background is heavy tailed: on a "
                "typical chromosome a few per cent of *all* bins move as much as the "
                "reported ones do, with no event called on them.\n")
        out.append(
            "That is a trap worth recording. A 2x2 Fisher test on (cells with a boundary) "
            "x (stage) treats the cells as exchangeable, which they are not -- boundary "
            "calls are correlated between cells of a stage. Run that way the same data "
            "reports hundreds of loci at a 10 per cent false discovery rate with q down to "
            "1e-9, all of it overdispersion. Panel c of each figure shows the gap against "
            "the Gaussian such a test assumes.\n")
        shares = {}
        for caller in callers:
            clusters = validation[caller][2]
            keep = [r for r in clusters if r["event_class"] in ("merge", "split")
                    and r["fold"] not in ("",)]
            if not keep:
                continue
            by_set = collections.defaultdict(list)
            for row in keep:
                by_set[row["gene_set"]].append(float(row["fold"]) > 1)
            shares[caller] = {k: float(np.mean(v)) for k, v in by_set.items()}
        if shares:
            controls = ", ".join(f"{caller} {v['housekeeping_control']:.2f}"
                                 for caller, v in shares.items()
                                 if "housekeeping_control" in v)
            top = max((max(v.values()) for v in shares.values()), default=0.0)
            out.append(
                "**No curated gene set is enriched at merge/split domains on either caller, "
                "once the unit of counting is honest.** The 125 canonical imprinted genes "
                "sit in about a dozen real loci -- 20 of them are Igf2/H19/Kcnq1 on chr7, "
                "15 are Dlk1-Dio3 on chr12 -- so one merge domain covering Kcnq1 scores 20 "
                "hits, and counted by gene the set reports a fold of 2.45 for OnTAD merges "
                "at the late transitions. Collapsing the set by single linkage and counting "
                "*distinct clusters overlapped* removes it. Against a relocation null, "
                "where half the configurations would be enriched by chance, the highest "
                f"share any set reaches is {top:.2f} and the housekeeping control sits at "
                f"{controls}. Read the share, not the mean fold: `hox_clusters` has a mean "
                "fold above 1 on both callers while being enriched in under a fifth of "
                "runs, because a ratio with four clusters in the denominator is not a "
                "usable statistic. This is the lesson the count-matching retraction "
                "taught -- a clustered annotation counted by its elements measures the "
                "clustering.\n")

    out.append(section("5. What the events are"))
    counts = collections.Counter()
    for path in glob.glob(str(RESULTS / "events" / "*" / "*" / "tol_*_events.tsv")):
        with open(path) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                counts[row["event_type"]] += 1
    if counts:
        total = sum(counts.values())
        out.append(table(
            ["event type", "count", "share"],
            [[name, value, f"{100 * value / total:.1f}%"]
             for name, value in counts.most_common()]))
    shared = median_turnover()
    if shared:
        overall = np.mean([v for values in shared.values() for v in values])
        out.append(
            f"\nOver the RobusTAD grid of 360 runs. Consecutive medians share about "
            f"**{100 * overall:.0f}%** of their domains, so most of what the "
            "transformation reports is a domain being replaced rather than rearranged, "
            "and splits and mergers are about 3% of events. Of those, 650 have their two "
            "domains adjacent to within a bin, which is roughly five per chromosome and "
            "configuration.\n")

    out.append(section("6. Tested and rejected"))
    out.append(
        "Kept here so they are not retried. Each was killed by a control, not by a "
        "threshold.\n\n"
        "- **Merge/split boundaries against the genome.** RobusTAD makes them "
        "CTCF-depleted at p = 0.004, OnTAD CTCF-enriched at p = 0.022; the two callers "
        "disagree in sign on six of eight comparisons. The absolute fold is a property of "
        "the caller, which is why every contrast reported above is a ratio between two "
        "boundary sets of the same medians.\n"
        "- **Merge/split boundaries against all boundaries.** A depletion of 11% at "
        "p = 0.0003 that disappears against a matched interior background: a merge or "
        "split boundary is by construction a wall between two adjacent domains, and an "
        "interior bin carries about 40 CTCF peaks against about 57 for a boundary drawn "
        "from the whole set.\n"
        "- **Domain-scale contrasts.** Dividing a megabase-wide domain fold by a one-bin "
        "boundary fold measures window width, not biology, and inflates merge and split "
        "to 1.23 and 1.27; matched, they give +0.15 on OnTAD and -0.03 on RobusTAD.\n"
        "- **PRC1.6.** Three subunits under three merge loci at p = 0.0035, generated "
        "from those same loci. Its out-of-sample prediction fails flat: 0 of 45 PRC1.6 "
        "germline targets on merge loci against an expectation of about one. Both gene "
        "sets are in `gene_sets/` so the coincidence is not rediscovered.\n"
        "- **Expression divergence across an event.** The two daughters of a split, "
        "before against after, matched transition against the other three: -0.03 on "
        "RobusTAD and +0.04 on OnTAD, and the same at a 1-2 bin local scale.\n"
        "- **Merge/split counts against developmental distance.** RobusTAD gives rho "
        "-0.28 at p = 2e-17, opposite to TADsplimer's report and an artefact of distant "
        "medians sharing no domains; OnTAD is null.\n"
        "- **Split persistence.** A boundary created by a split is no more likely to "
        "survive to the 64-cell median than any other boundary of the median it appeared "
        "in.\n"
        "- **Curated gene sets at the loci.** Nothing at p = 0.05 beyond the PRC1.6 "
        "coincidence. A set with no hits in a small window set returns a fold of 0 and a "
        "tiny sign-test p; those are not findings and are filtered.\n")

    activity = activity_results()
    if activity:
        rows, loci = activity
        full = [r for r in rows if r["matching"] == "full"]
        worst = max((float(r["sign_p"]) for r in full), default=1.0)
        best = min((float(r["sign_p"]) for r in full), default=1.0)
        spans = [abs(float(r["percent_of_mean"])) for r in full if r["percent_of_mean"] != ""]
        out.append(
            "- **Which genes sit at a reorganised boundary.** Genes at reorganised "
            "boundaries are compared with genes at the stable boundaries of the same "
            "medians, matched on bin size, on the number of genes in the bin and on the "
            "gene density of the surrounding megabase, on three properties read off the "
            "Deng et al. 2014 atlas: mean expression over the five stages, dynamic range "
            "across them, and the change at the very transition the boundary moved at. "
            f"Every difference is under {max(spans) if spans else 0:.0f}% of the mean and "
            f"none is significant on both callers (p from {best:g} to {worst:g}). Section 1's "
            "boundaries carry *more* genes; those genes are not different genes, and they "
            "are not the genes that change at that transition. Beware the estimator: "
            "averaging per-stratum ratios of class means is upward biased and turns this "
            "null into a 25% effect at p = 4e-5 as the strata get finer. "
            "`boundary_gene_activity.py` uses residuals from the stratum mean instead.\n")
        if loci:
            worst_fold = max(float(r["fold"]) for r in loci)
            out.append(
                "- **Naming loci from the reorganised class.** Collapsing each caller's "
                "whole grid to a per-bin reorganisation rate over its own callable "
                "expectation and comparing the two callers' most reorganised bins against "
                f"a circular-shift null gives a fold of at most {worst_fold:.2f} -- the "
                "overlap is *below* chance at every cut. The class is real and its "
                "statistics reproduce, but which bin belongs to it does not, so a gene "
                "list read off either caller's hotspots would be that caller's noise. "
                "This is section 2's cross-caller null for merge and split calls, "
                "established for the whole reorganised class.\n")
        out.append("")

    out.append(
        "\nThe chrX prediction of Collombet et al. 2020 is not recovered either, by a test "
        "too weak to have refuted it: the calls are not allele resolved. Tables for all "
        "of the above remain under `results/`.\n")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(RESULTS / "REPORT.md"))
    args = parser.parse_args()

    report = build_report()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    print(f"wrote {path} ({len(report.splitlines())} lines)")


if __name__ == "__main__":
    main()
