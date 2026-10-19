#!/usr/bin/env python3
"""Weighting merge/split calls between what the structure says and what biology says.

The merge/split analysis of this folder is negative: no individual locus survives a
held-out test, and the class-level signal is a couple of per cent of penetrance. What this
script does instead of taking the called set as fixed is put a weight ``alpha`` on it,
running from 0 to 1:

* ``alpha = 0`` ranks the called merge/split events by **our own evidence alone** -- the
  held-out per-cell penetrance shift of ``merge_split_locus_validation.py``, computed on
  cells no clustering run ever saw and using no annotation of any kind;
* ``alpha = 1`` ranks the same events by **what a boundary is supposed to look like** --
  CTCF together with cohesin, the loop-extrusion definition of a mammalian TAD boundary;
* in between, by the convex combination of the two rank percentiles.

The top ``K`` events of each ranking form a set of fixed size, so the sets at different
alpha are directly comparable, and the question is whether a set built from *both* sources
carries biology that neither source produces alone.

Why this is not circular
------------------------
The anchor is CTCF and RAD21 in mouse ES cells, and both tracks are **stage blind**: one
number per genomic position, identical at every embryo stage. Every readout here is the
opposite, a **transition matched** quantity. An event called at the 2-cell to 4-cell
transition is scored on what its genes do across 2-cell to 4-cell *relative to what the
same genes do across the other three transitions*. No static property -- CTCF, gene
density, promoter density, GC content -- can produce that contrast, because it enters all
four transitions equally and cancels in the difference.

The two ends of the sweep are therefore the controls that make the middle readable. If
``alpha = 1`` already produces the contrast, the structural calls added nothing and the
finding is about CTCF sites. If ``alpha = 0`` already produces it, the anchor added
nothing. The result worth having is a peak strictly inside ``(0, 1)`` that clears both
ends and both shuffling controls.

Two readouts
------------
``--readout dynamism`` (the boundary window)
    Genes whose start site falls in the changed boundary window. For each transition every
    gene's absolute log2 fold change is turned into its genome-wide percentile -- the four
    transitions are not equally eventful, zygotic genome activation moves most of the
    transcriptome and the 8-cell to 64-cell step much less, so raw magnitudes would score
    an event by which transition it happened to be called at. The contrast is the mean
    percentile at the event's own transition minus the mean over the other three.

``--readout crossboundary`` (the boundary as a boundary)
    The readout the event type actually predicts, and the one thing that distinguishes a
    merge or a split from a domain appearing or disappearing. A split puts a new boundary
    between two stretches of chromatin that were in one domain; a merge removes the
    boundary between two that were separate. So take the genes immediately on each side of
    the changed boundary and ask how far apart their expression moves *across that
    transition*:

        divergence(stage) = | mean expression percentile of the left side
                           - mean expression percentile of the right side |
        delta(transition) = divergence(after) - divergence(before)

    A split should raise it and a merge should lower it, so the statistic is multiplied by
    +1 for a split and -1 for a merge and the prediction becomes positive for both. As with
    the other readout, the event's own transition is scored against the other three, which
    removes both the event's own baseline divergence and the general drift of the embryo.

The null
--------
Events are not independent: the events of one chromosome at one transition come from a
single pair of medians of a single clustering run and share cells, calls and
neighbourhood. The null therefore permutes **the transition labels as a block within each
(resolution, chromosome)** -- every event of a chromosome called at transition A becomes
transition B together. The gene content of every window, the number of events per
transition and the correlation between events from one pair of medians all survive it, and
only the correspondence between a block of events and the developmental transition it was
called at is destroyed. Permuting one event at a time instead would treat the several
events a single pair of medians produced as independent replicates, and would read as
significant far too easily.

Because the alpha grid is scanned, the headline p-value is a max statistic: the whole curve
is rebuilt under every permutation and the largest standardised departure anywhere on it is
recorded, so scanning eleven values of alpha is paid for rather than ignored. Standardising
before the maximum matters, because the null spread shrinks as alpha rises -- the selected
sets overlap more between permutations there -- and a raw maximum would simply find the
noisiest end of the curve.

Two further controls sit next to the curve. ``shuffled-S`` rebuilds it with the structural
ranking permuted within chromosome and ``shuffled-B`` with the anchor permuted; a peak that
appears in either is a property of the ranking arithmetic and not of the data.

The machinery is checked with ``--self-test``, which assigns every event the transition its
own genes move most at and confirms the statistic and the null recover a planted signal.

Usage:
    python merge_split_weighted_biology.py --caller OnTAD
    python merge_split_weighted_biology.py --caller OnTAD --readout crossboundary
    python merge_split_weighted_biology.py --caller OnTAD --resolution 500kb --self-test
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path

import numpy as np

from annotate_events import GeneIndex
from cell_calls import SOURCES
import splithalf_penetrance
from gene_sets import (CHROMATIN_MARKS, STAGE_ORDER, TRANSITIONS,
                       derive_transition_modules, load_curated_sets, load_expression,
                       load_gene_table, load_mark, stage_level, transition_change)

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "results" / "weighted"

RESOLUTIONS = ["250kb", "500kb", "1000kb"]
RESOLUTION_BP = {"250kb": 250_000, "500kb": 500_000, "1000kb": 1_000_000}
TOLERANCES = range(6)
# the anchor: the loop-extrusion definition of a boundary. Both tracks are excluded from
# every readout, so nothing downstream can be a restatement of the anchor.
ANCHOR_MARKS = ("ctcf", "rad21")
# a split should gain per-cell penetrance in the later stage, a merge should lose it, and
# a split should push the two sides of its new boundary apart while a merge pulls them
# together. Both are multiplied through so that "larger is as predicted" everywhere.
EXPECTED_SIGN = {"split": +1.0, "merge": -1.0}
ALPHA_GRID = [round(0.1 * i, 1) for i in range(11)]

CURVE_FIELDS = ["readout", "resolution", "event_type", "top_k", "alpha", "n_events",
                "n_genes", "contrast", "contrast_gene_weighted", "secondary_contrast",
                "null_mean", "null_sd", "z", "p_pointwise", "secondary_p"]
CONTROL_FIELDS = ["readout", "resolution", "event_type", "top_k", "control", "alpha",
                  "contrast", "null_p05", "null_p95"]
PANEL_FIELDS = ["readout", "resolution", "event_type", "top_k", "alpha", "track",
                "observed_per_bin", "pool_per_bin", "fold", "p_two_sided"]
SET_FIELDS = ["readout", "resolution", "event_type", "top_k", "alpha", "gene_set",
              "events_hit", "pool_expected", "fold", "p_greater"]
SUMMARY_FIELDS = ["readout", "resolution", "event_type", "top_k", "n_pool", "best_alpha",
                  "best_contrast", "best_z", "contrast_at_0", "z_at_0", "contrast_at_1",
                  "z_at_1", "global_p", "interior_p", "beats_both_ends", "n_blocks"]


# --------------------------------------------------------------------------- the events


def load_structural(caller, validation=None):
    """``{(resolution, chrom, transition, event_type, bin): directed z}``.

    ``merge_split_locus_validation.py`` writes one row per merge/split boundary bin with
    the per-bin penetrance shift between the held-out cells of the two stages, standardised
    against that chromosome's own background. It is out-of-sample evidence for the event
    from the same caller and carries no annotation, which is what makes it the structural
    end of the sweep.
    """
    path = Path(validation) if validation else (
        HERE / "results" / "merge_split" / caller / "locus_validation.tsv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run merge_split_locus_validation.py --caller {caller}.")
    table = {}
    with open(path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                z = float(row["z_score"])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(z):
                continue
            key = (row["resolution"], row["chrom"], row["transition"],
                   row["event_type"], int(row["bin"]))
            table[key] = EXPECTED_SIGN[row["event_type"]] * z
    return table


def parse_bins(text):
    """``'29-32;32-34'`` to ``[(29, 32), (32, 34)]``."""
    out = []
    for part in str(text).split(";"):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        out.append((int(a), int(b)))
    return out


def load_events(events_dir, resolution, event_types, max_gap_bins, structural, chroms=None):
    """Distinct merge and split events of one resolution, with their structural score.

    The six tolerances of a bin size are re-analyses of one clustering, so an event called
    at several of them is one event and is kept once; ``tolerances`` records how many found
    it. ``max_gap_bins`` keeps only the events whose two domains are adjacent to within
    that many bins, which are the ones where a single boundary demonstrably appeared or
    disappeared -- a split whose daughters are megabases apart replaced a domain with two
    distant pieces and its edges are not one boundary change.
    """
    root = Path(events_dir) / resolution
    if not root.exists():
        raise FileNotFoundError(f"no events under {root}")
    resolution_bp = RESOLUTION_BP[resolution]
    merged = {}
    for path in sorted(root.glob("*/tol_*_events.tsv")):
        chrom = path.parent.name
        if chroms and chrom not in chroms:
            continue
        with open(path) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["event_type"] not in event_types:
                    continue
                gap = row.get("gap_bins", "")
                if gap == "" or int(gap) > max_gap_bins:
                    continue
                partners = parse_bins(row["partner_bins"])
                if len(partners) != 2:
                    continue
                left, right = sorted(partners)
                key = (chrom, row["transition"], row["event_type"], left, right)
                if key in merged:
                    merged[key]["tolerances"] += 1
                    continue
                edges = sorted({int(v) for v in str(row["focal_bins"]).split(";") if v})
                merged[key] = {
                    "resolution": resolution, "chrom": chrom,
                    "transition": row["transition"], "event_type": row["event_type"],
                    "left": left, "right": right, "edges": edges,
                    "gap_bins": int(gap), "tolerances": 1,
                    "sign": EXPECTED_SIGN[row["event_type"]],
                    # the anchor and the dynamism readout both look at the changed
                    # boundary itself: the bins the transformation charged for
                    "start": (min(edges) - 1) * resolution_bp,
                    "end": max(edges) * resolution_bp,
                }

    events = []
    for key, event in merged.items():
        scores = [structural[(event["resolution"], event["chrom"], event["transition"],
                              event["event_type"], b)]
                  for b in event["edges"]
                  if (event["resolution"], event["chrom"], event["transition"],
                      event["event_type"], b) in structural]
        if not scores:
            # no held-out evidence for either edge: the structural end of the sweep has
            # nothing to say about this event, so it cannot take part in the ranking
            continue
        event["structural"] = float(np.mean(scores))
        events.append(event)
    return events


def peak_counter(intervals):
    """Counts peaks whose start falls in ``[start, end)`` on one chromosome."""
    starts = {c: [a for a, _ in iv] for c, iv in (intervals or {}).items()}

    def count(chrom, start, end):
        positions = starts.get(chrom)
        if not positions:
            return 0
        return bisect.bisect_left(positions, end) - bisect.bisect_left(positions, start)
    return count


def attach_anchor(events, marks, anchor=ANCHOR_MARKS):
    """The anchor score: mean rank percentile of the architectural tracks at the boundary.

    Each track is rank-transformed **within (resolution, chromosome)** before averaging,
    because CTCF density varies several fold between chromosomes and an unnormalised sum
    would rank chromosomes rather than events.
    """
    counters = {name: peak_counter(marks[name]) for name in anchor if marks.get(name)}
    if not counters:
        raise RuntimeError("no architectural track on disk; run fetch_reference_data.py")
    for event in events:
        for name, count in counters.items():
            event[f"track_{name}"] = count(event["chrom"], event["start"], event["end"])
    for name in counters:
        rank_within_chromosome(events, f"track_{name}", f"rank_{name}")
    for event in events:
        event["biological"] = float(np.mean([event[f"rank_{n}"] for n in counters]))
    rank_within_chromosome(events, "structural", "rank_structural")
    rank_within_chromosome(events, "biological", "rank_biological")
    return sorted(counters)


def rank_within_chromosome(events, field, out_field):
    """Rank percentile of ``field`` inside each (resolution, chromosome) group, ties averaged."""
    groups = {}
    for i, event in enumerate(events):
        groups.setdefault((event["resolution"], event["chrom"]), []).append(i)
    for indices in groups.values():
        values = np.array([events[i][field] for i in indices], dtype=float)
        order = values.argsort(kind="mergesort")
        ranks = np.empty(len(values))
        ranks[order] = np.arange(len(values))
        for value in np.unique(values):
            mask = values == value
            if mask.sum() > 1:
                ranks[mask] = ranks[mask].mean()
        denominator = max(len(values) - 1, 1)
        for i, rank in zip(indices, ranks):
            events[i][out_field] = float(rank) / denominator


# ------------------------------------------------------------------------ expression


def expressed_genes(expression, genes, min_level):
    """Atlas genes the mm10 table can place, restricted to those ever seen expressed.

    Just under half of the atlas sits below log2(RPKM + 1) = 1 at every stage, so its fold
    changes are ratios of noise. Those genes cannot carry a transcriptional response and
    including them only adds variance, so they are dropped **before** anything is ranked.
    The cut is on expression level alone, a property of the gene and not of any event, so
    it cannot select for or against the hypothesis.
    """
    keep = set()
    for symbol in expression:
        if symbol not in genes:
            continue
        levels = [stage_level(expression, symbol, stage) for stage in STAGE_ORDER]
        if any(v is None for v in levels):
            continue
        if max(levels) >= min_level:
            keep.add(symbol)
    return keep


def percentile_map(values, symbols):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    denominator = max(len(values) - 1, 1)
    return {s: float(r) / denominator for s, r in zip(symbols, ranks)}


def transition_percentiles(expression, measured):
    """``{transition: {symbol: percentile of |log2 fold change| among measured genes}}``."""
    symbols = sorted(measured)
    return {t: percentile_map(
        np.array([abs(transition_change(expression, s, t)) for s in symbols]), symbols)
        for t in TRANSITIONS}


def stage_percentiles(expression, measured):
    """``{stage: {symbol: percentile of expression level among measured genes}}``.

    Levels are ranked within a stage so that stages of different sequencing depth are on
    one scale, which is what lets the two sides of a boundary be compared across stages.
    """
    symbols = sorted(measured)
    return {stage: percentile_map(
        np.array([stage_level(expression, s, stage) for s in symbols]), symbols)
        for stage in STAGE_ORDER}


def module_rates(modules, measured):
    """The genome-wide fraction of measured genes in each transition's response module."""
    rates = {}
    for transition in TRANSITIONS:
        members = (modules[f"up@{transition}"] | modules[f"down@{transition}"]) & measured
        rates[transition] = len(members) / max(len(measured), 1)
    return rates


def matched_minus_other(levels):
    """``levels[i, t]`` to ``levels[i, t] - mean of levels[i, t'] over t' != t``."""
    k = levels.shape[1]
    totals = levels.sum(axis=1, keepdims=True)
    return levels - (totals - levels) / (k - 1)


# --------------------------------------------------------------------- the two readouts


def dynamism_matrices(events, index, percentiles, modules, measured, rates, flank_bp=0):
    """Readout 1: how dynamic the genes in the changed boundary window are.

    ``contrast[i, t]`` is what event ``i`` scores if it is assigned transition ``t``: the
    mean genome-wide dynamism percentile of the genes in its boundary window at ``t``,
    minus the mean of the same quantity over the other three transitions. The secondary
    matrix is the same on response-module membership, each transition centred on its own
    genome-wide rate so that modules of different sizes are comparable.
    """
    n = len(events)
    levels = np.full((n, len(TRANSITIONS)), np.nan)
    secondary = np.full((n, len(TRANSITIONS)), np.nan)
    weights = np.zeros(n)
    for i, event in enumerate(events):
        start = max(0, event["start"] - flank_bp)
        end = event["end"] + flank_bp
        symbols = [g for g in index.in_window(event["chrom"], start, end) if g in measured]
        event["genes"] = symbols
        if not symbols:
            continue
        weights[i] = len(symbols)
        for j, transition in enumerate(TRANSITIONS):
            levels[i, j] = np.mean([percentiles[transition][g] for g in symbols])
            member = modules[f"up@{transition}"] | modules[f"down@{transition}"]
            hit = sum(1.0 for g in symbols if g in member) / len(symbols)
            secondary[i, j] = hit - rates[transition]
    return matched_minus_other(levels), matched_minus_other(secondary), weights, weights > 0


def crossboundary_matrices(events, index, stage_pct, transition_pct, expression, measured,
                           side_bins):
    """Readout 2: does the changed boundary separate or join two expression territories?

    For each event the genes immediately on each side of the changed boundary are taken --
    at most ``side_bins`` bins, clipped to the domain that side belongs to -- and the two
    sides are compared stage by stage:

        divergence(stage) = | mean level percentile left - mean level percentile right |
        delta(transition) = divergence(after) - divergence(before)

    multiplied by +1 for a split and -1 for a merge, so that "the sides moved apart when
    the split happened" and "the sides came together when the merge happened" are both
    positive. The matrix is then the matched-minus-other contrast of that quantity, which
    removes the event's own baseline divergence and the general drift of the embryo.

    The secondary matrix does the same on the *responses* rather than the levels: how
    differently the two sides move across a transition, again signed by event type.
    """
    n = len(events)
    levels = np.full((n, len(TRANSITIONS)), np.nan)
    secondary = np.full((n, len(TRANSITIONS)), np.nan)
    weights = np.zeros(n)
    stage_index = {s: i for i, s in enumerate(STAGE_ORDER)}
    for i, event in enumerate(events):
        resolution_bp = RESOLUTION_BP[event["resolution"]]
        (a1, b1), (a2, b2) = event["left"], event["right"]
        left_window = (max(a1, b1 - side_bins + 1), b1)
        right_window = (a2, min(b2, a2 + side_bins - 1))
        left = [g for g in index.in_window(event["chrom"], (left_window[0] - 1) * resolution_bp,
                                           left_window[1] * resolution_bp) if g in measured]
        right = [g for g in index.in_window(event["chrom"], (right_window[0] - 1) * resolution_bp,
                                            right_window[1] * resolution_bp) if g in measured]
        event["genes"] = left + right
        event["side_genes"] = (len(left), len(right))
        if not left or not right:
            continue
        weights[i] = min(len(left), len(right))
        divergence = []
        for stage in STAGE_ORDER:
            a = np.mean([stage_pct[stage][g] for g in left])
            b = np.mean([stage_pct[stage][g] for g in right])
            divergence.append(abs(a - b))
        for j, transition in enumerate(TRANSITIONS):
            before, after = transition.split("->")
            levels[i, j] = event["sign"] * (divergence[stage_index[after]]
                                            - divergence[stage_index[before]])
            response_left = np.mean([transition_change(expression, g, transition)
                                     for g in left])
            response_right = np.mean([transition_change(expression, g, transition)
                                      for g in right])
            secondary[i, j] = event["sign"] * abs(response_left - response_right)
    return matched_minus_other(levels), matched_minus_other(secondary), weights, weights > 0


READOUTS = ["dynamism", "crossboundary", "replication"]


# ------------------------------------------------------------------ readout 3


def load_replication(caller, resolutions, event_types, out_dir=None, rebuild=False):
    """The split-half held-out table, built if it is not already cached.

    Returns the loci of the requested resolutions and event types, each carrying ``z_a``
    (the half used to rank) and ``z_b`` (the disjoint half used to judge), plus the half-B
    background of every (resolution, chromosome, transition) for the null.
    """
    cached = None if rebuild else splithalf_penetrance.load_cached(caller, out_dir)
    if cached is None:
        rows, background = splithalf_penetrance.build(caller, quiet=True)
        splithalf_penetrance.save(caller, rows, background, out_dir)
    else:
        rows, background = cached
    loci = []
    for row in rows:
        if row["resolution"] not in resolutions or row["event_type"] not in event_types:
            continue
        loci.append(dict(row, structural=row["z_a"], readout=row["z_b"],
                         sign=EXPECTED_SIGN[row["event_type"]], genes=()))
    return loci, background


def replication_blocks(loci):
    """Indices grouped by (resolution, chromosome, transition).

    Finer than the chromosome blocks the other readouts use, because the statistic being
    permuted here is a per-bin quantity computed inside one transition of one chromosome,
    and the block is the set it was standardised against.
    """
    groups = {}
    for i, locus in enumerate(loci):
        key = (locus["resolution"], locus["chrom"], locus["transition"])
        groups.setdefault(key, []).append(i)
    return [np.array(v, dtype=int) for v in groups.values()]


def permute_within_blocks(values, blocks, rng):
    out = values.copy()
    for members in blocks:
        out[members] = rng.permutation(values[members])
    return out


def replication_sweep(loci, top_k, structural, biological, tiebreak, alphas=ALPHA_GRID,
                      permutations=2000, seed=0):
    """Does the ranking pick the merge/split calls that reappear in held-out cells?

    The readout is the mean directed half-B evidence of the selected set. The null permutes
    that evidence **within (resolution, chromosome, transition)**, which keeps every block
    mean, every chromosome's composition and the heavy tail of the statistic exactly as
    observed and breaks only the link between a locus's rank and how well it replicates.
    The block means are held fixed on purpose: the claim is that the ranking orders loci
    inside a run, not that it picks better runs.

    The direction is predicted here -- a good ranking should raise the replication -- so the
    pointwise p-value is one sided, unlike the two readouts above.
    """
    readout = np.array([l["readout"] for l in loci])
    blocks = replication_blocks(loci)

    selections, observed = [], []
    for alpha in alphas:
        selection = select(structural, biological, alpha, top_k, tiebreak)
        selections.append(selection)
        observed.append({"alpha": alpha, "contrast": float(readout[selection].mean()),
                         "contrast_gene_weighted": float(np.median(readout[selection])),
                         "secondary_contrast": float((readout[selection] > 0).mean()),
                         "n_events": int(selection.size), "n_genes": 0})

    rng = np.random.default_rng(seed)
    null = np.empty((permutations, len(alphas)))
    second_null = np.empty((permutations, len(alphas)))
    for p in range(permutations):
        permuted = permute_within_blocks(readout, blocks, rng)
        for j, selection in enumerate(selections):
            null[p, j] = permuted[selection].mean()
            second_null[p, j] = (permuted[selection] > 0).mean()

    centre = null.mean(axis=0)
    spread = null.std(axis=0, ddof=1)
    spread = np.where(spread > 0, spread, np.nan)
    values = np.array([e["contrast"] for e in observed])
    standardised_obs = (values - centre) / spread
    standardised_null = (null - centre) / spread

    for j, entry in enumerate(observed):
        entry["null_mean"] = float(centre[j])
        entry["null_sd"] = float(spread[j]) if np.isfinite(spread[j]) else float("nan")
        entry["z"] = float(standardised_obs[j])
        entry["p_pointwise"] = float((np.sum(null[:, j] >= entry["contrast"]) + 1)
                                     / (permutations + 1))
        entry["secondary_p"] = float(
            (np.sum(second_null[:, j] >= entry["secondary_contrast"]) + 1)
            / (permutations + 1))

    interior = [j for j, a in enumerate(alphas) if 0.0 < a < 1.0]
    max_obs = np.nanmax(standardised_obs)
    global_p = float((np.sum(np.nanmax(standardised_null, axis=1) >= max_obs) + 1)
                     / (permutations + 1))
    if interior:
        interior_obs = np.nanmax(standardised_obs[interior])
        interior_null = np.nanmax(standardised_null[:, interior], axis=1)
        interior_p = float((np.sum(interior_null >= interior_obs) + 1) / (permutations + 1))
    else:
        interior_p = float("nan")
    return observed, selections, global_p, interior_p, len(blocks)


def replication_control(loci, top_k, structural, biological, tiebreak, which,
                        alphas=ALPHA_GRID, repeats=200, seed=1):
    """The replication curve rebuilt with one of the two rankings permuted."""
    readout = np.array([l["readout"] for l in loci])
    blocks = chromosome_blocks(loci)
    rng = np.random.default_rng(seed)
    curves = np.empty((repeats, len(alphas)))
    for r in range(repeats):
        s, b = structural.copy(), biological.copy()
        target = s if which == "shuffled-S" else b
        for members in blocks:
            target[members] = rng.permutation(target[members])
        for j, alpha in enumerate(alphas):
            curves[r, j] = readout[select(s, b, alpha, top_k, tiebreak)].mean()
    return curves


# ------------------------------------------------------------------------- the sweep


def transition_indices(events):
    lookup = {t: j for j, t in enumerate(TRANSITIONS)}
    return np.array([lookup[e["transition"]] for e in events], dtype=int)


def chromosome_blocks(events):
    groups = {}
    for i, event in enumerate(events):
        groups.setdefault((event["resolution"], event["chrom"]), []).append(i)
    return [np.array(v, dtype=int) for v in groups.values()]


def select(structural, biological, alpha, top_k, tiebreak):
    """Indices of the ``top_k`` events under the convex combination of the two rankings.

    ``tiebreak`` is a fixed arbitrary offset far below the score's own scale, so that rank
    plateaus -- and at alpha 0 and 1 there are plateaus, because ranks tie -- resolve the
    same way on every call instead of following the sort's treatment of equal keys.
    """
    score = (1.0 - alpha) * structural + alpha * biological + tiebreak
    return np.argsort(-score, kind="mergesort")[:top_k]


def block_permutation(blocks, columns, rng):
    """Relabel the transitions as a block inside each (resolution, chromosome)."""
    out = columns.copy()
    for members in blocks:
        present = np.unique(columns[members])
        if present.size < 2:
            continue
        mapping = dict(zip(present.tolist(), rng.permutation(present).tolist()))
        out[members] = [mapping[c] for c in columns[members]]
    return out


def score_selection(selection, columns, contrast, secondary, weights, valid):
    """Mean contrast over a selection under one assignment of transitions to events."""
    keep = selection[valid[selection]]
    if keep.size == 0:
        return float("nan"), float("nan"), float("nan"), 0, 0
    values = contrast[keep, columns[keep]]
    second = secondary[keep, columns[keep]]
    w = weights[keep]
    return (float(np.nanmean(values)),
            float(np.nansum(values * w) / np.nansum(w)),
            float(np.nanmean(second)),
            int(keep.size), int(np.nansum(w)))


def sweep(events, top_k, matrices, structural, biological, tiebreak, alphas=ALPHA_GRID,
          permutations=2000, seed=0):
    """The alpha curve with its pointwise and max-statistic null."""
    contrast, secondary, weights, valid = matrices
    columns = transition_indices(events)
    blocks = chromosome_blocks(events)

    selections, observed = [], []
    for alpha in alphas:
        selection = select(structural, biological, alpha, top_k, tiebreak)
        selections.append(selection)
        value, weighted, second, n_events, n_genes = score_selection(
            selection, columns, contrast, secondary, weights, valid)
        observed.append({"alpha": alpha, "contrast": value,
                         "contrast_gene_weighted": weighted,
                         "secondary_contrast": second,
                         "n_events": n_events, "n_genes": n_genes})

    rng = np.random.default_rng(seed)
    null = np.empty((permutations, len(alphas)))
    second_null = np.empty((permutations, len(alphas)))
    for p in range(permutations):
        permuted = block_permutation(blocks, columns, rng)
        for j, selection in enumerate(selections):
            scored = score_selection(selection, permuted, contrast, secondary,
                                     weights, valid)
            null[p, j], second_null[p, j] = scored[0], scored[2]

    # Standardise before taking the maximum. The null spread shrinks as alpha rises,
    # because the selected sets overlap more between permutations there, so a raw maximum
    # would find the noisiest end of the curve rather than the signal.
    centre = np.nanmean(null, axis=0)
    spread = np.nanstd(null, axis=0, ddof=1)
    spread = np.where(np.isfinite(spread) & (spread > 0), spread, np.nan)
    standardised_obs = np.abs((np.array([e["contrast"] for e in observed]) - centre) / spread)
    standardised_null = np.abs((null - centre) / spread)

    for j, entry in enumerate(observed):
        column = null[:, j][np.isfinite(null[:, j])]
        entry["null_mean"] = float(centre[j])
        entry["null_sd"] = float(spread[j]) if np.isfinite(spread[j]) else float("nan")
        entry["z"] = (float((entry["contrast"] - centre[j]) / spread[j])
                      if np.isfinite(spread[j]) else float("nan"))
        # two sided: the direction of the dynamism readout is not predicted, and reporting
        # it one sided would hide a departure that points the other way
        entry["p_pointwise"] = (float((np.sum(np.abs(column - centre[j])
                                              >= abs(entry["contrast"] - centre[j])) + 1)
                                      / (column.size + 1)) if column.size else float("nan"))
        scolumn = second_null[:, j][np.isfinite(second_null[:, j])]
        entry["secondary_p"] = (
            float((np.sum(np.abs(scolumn - scolumn.mean())
                          >= abs(entry["secondary_contrast"] - scolumn.mean())) + 1)
                  / (scolumn.size + 1)) if scolumn.size else float("nan"))

    interior = [j for j, a in enumerate(alphas) if 0.0 < a < 1.0]
    max_obs = np.nanmax(standardised_obs)
    global_p = float((np.sum(np.nanmax(standardised_null, axis=1) >= max_obs) + 1)
                     / (permutations + 1))
    if interior:
        interior_obs = np.nanmax(standardised_obs[interior])
        interior_null = np.nanmax(standardised_null[:, interior], axis=1)
        interior_p = float((np.sum(interior_null >= interior_obs) + 1) / (permutations + 1))
    else:
        interior_p = float("nan")
    return observed, selections, global_p, interior_p, len(blocks)


def shuffled_control(events, top_k, matrices, structural, biological, tiebreak, which,
                     alphas=ALPHA_GRID, repeats=200, seed=1):
    """The whole curve rebuilt with one of the two rankings permuted within chromosome.

    A peak the ranking arithmetic can produce on its own -- because the two scores are
    correlated, or because a fixed-size top-K of a mixture behaves oddly near the ends --
    appears here too, and is then not a result. The transition labels are left alone, so
    this control asks about the selection and not about the readout.
    """
    contrast, secondary, weights, valid = matrices
    columns = transition_indices(events)
    blocks = chromosome_blocks(events)
    rng = np.random.default_rng(seed)
    curves = np.empty((repeats, len(alphas)))
    for r in range(repeats):
        s, b = structural.copy(), biological.copy()
        target = s if which == "shuffled-S" else b
        for members in blocks:
            target[members] = rng.permutation(target[members])
        for j, alpha in enumerate(alphas):
            selection = select(s, b, alpha, top_k, tiebreak)
            curves[r, j] = score_selection(selection, columns, contrast, secondary,
                                           weights, valid)[0]
    return curves


def self_test(events, matrices, permutations=500, seed=0):
    """Plant the signal the readout is meant to find and check it is found.

    Every event is assigned the transition its own genes score highest at, which is the
    strongest transition-matched signal the data could carry. If the statistic and the
    block permutation cannot separate that from the null, a null result elsewhere says
    nothing about biology and everything about the test.
    """
    contrast, secondary, weights, valid = matrices
    columns = transition_indices(events)
    blocks = chromosome_blocks(events)
    planted = columns.copy()
    planted[valid] = np.nanargmax(contrast[valid], axis=1)
    selection = np.arange(len(events))
    observed = score_selection(selection, planted, contrast, secondary, weights, valid)[0]
    rng = np.random.default_rng(seed)
    null = np.array([score_selection(selection, block_permutation(blocks, planted, rng),
                                     contrast, secondary, weights, valid)[0]
                     for _ in range(permutations)])
    spread = null.std(ddof=1)
    return observed, float(null.mean()), float(spread), \
        float((observed - null.mean()) / spread) if spread else float("nan")


# ------------------------------------------------------- descriptive secondary reads


def selection_null(events, size, structural, biological, alpha, tiebreak, draws, rng):
    """Selections of the same size drawn by permuting the ranking **within chromosome**.

    Drawing a plain random subset of the pool would let a chromosome that happens to be
    both event rich and gene rich carry the fold. Permuting the ranking inside each
    chromosome instead gives every draw the per-chromosome composition a random ranking
    would give, which is the composition the observed selection is entitled to.
    """
    blocks = chromosome_blocks(events)
    for _ in range(draws):
        s = permute_within_blocks(structural, blocks, rng)
        b = permute_within_blocks(biological, blocks, rng)
        yield select(s, b, alpha, size, tiebreak)


def panel_folds(events, selection, marks, extra, draws_iter):
    """Per-window density of each panel track in the selected set against the null draws.

    ``gene_density`` is included on purpose and has to be read first. Every one of these
    tracks correlates with it, so a set that is gene rich is enriched for all of them at
    once and a set that is gene poor is depleted for all of them at once; only a track that
    departs from the gene-density fold says anything of its own. The two anchor tracks are
    reported too rather than hidden, since at high alpha their fold is the selection
    restated and seeing that makes the rest readable.
    """
    tracks = {name: peak_counter(iv) for name, iv in marks.items() if iv}
    values = {name: np.array([count(e["chrom"], e["start"], e["end"]) for e in events],
                             dtype=float) for name, count in tracks.items()}
    values.update(extra)
    observed = {name: float(v[selection].mean()) for name, v in values.items()}
    null = {name: [] for name in values}
    for draw in draws_iter:
        for name, v in values.items():
            null[name].append(v[draw].mean())
    rows = []
    for name, v in values.items():
        column = np.asarray(null[name])
        centre = float(column.mean())
        p = (min(int((column >= observed[name]).sum()),
                 int((column <= observed[name]).sum())) + 1) / (column.size + 1)
        rows.append({"track": name, "observed_per_bin": round(observed[name], 3),
                     "pool_per_bin": round(centre, 3),
                     "fold": round(observed[name] / centre, 4) if centre else "",
                     "p_two_sided": round(float(min(1.0, 2 * p)), 5)})
    return rows


def gene_set_folds(events, selection, curated, draws_iter):
    """Curated sets against the null draws, counted by event and never by gene.

    Counting by gene lets one dense cluster carry a result: twenty of the hundred and
    twenty-five imprinted genes sit together on chr7, and a single window over them reports
    a two-fold enrichment that is really one event.
    """
    names = [n for n in sorted(curated) if curated[n]]
    hits = {n: np.array([1.0 if set(e.get("genes", ())) & curated[n] else 0.0
                         for e in events]) for n in names}
    observed = {n: float(h[selection].sum()) for n, h in hits.items()}
    null = {n: [] for n in names}
    for draw in draws_iter:
        for n, h in hits.items():
            null[n].append(h[draw].sum())
    rows = []
    for n in names:
        if hits[n].sum() == 0:
            continue
        column = np.asarray(null[n])
        expected = float(column.mean())
        rows.append({"gene_set": n, "events_hit": int(observed[n]),
                     "pool_expected": round(expected, 2),
                     "fold": round(observed[n] / expected, 3) if expected else "",
                     "p_greater": round(float((int((column >= observed[n]).sum()) + 1)
                                              / (column.size + 1)), 5)})
    return rows


# ------------------------------------------------------------------------- output


def write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--caller", default="OnTAD", choices=sorted(SOURCES))
    parser.add_argument("--events-dir", help="defaults to the caller's own events tree")
    parser.add_argument("--validation", help="a locus_validation.tsv to read instead")
    parser.add_argument("--readout", action="append", dest="readouts",
                        choices=sorted(READOUTS), help="repeatable; default both")
    parser.add_argument("--resolution", action="append", dest="resolutions",
                        choices=RESOLUTIONS, help="repeatable; default every resolution")
    parser.add_argument("--event-type", action="append", dest="event_types",
                        choices=["merge", "split"],
                        help="repeatable; default merge and split together and apart")
    parser.add_argument("--top-k", type=int, action="append", dest="top_ks",
                        help="repeatable; default 0, 60, 100 and 150. 0 is the whole "
                             "pool, i.e. the no-ranking baseline")
    parser.add_argument("--max-gap-bins", type=int, default=1,
                        help="how far apart the two domains may be and still count as one "
                             "boundary change (the folder's convention is 1)")
    parser.add_argument("--side-bins", type=int, default=4,
                        help="crossboundary readout: bins taken on each side of the change")
    parser.add_argument("--flank-bins", type=int, default=0,
                        help="dynamism readout: widen the boundary window by this many bins")
    parser.add_argument("--min-expression", type=float, default=1.0,
                        help="drop genes never reaching this log2(RPKM + 1) at any stage")
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--control-repeats", type=int, default=200)
    parser.add_argument("--draws", type=int, default=2000,
                        help="null draws for the descriptive panel and gene-set folds")
    parser.add_argument("--min-events", type=int, default=60)
    parser.add_argument("--rebuild-splithalf", action="store_true",
                        help="recompute the split-half table instead of using the cache")
    parser.add_argument("--self-test", action="store_true",
                        help="check the statistic and the null on a planted signal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir")
    args = parser.parse_args()

    resolutions = args.resolutions or RESOLUTIONS
    readouts = args.readouts or sorted(READOUTS)
    top_ks = sorted(args.top_ks or [0, 60, 100, 150])
    events_dir = Path(args.events_dir) if args.events_dir else SOURCES[args.caller]["events"]
    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / args.caller
    out_dir.mkdir(parents=True, exist_ok=True)

    genes = load_gene_table()
    index = GeneIndex(genes)
    expression, _ = load_expression()
    modules = derive_transition_modules(expression)
    measured = expressed_genes(expression, genes, args.min_expression)
    dynamism_pct = transition_percentiles(expression, measured)
    level_pct = stage_percentiles(expression, measured)
    rates = module_rates(modules, measured)
    curated = load_curated_sets(genes=genes)
    marks = {name: load_mark(name) for name in CHROMATIN_MARKS}
    structural_table = load_structural(args.caller, args.validation)

    print(f"{args.caller}: {len(measured)} genes above log2(RPKM+1) = "
          f"{args.min_expression} at some stage; anchor {'+'.join(ANCHOR_MARKS)}; "
          f"events from {events_dir}")

    combos = ([tuple(sorted(set(args.event_types)))] if args.event_types
              else [("merge", "split"), ("split",), ("merge",)])

    curve_rows, control_rows, panel_rows, set_rows, summary_rows = [], [], [], [], []
    for readout in readouts:
        print(f"\n=== readout: {readout}")
        for resolution in resolutions:
            for event_types in combos:
                label = "+".join(event_types)
                if readout == "replication":
                    events, replication_background = load_replication(
                        args.caller, {resolution}, set(event_types), args.out_dir,
                        rebuild=args.rebuild_splithalf)
                    for event in events:
                        event["genes"] = [g for g in index.in_window(
                            event["chrom"], event["start"], event["end"])
                            if g in measured]
                else:
                    events = load_events(events_dir, resolution, set(event_types),
                                         args.max_gap_bins, structural_table)
                if len(events) < args.min_events:
                    print(f"  {resolution:7} {label:12} {len(events):4} events "
                          f"-- too few, skipped")
                    continue
                attach_anchor(events, marks)
                matrices = None
                if readout == "replication":
                    pass
                elif readout == "dynamism":
                    matrices = dynamism_matrices(
                        events, index, dynamism_pct, modules, measured, rates,
                        flank_bp=args.flank_bins * RESOLUTION_BP[resolution])
                else:
                    matrices = crossboundary_matrices(
                        events, index, level_pct, dynamism_pct, expression, measured,
                        args.side_bins)
                structural = np.array([e["rank_structural"] for e in events])
                biological = np.array([e["rank_biological"] for e in events])
                # a fixed, tiny, arbitrary offset so rank plateaus break identically on
                # every call and the curve is not an artefact of sort order
                tiebreak = (np.random.default_rng(12345).random(len(events)) - 0.5) * 1e-9
                correlation = float(np.corrcoef(structural, biological)[0, 1])
                usable = len(events) if matrices is None else int(matrices[3].sum())
                print(f"  {resolution:7} {label:12} {len(events):4} events "
                      f"({usable} scorable)  corr(S,B)={correlation:+.3f}")
                if args.self_test and matrices is not None:
                    obs, mean, sd, z = self_test(events, matrices, seed=args.seed)
                    print(f"      self-test: planted {obs:+.4f} vs null {mean:+.4f} "
                          f"+- {sd:.4f}  ->  z = {z:+.1f}")

                for requested in top_ks:
                    # 0 means the whole pool: no ranking at all, which is the baseline the
                    # sweep has to beat and the only honest way to ask whether a readout
                    # carries anything before any selection is applied
                    top_k = len(events) if requested == 0 else requested
                    if requested and top_k > len(events) * 0.9:
                        continue
                    if readout == "replication":
                        observed, selections, global_p, interior_p, n_blocks = \
                            replication_sweep(events, top_k, structural, biological,
                                              tiebreak, permutations=args.permutations,
                                              seed=args.seed)
                    else:
                        observed, selections, global_p, interior_p, n_blocks = sweep(
                            events, top_k, matrices, structural, biological, tiebreak,
                            permutations=args.permutations, seed=args.seed)
                    for entry in observed:
                        curve_rows.append({
                            "readout": readout, "resolution": resolution,
                            "event_type": label, "top_k": top_k, "alpha": entry["alpha"],
                            "n_events": entry["n_events"], "n_genes": entry["n_genes"],
                            "contrast": round(entry["contrast"], 5),
                            "contrast_gene_weighted":
                                round(entry["contrast_gene_weighted"], 5),
                            "secondary_contrast": round(entry["secondary_contrast"], 5),
                            "null_mean": round(entry["null_mean"], 5),
                            "null_sd": round(entry["null_sd"], 5),
                            "z": round(entry["z"], 3) if np.isfinite(entry["z"]) else "",
                            "p_pointwise": round(entry["p_pointwise"], 5),
                            "secondary_p": round(entry["secondary_p"], 5)})

                    values = np.array([e["contrast"] for e in observed])
                    # for the replication readout the direction is predicted, so the peak
                    # is the largest z and not the largest departure in either direction
                    strength = np.array([(e["z"] if readout == "replication"
                                          else abs(e["z"]))
                                         if np.isfinite(e["z"]) else np.nan
                                         for e in observed])
                    best = int(np.nanargmax(strength))
                    best_alpha = observed[best]["alpha"]
                    # a whole-pool row selects every event at every alpha, so its curve
                    # is flat and an "interior peak" there is only a tie broken by noise
                    beats = bool(requested and 0.0 < best_alpha < 1.0
                                 and strength[best] > strength[0]
                                 and strength[best] > strength[-1])
                    summary_rows.append({
                        "readout": readout, "resolution": resolution,
                        "event_type": label, "top_k": top_k, "n_pool": len(events),
                        "best_alpha": best_alpha,
                        "best_contrast": round(float(values[best]), 5),
                        "best_z": round(float(observed[best]["z"]), 3),
                        "contrast_at_0": round(float(values[0]), 5),
                        "z_at_0": round(float(observed[0]["z"]), 3),
                        "contrast_at_1": round(float(values[-1]), 5),
                        "z_at_1": round(float(observed[-1]["z"]), 3),
                        "global_p": round(global_p, 5), "interior_p": round(interior_p, 5),
                        "beats_both_ends": beats, "n_blocks": n_blocks})

                    for which in ("shuffled-S", "shuffled-B"):
                        maker = (replication_control if readout == "replication"
                                 else lambda *a, **k: shuffled_control(
                                     a[0], a[1], matrices, *a[2:], **k))
                        curves = maker(events, top_k, structural, biological, tiebreak,
                                       which, repeats=args.control_repeats,
                                       seed=args.seed + 1)
                        for j, entry in enumerate(observed):
                            column = curves[:, j][np.isfinite(curves[:, j])]
                            control_rows.append({
                                "readout": readout, "resolution": resolution,
                                "event_type": label, "top_k": top_k, "control": which,
                                "alpha": entry["alpha"],
                                "contrast": round(entry["contrast"], 5),
                                "null_p05": round(float(np.percentile(column, 5)), 5)
                                if column.size else "",
                                "null_p95": round(float(np.percentile(column, 95)), 5)
                                if column.size else ""})

                    density = {"gene_density": np.array(
                        [index.count_in_window(e["chrom"], e["start"], e["end"])
                         for e in events], dtype=float)}
                    for alpha in sorted({0.0, best_alpha, 1.0}):
                        selection = selections[ALPHA_GRID.index(alpha)]
                        common = {"readout": readout, "resolution": resolution,
                                  "event_type": label, "top_k": top_k, "alpha": alpha}
                        rng = np.random.default_rng(args.seed + 7)
                        draws = list(selection_null(events, top_k, structural, biological,
                                                    alpha, tiebreak, args.draws, rng))
                        for row in panel_folds(events, selection, marks, density, draws):
                            row.update(common)
                            panel_rows.append(row)
                        for row in gene_set_folds(events, selection, curated, draws):
                            row.update(common)
                            set_rows.append(row)

                    flag = "  <-- interior peak" if beats else ""
                    print(f"      K={top_k:<4} peak alpha={best_alpha:.1f} "
                          f"contrast={values[best]:+.4f} z={observed[best]['z']:+.2f} "
                          f"(alpha 0: z={observed[0]['z']:+.2f}, "
                          f"alpha 1: z={observed[-1]['z']:+.2f})  "
                          f"global p={global_p:.4f} interior p={interior_p:.4f}{flag}")

    write(out_dir / "alpha_curve.tsv", CURVE_FIELDS, curve_rows)
    write(out_dir / "alpha_controls.tsv", CONTROL_FIELDS, control_rows)
    write(out_dir / "alpha_panel.tsv", PANEL_FIELDS, panel_rows)
    write(out_dir / "alpha_gene_sets.tsv", SET_FIELDS, set_rows)
    write(out_dir / "alpha_summary.tsv", SUMMARY_FIELDS, summary_rows)
    (out_dir / "config.json").write_text(json.dumps({
        "caller": args.caller, "events_dir": str(events_dir), "readouts": readouts,
        "resolutions": resolutions, "top_ks": top_ks,
        "max_gap_bins": args.max_gap_bins, "side_bins": args.side_bins,
        "flank_bins": args.flank_bins, "min_expression": args.min_expression,
        "permutations": args.permutations, "control_repeats": args.control_repeats,
        "seed": args.seed, "anchor": list(ANCHOR_MARKS), "alpha_grid": ALPHA_GRID,
    }, indent=2) + "\n")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
