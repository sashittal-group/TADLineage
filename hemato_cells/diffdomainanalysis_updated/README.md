# diffDomain whole-TAD agreement (updated)

This folder is an updated version of `../diffdomainanalysis`. It compares our
merge/split calls against diffDomain, but changes **what is scored**.

## How this differs from `diffdomainanalysis`

The original analysis only looked at the source TADs that *our* pipeline flags
as merge or split, and asked whether diffDomain agrees on those TADs. The
accuracy was therefore measured only over our own events (a precision-style
number): "of the TADs we call merge/split, how many does diffDomain confirm?"

This updated analysis scores **every source TAD** on each tree edge. For each
source TAD it records:

- our label: `split`, `merge`, or `none` (untouched by our transformation), and
- diffDomain's label: `split`, `merge`, or `none`.

It then reports how often the two labels agree over all source TADs. This now
captures the case the original analysis could not: a TAD we call `none` but
diffDomain calls `split`/`merge` (and the reverse). The headline number is a
plain accuracy:

```text
accuracy = agreeing source TADs / all source TADs
```

## What the analysis does

For each variant, chromosome, resolution, leaf pair, and tolerance, and for each
tree edge:

1. Loads the source and target TAD files for the edge.
2. Runs `tad_sequence.transform_sequence` and labels **every** source TAD as
   `split`, `merge`, or `none` based on our detected events.
3. Runs diffDomain's two-sample test on **every** source TAD region using the
   edge's source and target Hi-C matrices (BH adjustment across all of them).
4. Runs diffDomain's classifier over the full source and target TAD lists and
   labels every significant source TAD as `split`, `merge`, or (single / complex
   / loss / not significant →) `none`.
5. Builds a 3x3 confusion matrix (`none`/`split`/`merge`) over all source TADs
   and computes the overall agreement accuracy.
6. Writes JSON summaries and summary figures.

diffDomain is run honestly here: there is **no** event-specific p-value biasing
and **no** target-TAD weighting (the original `--event-source-bias-factor`,
`--classify-all-event-tads`, and `--event-target-weight` knobs are gone). Every
source TAD is tested and classified on equal footing, and a custom
`--diffdomain-alpha` is honoured by remapping adjusted p-values around the
classifier's hard-coded 0.05 cutoff, applied uniformly to all source TADs.

## Tree edges

Same edges as the original analysis. For `inferred`:

```text
MPP-CLP
MPP-CMP
CMP-{leaf1}
CMP-{leaf2}
```

For given-root variants, also `LT-MPP`. Leaf pairs are built from
`leaf1: MEP, MKP, MK` and `leaf2: GMP, G`.

## Main files

- `check_diffdomain_edge_merge_split_events_snakemake`
  Snakemake workflow defining the full batch of comparisons and plots.

- `analyze_diffdomain_merge_split_events.py`
  Runs one whole-TAD comparison for one variant, chromosome, resolution, leaf
  pair, and tolerance. Produces the JSON summary.

- `plot_diffdomain_merge_split_events.py`
  Plots per-edge agreement trees and aggregate agreement summaries.

## Outputs

JSON summaries are written under each result variant (note the `_updated` tag,
so they never collide with the original analysis outputs):

```text
../{variant}/jsons/diffdomain_edge_events_updated_chr{chr}_{res}_{leaf1}_{leaf2}_{tol}.json
```

Each JSON records, per edge, under `tad_agreement`:

- `total_source_tads`, `agree`, `disagree`, `accuracy_percent`,
- `confusion` (3x3 matrix keyed `our_label -> diffdomain_label`),
- `our_counts` and `diffdomain_counts`,
- `we_none_diffdomain_event` (we say none, diffDomain says split/merge),
- `we_event_diffdomain_none` (we say split/merge, diffDomain says none),
- `per_tad` (every source TAD with both labels and whether they agree).

The top-level `aggregate` block holds the same fields summed across all edges.

Figures are written to:

```text
../{variant}/figures/diffdomain_merge_split_updated/
```

Per-edge tree figures label each edge with `agree/total` and the edge accuracy.
Aggregate figures summarize whole-TAD agreement by chromosome or resolution.

## Running

Run the workflow from this folder:

```bash
cd hemato_cells/diffdomainanalysis_updated
conda run -n snakemake-tutorial snakemake -s check_diffdomain_edge_merge_split_events_snakemake
```

Dry-run first to inspect which jobs will run:

```bash
conda run -n snakemake-tutorial snakemake -n -s check_diffdomain_edge_merge_split_events_snakemake
```
