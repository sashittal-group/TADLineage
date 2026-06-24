# diffDomain merge/split comparison

This folder compares merge and split events detected by our TAD transformation
workflow against calls made by the original diffDomain implementation.

## What diffDomain does

diffDomain compares Hi-C contact data between two conditions and identifies TADs
whose internal contact pattern changes significantly. Its classifier then
compares those differential TADs with a TAD list from the second condition and
labels the reorganization pattern. The relevant labels for this analysis are:

- `split`: one source-condition TAD corresponds to multiple target-condition TADs.
- `merge`: multiple source-condition TADs correspond to one target-condition TAD.
- `single`, `loss`, and `complex`: other types of differential TAD change.

In this repository, the original diffDomain code is loaded from:

```text
../../diffDomain/diffdomain-py3
```

## What this analysis does

Our own pipeline first detects edge-level TAD edit events on the hematopoietic
tree. For each edge, it compares a source TAD set with a target TAD set using
`tad_sequence.transform_sequence`. That transformation identifies operations
needed to convert one TAD set into the other, including split and merge events.

This folder then asks whether diffDomain assigns the same split or merge label
to the TADs involved in those events.

For each variant, chromosome, resolution, leaf pair, and tolerance value, the
analysis:

1. Loads the source and target TAD files for each tree edge.
2. Detects our split and merge events on that edge.
3. Extracts the source TADs participating in those events.
4. Runs diffDomain's two-sample test on those event TAD regions using the
   corresponding source and target Hi-C matrices.
5. Runs diffDomain's classifier against the target TAD set.
6. Checks whether diffDomain's label matches our event type.
7. Writes JSON summaries and summary figures.

## Tree edges

For the `inferred` topology, the analyzed edges are:

```text
MPP-CLP
MPP-CMP
CMP-{leaf1}
CMP-{leaf2}
```

For given-root variants, the analysis also includes:

```text
LT-MPP
```

The leaf pairs are built from:

```text
leaf1: MEP, MKP, MK
leaf2: GMP, G
```

## Main files

- `check_diffdomain_edge_merge_split_events_snakemake`  
  Snakemake workflow that defines the full batch of comparisons and plots.

- `analyze_diffdomain_merge_split_events.py`  
  Runs one comparison for one variant, chromosome, resolution, leaf pair, and
  tolerance. It produces the JSON summary.

- `plot_diffdomain_merge_split_events.py`  
  Plots per-edge agreement figures and aggregate agreement summaries.

- `snakemake.log`  
  Log from Snakemake runs.

## Outputs

JSON summaries are written under each result variant:

```text
../{variant}/jsons/diffdomain_edge_events_chr{chr}_{res}_{leaf1}_{leaf2}_{tol}.json
```

Each JSON records:

- metadata for the run,
- source and target TAD files,
- source and target Hi-C matrices,
- detected split and merge events,
- diffDomain test results,
- diffDomain classifier output,
- per-edge match counts,
- aggregate match counts.

Figures are written to:

```text
../{variant}/figures/diffdomain_merge_split/
```

The per-edge tree figures show split and merge match counts on each edge. The
aggregate figures summarize diffDomain agreement by chromosome or resolution.

## Running

Run the workflow from this folder:

```bash
cd hemato_cells/diffdomainanalysis
conda run -n snakemake-tutorial snakemake -s check_diffdomain_edge_merge_split_events_snakemake
```

Use a dry-run first to inspect which jobs will run:

```bash
conda run -n snakemake-tutorial snakemake -n -s check_diffdomain_edge_merge_split_events_snakemake
```

