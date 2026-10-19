# scPATR RobustTAD Clustering

This folder clusters single-cell TAD structures using RobustTAD calls as the
input TADs. The main idea is:

```text
RobustTAD TADs per cell
  -> convert base-pair TADs to chromosome/bin TAD files
  -> harmonize nearby boundaries
  -> impute/clean each stage with weighted Nussinov DP
  -> EM clustering over TAD-set distances
  -> compare clusters to known cell stages
  -> build an MST between cluster medians
```

The clustering is TAD based. It is not clustering raw Hi-C matrices directly.
Each cell is represented by the set of TAD intervals detected for one
chromosome and resolution.

## Input Layout

The original RobustTAD outputs are expected under:

```text
robustad_scptr_data/{resolution_kb}/{stage}/{GSM}/
  robustad_tads_all.bed
  robustad_boundaries.bed
```

The converter writes an OnTAD/scPATR-like layout under:

```text
cell_differentiation_robustad_ontad/{stage}/{GSM}/{chrom}/{resolution}/
  tads.tad
  boundaries.bed
```

Here, `stage` is one of:

```text
1-cell, 2-cell, 4-cell, 8-cell, 64-cell
```

The converter changes base-pair coordinates into 1-based bin coordinates. For
example, at `500kb`, a genomic coordinate is mapped to its 500 kb bin.

Run conversion with:

```bash
python convert_scptr_robustad_to_ontad_chr_layout.py
```

## How Clustering Works

The main clustering script is:

```text
em_cluster_scptr_robustad_weighted.py
```

For each chromosome, resolution, and tolerance value, it does the following.

### 1. Collect TAD Files

The script finds all `tads.tad` files for one chromosome and resolution across
the five developmental stages.

Example:

```text
cell_differentiation_robustad_ontad/4-cell/GSM.../chr1/1000kb/tads.tad
```

### 2. Select Cells Under a Boundary Budget

The script does not always use every available cell. It limits the number of
unique TAD boundaries so the dynamic programming step stays tractable.

Current defaults in the Snakefile:

```text
K = 5
TOP_TADS_PER_CELL = 20
MAX_UNIQUE_BOUNDARIES = 100
TOLERANCES = 0, 1, 2, 3, 4, 5
RESOLUTIONS = 1000kb
```

For `250kb`, the Snakefile can force a fixed number of files per stage with
`PER_STAGE_BY_RESOLUTION`. For `500kb`, selection is done stage by stage so
every stage gets a share of the boundary budget. Other per-chromosome
resolutions use the global boundary budget unless `--per-stage` is passed.

### 3. Harmonize Nearby Boundaries

TAD boundaries that are close to each other are treated as equivalent. The
`--tolerance` value controls how close two boundaries can be and still be
merged.

Small tolerance means stricter matching. Larger tolerance means nearby boundary
positions are allowed to count as the same boundary.

The harmonized files are written as:

```text
tads_filtered.tad
```

### 4. Weighted Imputation Within Each Stage

After filtering, the script runs weighted Nussinov-style dynamic programming
separately for each stage.

This step uses:

- TAD interval scores from `tads.tad`
- left/right boundary scores from `boundaries.bed`
- `Ct`, the TAD interval mismatch weight
- `Cb`, the boundary mismatch weight

The purpose is to produce cleaned/imputed child TAD sets for the selected cells.
These files are written as:

```text
tads_filtered_imputed.tad
```

For `250kb` and `1000kb`, the script keeps only the top-scoring TADs per cell
before imputation. The current default is the top `20` TADs.

### 5. EM Clustering

Each imputed cell is represented as a set of TAD intervals:

```text
{[start1,end1], [start2,end2], ...}
```

The EM clustering alternates between two steps:

1. Assign each cell to the closest cluster median.
2. Recompute each cluster median using Nussinov dynamic programming.

Distance between two TAD sets is:

```text
Ct * relaxed_TAD_RF + Cb * relaxed_boundary_RF
```

In easy words, two cells are close if they share similar TAD intervals and
similar TAD boundaries, allowing small boundary shifts controlled by
`--tolerance`.

The current workflow uses `k=5`, matching the five known stages.

### 6. Evaluate Against Known Stages

The known stage labels are used only for evaluation and plotting, not for the
EM assignment step.

The JSON output reports:

- `ARI`: adjusted Rand index
- `NMI`: normalized mutual information
- `homogeneity`
- `completeness`
- `v_measure`
- `silhouette`
- `true_labels`
- `predicted_labels`
- `cluster_majorities`

High ARI/NMI means the inferred clusters agree well with the known
developmental stages.

### 7. Build A Median MST

After clustering, each cluster has a median TAD set. The script computes
distances between cluster medians and builds a minimum spanning tree.

Each cluster is labeled by the majority true stage of the cells assigned to it.
The inferred tree is compared to the expected developmental chain:

```text
1-cell -> 2-cell -> 4-cell -> 8-cell -> 64-cell
```

The tree comparison is saved as `tree_distance`.

## Running The Workflow

The Snakemake workflow runs all configured chromosomes, tolerances, and
resolutions:

```bash
snakemake -s Snakefile --cores 1
```

The current Snakefile is configured for:

```text
chromosomes: chr1-chr19, chrX
resolution: 1000kb
tolerances: 0-5
```

To run one case manually:

```bash
python em_cluster_scptr_robustad_weighted.py \
  --chromosome chr1 \
  --resolution 1000kb \
  --tolerance 1 \
  --top-tads-per-cell 20 \
  --max-unique-boundaries 100 \
  --k 5 \
  --json-out results/scptr_runs_all_cells/1000kb/chr1/tol_1.json \
  --output results/scptr_runs_all_cells/1000kb/chr1/tol_1.png
```

## Outputs

Per-run outputs:

```text
results/scptr_runs_all_cells/{resolution}/{chrom}/tol_{tolerance}.json
results/scptr_runs_all_cells/{resolution}/{chrom}/tol_{tolerance}.png
```

The PNG has two panels:

- true clustering by known stage
- EM clustering by inferred cluster, with cluster medians and MST edges

The JSON stores the clustering scores, selected files, imputed files, cluster
assignments, MDS coordinates, median coordinates, and the median MST.

Aggregate all JSON files into one CSV:

```bash
python aggregate_tree_distance_json.py \
  --input-glob 'results/scptr_runs_all_cells/**/*.json' \
  --output-csv results/scptr_tree_distance/all_tree_distances.csv
```

Build a merged MST for one resolution:

```bash
python merge_resolution_trees_mst.py \
  --csv results/scptr_tree_distance/all_tree_distances.csv \
  --resolution 1000kb
```

Plot ARI summaries:

```bash
python plot_ari_summaries.py
```

This writes summary figures under:

```text
results/figure/
```

## Short Interpretation

In plain words, this folder asks:

> If each cell is described by its RobustTAD intervals, can we group cells into
> developmental stages based on how similar their TAD structures are?

The answer is measured with ARI/NMI against the known stage labels, and with the
MST/tree-distance score against the expected developmental order.

## Alluvial (Sankey) view of the clustering

`plot_cluster_sankey.py` draws the cells flowing from their known developmental stage
into the clusters the median clustering returned, which shows *how* a clustering fails
rather than only how much. The drawing lives in `cluster_sankey.py` at the repository
root and is shared with `schiccluster_scptr/` and `higashi_scptr/`.

```bash
python plot_cluster_sankey.py --resolution 500kb --tolerance 1
python plot_cluster_sankey.py --all      # every resolution, chromosome and tolerance
```

A run clusters a subsample of the cells of each stage. By default the panel scales each
stage up to `available_counts_by_stage`, replicating the clustered cells in the
proportions the run produced, so the panel shows the dataset at full size; this is the
convention `plot_median_chain.py` already follows for the MDS panel. The scaling is exact
within a stage but changes the mix of stages, so the ARI, the NMI and the matched-diagonal
fraction printed on the panel are always computed on the cells that were actually
clustered, never on the scaled drawing. Pass `--no-scale` to draw only the clustered
cells, and `--show-label-note` to print how many cells were clustered and how many had
their stage label used by the preimputation step.

- `results_updated/scptr_runs_all_cells/{resolution}/{chrom}/tol_{t}_sankey.png`

## Merge and split events, and the biology on them

`merge_split_real_biology/` takes the cluster medians produced here and asks what changes
between the stages: which domains split, which merged, and whether the genes at those
boundaries do anything at the same transition. It runs the same
`tad_sequence.transform_sequence` that `hemato_cells/analyze_edge_merge_split_events.py`
runs along the hematopoietic tree, so the event definitions match between the two, and it
annotates the results against mm10 genes, the Deng et al. 2014 expression atlas, ENCODE
CTCF peaks and curated sets for imprinting, X-inactivation escape, the Hox clusters and
the pluripotency network.

Read `merge_split_real_biology/results/REPORT.md` before using any of it. The events are
mostly not reproducible across bin sizes, the boundaries they define are depleted of genes
and CTCF rather than enriched, and no transition-specific transcriptional signal survives
the multiplicity correction. The pipeline and the negative controls are the deliverable;
the locus list is a shortlist of candidates.

The obvious objection -- that sparse single-cell contacts jitter the boundary positions and
a one-bin window therefore misses the CTCF site that is really there -- is tested in
section 8 of that report. It does not hold: the CTCF density profile has its minimum at the
call itself, so the calls are positionally precise, and widening the window only dilutes
the fold toward 1. Positional tolerance does help elsewhere, nearly tripling the loci
recovered at more than one bin size, and `results/CANDIDATES.md` is the short list built
that way, with a literature note on each locus worth a second look.

```bash
cd merge_split_real_biology
snakemake --cores 4
```
