# dcHiC Trials

Local helper code for running the core dcHiC R method on hemato cell Hi-C
matrices, summarizing pairwise compartment differences, and building genomewide
MSTs.

## Contents

- `dchic_compartment_distance.py`: converts dense Hi-C matrices into dcHiC
  input files, runs `dcHiC/dchicf.r` (`cis`, `select`, `analyze`), and
  summarizes pairwise compartment differences for one chromosome/resolution.
  The bulky dcHiC pair working directory is deleted after the needed CSVs are
  written unless `--keep-work-dir` is used.
- `check_agg_mst_dchic.py`: aggregates per-chromosome pairwise dcHiC-style
  distances and builds genomewide MSTs.
- `mst_agg_snakemake`: Snakemake workflow for all chromosomes/resolutions.

## dcHiC Distance Workflow

This workflow uses the core dcHiC R script. For each cell pair, chromosome, and
resolution, it performs:

```text
dense Hi-C matrices
  -> dcHiC sparse input + BED
  -> dchicf.r cis
  -> dchicf.r select
  -> dchicf.r analyze
  -> differential.intra_sample_group.pcQnm.bedGraph
  -> pairwise distance summary
```

Then it reports per-bin compartment differences and pairwise distance summaries.

The wrapper keeps one shared dcHiC goldenPath folder per resolution under:

```text
goldenpath/{genome}_{resolution_bp}_goldenpathData/
```

dcHiC creates large genome files such as `mm10.fa` during PC selection. Keeping
them in this shared folder prevents a separate multi-GB copy from being created
inside every cell-pair run. Pair-specific dcHiC folders under `dchic_runs/` are
temporary and are removed after the pair's summary and per-bin CSV files have
been extracted.

## What The Scores Mean

dcHiC is designed around **A/B compartment differences**, not TAD boundary
differences. A compartment score is a per-bin value derived from the Hi-C matrix
PC signal:

- positive values are interpreted as more A-like compartment signal,
- negative values are interpreted as more B-like compartment signal,
- larger absolute values mean stronger compartment tendency.

For two cells, the script computes one compartment score vector for each cell on
the same chromosome/resolution, then compares matching bins:

```text
signed_difference = score_cell2 - score_cell1
abs_difference    = abs(signed_difference)
```

The per-bin output therefore tells us where the two cells differ in compartment
state or compartment strength. A `compartment_switch` is marked when the two
scores have opposite signs, for example CMP is B-like and GMP is A-like in the
same bin.

dcHiC also reports `sample_maha`, a Mahalanobis-style differential compartment
score. Larger `sample_maha` means dcHiC considers that bin more different
between the two cells. The script summarizes this as `mean_dchic_sample_maha`
and `max_dchic_sample_maha`.

The script also writes an `aggregate_bin_score`, which is the per-bin absolute
compartment-score difference rescaled to 0-1 within that pair/chromosome. This
is useful for ranking bins in a pairwise comparison, but it is not a p-value.

For MST construction, we collapse all per-bin differences for a cell pair into
one chromosome-level distance. The main distance summaries are:

- `mean_abs_compartment_diff`: average absolute compartment-score difference
  across bins.
- `rms_compartment_diff`: root-mean-square difference, which gives more weight
  to large bin differences.
- `compartment_switch_count`: number of bins where the compartment sign flips.
- `mean_aggregate_bin_score`: average 0-1 normalized per-bin score.
- `mean_dchic_sample_maha`: average dcHiC Mahalanobis differential score across
  bins.

The Snakemake workflow computes these pairwise distances for every chromosome
and resolution, then `check_agg_mst_dchic.py` aggregates chromosome distances
with either `mean` or `sum` before building the genomewide MST.

Run one direct two-file comparison:

```bash
python dchic_compartment_distance.py \
  --chrom chr19 \
  --resolution 1mb \
  --matrix1 ../CMP/chr19/1mb/hic_matrix.cleaned.txt \
  --matrix2 ../GMP/chr19/1mb/hic_matrix.cleaned.txt \
  --cell1 CMP \
  --cell2 GMP \
  --output dchic_distances/CMP_vs_GMP_chr19_1mb.csv \
  --per-bin-dir dchic_distances/per_bin/chr19_1mb \
  --dchic-script ../../dcHiC/dchicf.r \
  --conda-env dchic-lite \
  --genome mm10
```

Run all pairwise cell comparisons for one chromosome/resolution:

```bash
python dchic_compartment_distance.py \
  --chrom chr19 \
  --resolution 1mb \
  --output dchic_distances/pairwise_dchic_distances_chr19_1mb.csv \
  --per-bin-dir dchic_distances/per_bin/chr19_1mb \
  --dchic-script ../../dcHiC/dchicf.r \
  --conda-env dchic-lite \
  --genome mm10
```

Run the full chromosome/resolution MST workflow from this folder:

```bash
snakemake -s mst_agg_snakemake --cores 1
```

Pairwise outputs:

```text
dchic_distances/pairwise_dchic_distances_{chrom}_{resolution}.csv
dchic_distances/per_bin/{chrom}_{resolution}/bin_scores_{chrom}_{resolution}_{CELL1}_vs_{CELL2}.csv
```

Genomewide MST outputs:

```text
msts/genomewide/{resolution}/{distance_column}/{mean|sum}/
```

Distance columns available for MST construction:

```text
mean_abs_compartment_diff
rms_compartment_diff
compartment_switch_count
mean_aggregate_bin_score
mean_dchic_sample_maha
```

For Snakemake, `mst_agg_snakemake` accepts optional config values:

```bash
snakemake -s mst_agg_snakemake --cores 1 \
  --config dchic_script=../../dcHiC/dchicf.r \
           dchic_conda_env=dchic-lite \
           genome=mm10
```

Use `gfolder=/path/to/{genome}_{resolution_bp}_goldenpathData` only when running
one resolution at a time and you want to force a specific prebuilt dcHiC
goldenPath folder. Otherwise leave it unset so the Python wrapper chooses the
correct shared folder for each resolution.
