# Higashi scptr

This folder prepares and runs chromosome-specific Higashi jobs for the `scptr/cell_differentiation` dataset.

## Files

- `setup_inputs.py` converts each per-cell contact file into a Higashi v2 input file for one chromosome.
- `run_higashi_scptr.py` writes a run-specific config and runs training/imputation.
- `cluster_embeddings.py` clusters Higashi embeddings, not compartment features.
- `cluster_all_embeddings.py` concatenates cached per-chromosome Higashi cell
  embeddings into a genome-wide feature matrix and clusters that representation.
- `plot_clusters.py` plots the embedding clusters.
- `Snakefile` runs the full Higashi embedding clustering workflow for chr1
  through chr19 plus chrX, then runs an all-chromosome clustering target.
- `config.json` is the base CPU-only configuration; run-specific copies are written under `results/<resolution>/<chromosome>/config.json`.

## Usage

Run the full chr1-chr5, 1000kb workflow with Snakemake:

```bash
cd higashi_scptr
snakemake --cores 1
```

Run the genome-wide clustering step from already trained per-chromosome
embeddings:

```bash
cd higashi_scptr
/Users/PSLabMonisha/miniforge3/envs/higashi/bin/python cluster_all_embeddings.py --resolution 500000
```

The Higashi wiki describes `chrom_list` as the list of chromosomes to train on,
and the main output docs state that `{embedding_name}_0_origin.npy` is the
single-cell embedding while later indices are genomic-bin embeddings. The
previous workflow trained and clustered one chromosome at a time. The new
all-chromosome clustering target is a fast post-training approximation: it
concatenates `scptr_embed_0_origin.npy` from all chromosomes, reduces the
genome-wide embedding with PCA, adds standardized log total coverage as a depth
correction feature, and clusters with nearest-neighbor spectral clustering.

Default PCA dimensions are `20` for 500kb and `10` for 1000kb or coarser. In
the cached runs this improves embedding clustering to:

- `500000/all`: ARI `0.462502`, NMI `0.480853`
- `1000000/all`: ARI `0.599195`, NMI `0.571543`

To inspect the planned jobs without running them:

```bash
cd higashi_scptr
snakemake --dry-run --cores 1
```

Outputs are isolated by run:

- processed data: `higashi_scptr/data/<resolution>/<chromosome>/processed`
- Higashi temp/model/embed files: `higashi_scptr/temp/<resolution>/<chromosome>`
- non-compartment clustering results: `higashi_scptr/results/<resolution>/<chromosome>/cluster_results.txt`
- genome-wide clustering results: `higashi_scptr/results/<resolution>/all/cluster_results.txt`
- cluster figure: `higashi_scptr/results/<resolution>/<chromosome>/cluster_figure.png`

For the provided Snakefile, `<resolution>` is controlled by `RESOLUTION_BP`.

## Alluvial (Sankey) view of the clustering

`plot_cluster_sankey.py` draws the cells flowing from their known developmental stage
into the clusters Higashi returned, which shows *how* a clustering fails rather than only
how much. The drawing lives in `cluster_sankey.py` at the repository root and is shared
with `schiccluster_scptr/` and `scptr_robustad/`.

```bash
python plot_cluster_sankey.py --resolution 1000000
python plot_cluster_sankey.py --all      # every resolution and chromosome
```

- alluvial plot: `higashi_scptr/results/<resolution>/all/cluster_sankey.png`
