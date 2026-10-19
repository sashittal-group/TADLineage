# scHiCluster on scP/T Cell Differentiation

This folder runs the cloned `zhoujt1994/scHiCluster` workflow on cells under:

`../scptr/cell_differentiation/{1-cell,2-cell,4-cell,8-cell,64-cell}/*.txt`

Current target:

- chromosomes: `chr1`-`chr19`, `chrX`
- resolution: `250kb` in the Snakemake workflow
- scHiCluster mode: `pad1_std1_rp0.5_sqrtvc`
- clustering: all-chromosome scHiCluster embedding, then KMeans on 20 SVD dimensions plus standardized log contact depth

## Environment

Use the dedicated conda environment:

```bash
conda activate schiccluster
```

The workflow script is written to call:

```bash
/Users/PSLabMonisha/miniforge3/envs/schiccluster/bin/python
```

The local `scHiCluster` checkout is used through `PYTHONPATH`, so `hicluster` commands are run as:

```bash
python -m schicluster ...
```

## Run

Prepare sparse scHiCluster matrices only:

```bash
/Users/PSLabMonisha/miniforge3/envs/schiccluster/bin/python run_schiccluster_scptr.py --prepare-only
```

Run the full chr1 1000kb workflow:

```bash
/Users/PSLabMonisha/miniforge3/envs/schiccluster/bin/python run_schiccluster_scptr.py
```

Run one 250kb workflow target:

```bash
/Users/PSLabMonisha/miniforge3/envs/schiccluster/bin/python run_schiccluster_scptr.py --chrom chr1 --res 250000
```

Run the all-chromosome merged clustering target after the per-chromosome targets
exist:

```bash
/Users/PSLabMonisha/miniforge3/envs/schiccluster/bin/python run_schiccluster_scptr.py --chrom all --res 250000
```

Rescore an existing cached embedding without rerunning contact conversion or
imputation:

```bash
/Users/PSLabMonisha/miniforge3/envs/schiccluster/bin/python run_schiccluster_scptr.py --chrom chr1 --res 250000 --analysis-only
```

Smoke test on the first few cells:

```bash
/Users/PSLabMonisha/miniforge3/envs/schiccluster/bin/python run_schiccluster_scptr.py --limit 5
```

## scHiCluster Commands Used

For each cell:

```bash
python -m schicluster impute-cell \
  --indir work/cell_matrix/chr1/ \
  --outdir work/imputed_matrix/chr1/ \
  --cell CELL_ID \
  --chrom chr1 \
  --res 1000000 \
  --pad 1 \
  --std 1 \
  --rp 0.5 \
  --mode pad1_std1_rp0.5_sqrtvc \
  --chrom_file work/chr1.chrom.sizes
```

Then concatenate chr1 cells:

```bash
python -m schicluster embed-concatcell-chr \
  --cell_list work/imputed_matrix/filelist/imputelist_pad1_std1_rp0.5_sqrtvc_chr1.txt \
  --outprefix work/imputed_matrix/merged/embed/pad1_std1_rp0.5_sqrtvc_chr1 \
  --res 1000000 \
  --dist 10000000 \
  --dim 50
```

Then merge the chromosome embedding:

```bash
python -m schicluster embed-mergechr \
  --embed_list work/imputed_matrix/filelist/embedlist_pad1_std1_rp0.5_sqrtvc.txt \
  --outprefix work/imputed_matrix/merged/embed/pad1_std1_rp0.5_sqrtvc \
  --dim 20
```

For this request the embed list contains only `chr1`.

The downstream clustering step now defaults to:

```bash
--cluster-dims 20 --cluster-feature-mode embedding-depth --kmeans-n-init 100
```

This replaces two earlier problems:

- the per-chromosome workflow only merged one chromosome at a time, while the
  scHiCluster documentation expects chromosome embeddings to be concatenated and
  merged into a genome-wide embedding;
- the earlier KS-selected dimension rule frequently selected only one SVD
  dimension and produced near-zero ARI.

The result file also reports `depth_only_ARI` and `depth_only_NMI` as
diagnostics because library/contact depth is stage-associated in this dataset.
The scHiCluster tutorial also L2-normalizes the PCA/SVD representation before
neighbor-graph clustering; that behavior can be tested with:

```bash
python run_schiccluster_scptr.py --chrom all --cluster-feature-mode embedding-l2
```

To reproduce the old embedding-only behavior, run:

```bash
python run_schiccluster_scptr.py --cluster-feature-mode embedding --cluster-dims 0
```

## Outputs

- `work/{resolution}/{chrom}/cell_matrix/*_{chrom}.txt`: sparse raw matrices in scHiCluster format
- `work/{resolution}/{chrom}/imputed_matrix/*_{chrom}_pad1_std1_rp0.5_sqrtvc.hdf5`: imputed matrices
- `work/{resolution}/{chrom}/merged/embed/*.hdf5`: per-chromosome scHiCluster embedding
- `work/{resolution}/all/merged/embed/*.hdf5`: all-chromosome merged scHiCluster embedding
- `results/{resolution}/{chrom}/schiccluster_results.txt`: KMeans clustering metrics against stage labels
- `results/{resolution}/{chrom}/schiccluster_figure.png`: UMAP plot

## Alluvial (Sankey) view of the clustering

`plot_cluster_sankey.py` draws the cells flowing from their known developmental stage
into the clusters scHiCluster returned, which shows *how* a clustering fails rather than
only how much. The drawing lives in `cluster_sankey.py` at the repository root and is
shared with `higashi_scptr/` and `scptr_robustad/`.

```bash
python plot_cluster_sankey.py --resolution 250kb
python plot_cluster_sankey.py --all      # every resolution and chromosome
```

- `results/{resolution}/{chrom}/schiccluster_sankey.png`: stage-to-cluster alluvial plot
