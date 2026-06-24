# scHiCluster on scP/T Cell Differentiation

This folder runs the cloned `zhoujt1994/scHiCluster` workflow on cells under:

`../scptr/cell_differentiation/{1-cell,2-cell,4-cell,8-cell,64-cell}/*.txt`

Current target:

- chromosome: `chr1`
- resolution: `1000kb`
- scHiCluster mode: `pad1_std1_rp0.5_sqrtvc`

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

## Outputs

- `work/cell_matrix/chr1/*_chr1.txt`: sparse raw matrices in scHiCluster format
- `work/imputed_matrix/chr1/*_chr1_pad1_std1_rp0.5_sqrtvc.hdf5`: imputed matrices
- `work/imputed_matrix/merged/embed/*.hdf5`: scHiCluster embedding
- `schiccluster_chr1_1000kb_results.txt`: KMeans clustering metrics against stage labels
- `schiccluster_chr1_1000kb_figure.png`: UMAP plot
