# scHiCPTR on the scP/T cell-differentiation cells

This folder reproduces the clustering of **scHiCPTR** (Lyu *et al.*, *Bioinformatics* 38(23):5151–5159,
2022, `../scptr/btac670.pdf`) on the same 508 early mouse embryo cells the rest of the `scptr*`
pipelines use, and scores it with the adjusted Rand index against the five known developmental
stages.

scHiCPTR matters as a baseline for a reason the other two do not share: **it was published on this
dataset**. Section 3.4 of that paper runs the scHiCPTR embedding on the GSE129029
cell-differentiation cells at 250 kb and clusters it with Louvain, reporting an accuracy of 0.91 and
an NMI of 0.81. It reports no ARI, so we recompute all three from the clustering itself.

## What is reproduced

The pipeline of `../scHiCPTR/src/scHiCPTR/embs.py` and `scHiCPTR.py`, step for step:

1. per chromosome, per cell: read the sparse contact matrix, symmetrize, `log2(x + 1)`, convolve
   with a `2*pad+1` box filter (`pad = 3`), random walk with restart (`rp = 0.5`);
2. binarize each cell's flattened imputed matrix at the top `prct = 20` percent;
3. PCA the `cells x bins^2` matrix of one chromosome down to `int(0.2 * n_cells) - 1 = 100` dimensions;
4. concatenate the per-chromosome reductions, PCA again, keep `pca_ncomponent = 120` dimensions;
5. UMAP to `umap_ncomponent = 20` dimensions — this is the scHiCPTR embedding;
6. KNN graph (`n_neighbors = 100`, euclidean) and its UMAP fuzzy connectivities, exactly as
   `Trajectory._compute_graph`, then Louvain at `louvain_res = 0.5`, exactly as
   `Trajectory._detect_communities`.

All parameters are the defaults of `../scHiCPTR/src/scHiCPTR/args.py`, unchanged.

Steps 6 onward in the published pipeline continue into dual graph refinement and DPT pseudotime.
Those affect the *trajectory*, not the clustering, so they are not on the path to an ARI and are not
run here.

Two departures from the published code, both forced and neither touching the method:

- `embs.py` imports `umap` under a `platform.system()` test that admits only Windows and Linux, and
  `scHiCPTR.py` imports `numba.cuda` and calls `louvain.set_rng_seed`, which louvain 0.7 replaced
  with the `seed` argument of `find_partition`. The pipeline is therefore re-expressed here rather
  than imported, with the numerical routines copied verbatim from the source and cited in the
  docstrings.
- `impute_embedding` stacks the pool results into a second copy of a matrix that reaches 1.3 GB at
  250 kb on chr1; this runner streams them into one preallocated array instead. Same matrix.

## Four clusterings

| `--cluster-method` | what it is |
|---|---|
| `louvain` | what the scHiCPTR paper uses (`louvain_res = 0.5` on the KNN connectivities). It picks its own number of clusters, which is three here, not five. |
| `louvain5` | the same Louvain, at the resolution whose partition has exactly **five** communities. Among the five-community partitions on the grid the choice is by silhouette, which reads only the embedding and never the labels, so no more truth enters than the `k = 5` that `kmeans` and `spectral` already get. |
| `kmeans` | k-means with `k = 5` on the embedding, `n_init = 100`. This hands scHiCPTR the true number of stages and is the protocol `schiccluster_scptr/` and `higashi_scptr/` run under. |
| `spectral` | spectral clustering at `k = 5`, the third clustering of the paper's own Fig. 2b. |

All four are run by default and all four are written. `plot_ari_summaries.py` emits a fifth `kind`
named `cluster`, which for each (bin size, chromosome) is whichever scored highest, so the
comparison figure reports scHiCPTR at its best.

`louvain5` exists because `eva.cluster_acc` in the released source **refuses** a clustering whose
community count differs from the number of true classes, so the run behind the published accuracy
of 0.91 must have returned exactly five communities. The shipped `louvain_res = 0.5` returns three.
It is run genome-wide only: the per-chromosome runs already return five communities under `kmeans`
and `spectral`, so the per-chromosome rows of `ari_summary.tsv` come from those three clusterings.

**`louvain5` at 1 Mb genome-wide is the best result scHiCPTR reaches on this data**, at ARI 0.534,
NMI 0.542 and ACC 0.766, against 0.483 / 0.511 / 0.632 for the shipped resolution at that bin size.
It is also the panel the paper draws, because it is the one with five clusters. At 250 kb and
500 kb, forcing five communities *lowers* the ARI (0.406 and 0.443) while raising the accuracy, so
the gain is specific to 1 Mb rather than general.

## Environment

A dedicated conda env, because `louvain` and `umap-learn` do not coexist with the solvers in
`snakemake-tutorial`:

```bash
conda create -n schicptr -c conda-forge python=3.10 numpy=1.26 scipy scikit-learn pandas \
    matplotlib umap-learn python-igraph louvain leidenalg natsort anndata
/Users/PSLabMonisha/miniforge3/envs/schicptr/bin/python -m pip install "setuptools<81"
```

The `setuptools<81` pin is required: `louvain` 0.8.2 imports `pkg_resources`, which setuptools 81
removed.

## Input

The sparse per-chromosome matrices are the ones `../schiccluster_scptr/work/{res}/{chrom}/cell_matrix/`
already holds, so **all three baselines see byte-identical contact maps**. They are binned from
`../scptr/cell_differentiation/{stage}/*.txt` with mm9 chromosome lengths, which are also the
lengths `scHiCPTR/src/scHiCPTR/args.py` ships as its default `chromsize`, so `ngene` lines up
exactly. (The underlying data are mm10; see the note in the repo `README.md`. The mismatch clamps a few
contacts into the last bin of the longer chromosomes and affects every method here identically.)

If those matrices are absent:

```bash
cd ../schiccluster_scptr
/Users/PSLabMonisha/miniforge3/envs/schiccluster/bin/python run_schiccluster_scptr.py --prepare-only
```

## Run

Step 3 is the expensive one and is cached per (bin size, chromosome) in
`work/{res}/{chrom}_reduction.npy`, so the genome-wide run and the 20 single-chromosome runs share
one pass of imputation. Build the cache first:

```bash
PY=/Users/PSLabMonisha/miniforge3/envs/schicptr/bin/python
for res in 1000000 500000 250000; do $PY run_schicptr_scptr.py --prepare-only --res $res --ncpus 8; done
```

Then the clusterings — `--all-chroms` runs `all` followed by each chromosome on its own:

```bash
for res in 1000000 500000 250000; do $PY run_schicptr_scptr.py --all-chroms --res $res; done
```

Single targets:

```bash
$PY run_schicptr_scptr.py --chrom all  --res 250000
$PY run_schicptr_scptr.py --chrom chr1 --res 500000 --cluster-method louvain
```

Long runs (250 kb takes a few hours to impute):

```bash
screen -dmS schicptr caffeinate -is $PY run_schicptr_scptr.py --prepare-only --res 250000 --ncpus 8
```

## Aggregate and plot

```bash
$PY plot_ari_summaries.py     # results/figure/ari_summary.tsv + ari_summary.png
$PY plot_cluster_sankey.py --all
```

Two diagnostic scripts, both described under the reproduction note below:

```bash
$PY prepare_matrices.py --res 500000            # matrices with the diagonal kept
$PY run_schicptr_scptr.py --chrom all --res 500000 --matrix-source diag --seeds 0 1 2
$PY tune_louvain.py --resolution 250kb --chrom all
```

`results/figure/ari_summary.tsv` carries the schema the other baselines write
(`kind / resolution / chromosome / ari`), which is what
`../scptr_robustad/plot_ari_comparison.py` reads to draw the four-method comparison figure of the
paper.

## We do not reproduce the accuracy that the scHiCPTR paper reports

The scHiCPTR paper reports an accuracy of 0.91 and an NMI of 0.81 for these cells at 250 kb.
This folder reaches **0.63 and 0.47** on the same cells, genome-wide at 250 kb, with the released
source at its released defaults. The paper reports no ARI and releases no clustering, so the
accuracy and the NMI are the only two numbers that can be compared.

Seven things were tried and none closed the gap. Only one of them, the five-community Louvain
clustering, raises the score at all, and it lifts the best genome-wide ARI from 0.502 to **0.534**
and the best accuracy from 0.63 to **0.766** — still well short of the published 0.91.

Three act on the input:

| tried | result |
|---|---|
| keeping the main diagonal (`--matrix-source diag`) | ARI 0.429 Louvain / 0.480 k-means at 500 kb, against 0.502 / 0.453 on the shared matrices — no better |
| binning on **mm10** instead of mm9 (`--matrix-source mm10`) | ARI 0.436 / 0.399 / 0.415 at 250 kb, against 0.476 / 0.419 on the shared matrices — **worse** |
| both together (`--matrix-source mm10_diag`) | available; neither component helped alone |

The mm10 point deserves its own note, because it is a genuine defect in the shared matrices that
turns out not to be the culprit. `genome.chrom.sizes` holds mm9 lengths while these reads are
mapped to mm10 — the largest coordinates in the raw files are 195,371,443 on chr1, 182,011,794 on
chr2 and 170,673,392 on chrX, and the last two exceed mm9. The converter clamps anything past the
end into the last bin, so at 250 kb **1.41% of all chrX contacts land in a single bin** (about 9×
its fair share) and 0.27% of chr2's. Correcting it lowers scHiCPTR's ARI rather than raising it.

Three act on the clustering:

| tried | result |
|---|---|
| sweeping the Louvain resolution 0.3–4.0 (`tune_louvain.py`) | ARI stays in 0.21–0.48; best 0.4815 at resolution 0.7, and the best five-cluster run is ARI 0.406 / ACC 0.634 at resolution 1.6 |
| k-means at the true k = 5 | ARI 0.419, ACC 0.624 at 250 kb |
| spectral clustering at the true k = 5 | the third clustering of the paper's own Fig. 2b; ARI 0.478 / ACC 0.721 at 1 Mb |
| Louvain at the resolution giving exactly five communities (`louvain5`) | **the one change that helps**, and only at 1 Mb: ARI 0.534 / **ACC 0.766**, against 0.483 / 0.632 for the shipped resolution. At 250 kb and 500 kb it lowers the ARI to 0.406 and 0.443. |
| varying the UMAP seed (`--seeds 0 1 2`) | ARI moves by at most 0.05 |

Two further checks came back clean: the cell set matches the paper's stage counts exactly
(55 / 94 / 87 / 101 / 171 = 508), and the raw contact files carry no duplicate read pairs, so the
binning is not double-counting.

Note that `eva.cluster_acc` in the released source *refuses* a clustering whose cluster count
differs from the number of true classes, so the run behind the published 0.91 returned exactly
five Louvain clusters. At the shipped `louvain_res = 0.5` this folder gets three, and forcing five
(by resolution or by k-means) still leaves the accuracy near 0.63. The ceiling is therefore the
**embedding**, not the clustering of it. The most likely remaining cause is a difference in how
the contact maps were binned from the reads, which the scHiCPTR paper does not specify beyond
naming `cooler`.

## Outputs

- `work/{res}/{chrom}_reduction.npy` — the cached per-chromosome PCA reduction, `508 x 100`
- `results/{res}/{chrom}/embedding.npy` — the 20-dimensional scHiCPTR embedding
- `results/{res}/{chrom}/{method}/schicptr_results.json` — ARI, NMI, Hungarian-matched ACC,
  silhouette, cluster count
- `results/{res}/{chrom}/{method}/schicptr_clusters.tsv` — per-cell stage and cluster
- `results/{res}/{chrom}/{method}/schicptr_sankey.png` — stage-to-cluster alluvial plot
- `results/{res}/{chrom}/louvain_resolution_sweep.tsv` — `tune_louvain.py` output
- `results/{res}/{chrom}/seed_sweep.tsv` — written when more than one `--seeds` is given
- `results/figure/ari_summary.tsv`, `results/figure/ari_summary.png`

## Measured ARI

Genome-wide (every chromosome pooled) and the mean over the 20 single-chromosome runs, at the
better of the two clusterings:

| bin size | genome-wide | mean single-chromosome |
|---|---|---|
| 250 kb | 0.476 | 0.262 |
| 500 kb | 0.502 | 0.232 |
| 1 Mb | **0.534** | 0.131 |

The 1 Mb genome-wide figure is `louvain5`; the other two are `louvain` at the shipped resolution.

For context, on the same cells scHiCluster reaches 0.702 / 0.583 / 0.576 genome-wide and Higashi
0.256 / 0.463 / 0.599. scHiCPTR is the weakest of the three genome-wide and the strongest of the
three on a single chromosome.
