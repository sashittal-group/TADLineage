# Stage-blind contact imputation and median hierarchy clustering

The frozen configuration reaches **ARI 0.883712** and **NMI 0.868411** on **all 508 cells**. No stage labels, stage-specific reference panels, accession numbers, or stage-bearing filenames enter its features, neighbor weights, hierarchy inference, or median fitting. All new scripts and outputs are inside this folder; existing scripts were not edited.

**Scope of the result:** this is an exploratory, stage-blind fit with **k = 5 fixed**. We inspected ARI while developing the method. It is not an independently validated claim that a completely automatic, label-blind parameter selector reaches 0.8. Automatic k selection chose four clusters (ARI 0.708), and the tested automatic graph-parameter selectors did worse. These failures are preserved below.

Open [the benchmark figure](final/benchmark.png), [the exact metrics](final/evaluation.json), or [all hierarchy experiments](experiment_summary.tsv).

## What changed

The previous updated workflow specifies `LABEL_FRACTION = 0.93`, selects cells by stage, and imputes within known stages. Its input RobusTAD calls also used stage-specific reference panels: for example, the existing GSM3690996 metadata records `boundary_mode=lmcc` and a `1-cell` reference panel. Removing just the preimputation labels would therefore leave upstream label information.

This experiment starts from **raw, unimputed contact counts**, cached under `schicptr_scptr/work/250kb/*/cell_matrix_mm10/`. The original binning code reads raw pairs, retains cis contacts on mm10, and drops the 250 kb diagonal; it does not use labels to transform counts. `check_inputs.py` independently reconstructed all 20 chromosomes for two hash-selected cells and found exact agreement with the raw reads. All 10,160 input files have SHA-256 hashes in [the input manifest](data/input_manifest.json).

The new pipeline is:

1. Coarsen counts to **1 Mb**, separately for each chromosome. All 508 cells remain; no stage balancing, stage-based filtering, top-TAD selection, or boundary budget is applied. Cell order is determined by a hash of accession, and accession is never a numerical feature.
2. Concatenate upper-triangular, off-diagonal contacts across chr1–chr19 and chrX. Normalize each cell by its total, take square roots, fit 30 principal components with seed 0, and retain the first **10**. This extracts broad spatial contact patterns, rather than a known stage indicator.
3. Build the same symmetric nearest-neighbor graph used by spectral clustering: `A = (G + G.T)/2`, with `n_neighbors=10`, including the cell itself. Let `P` be its row-normalized transition matrix.
4. Set **`W = 0.05 I + 0.95 P^5`**. This retains 5% direct self contribution plus the self contribution in the diffusion. Pool each chromosome's normalized counts with W. No hard cluster assignments are used for pooling. Apply Gaussian smoothing with sigma 0.7 bins.
5. Infer a hierarchy from the imputed contact map by repeatedly merging the adjacent genomic blocks with the largest mean cross-block contact density. Keep intervals spanning **2–40 Mb**, omit the chromosome root, and never join different chromosomes. These are experimental contact-derived domain hierarchies, not new RobusTAD calls or independently validated biological TAD boundaries.
6. Alternate nearest-median assignment with an **exact interval-RF median** update. Use spectral initialization and 19 random RF kmeans++ initializations; choose the lowest objective, without ARI. Assignments depend only on TAD intervals once this stage starts.

The RF median consists of intervals occurring in strictly more than half of a cluster's cells. It is exact here: majority intervals from laminar trees cannot cross, and each independently minimizes symmetric-difference cost. Ties are omitted. This avoids needing Nussinov DP for this particular metric. It is a change from the repository's relaxed weighted interval-and-boundary metric: **Ct = 1, Cb = 0, tolerance = 0**. Reported objectives sum ordinary interval symmetric differences across chromosomes. Assignment ties retain the current cluster.

## Evidence for a stage-associated signal

Spatial contact-pattern PCA was more useful than the distance-decay profiles examined here. Spectral clustering of the selected contact graph alone gives ARI **0.839678**; the imputed hierarchy median fit improves it to **0.883712**. After fitting, 87.7% of off-diagonal neighbor weight connects cells sharing a stage label. This is an evaluation statistic, not a constraint used to create weights.

| Evaluation | Result |
|---|---:|
| Raw hierarchies, no imputation | ARI 0.341410 |
| Global pooling (20% own, 80% uniform pool) | ARI 0.528802 |
| Permuted neighbor graph | ARI -0.004304 |
| Contact graph spectral clustering alone | ARI 0.839678 |
| Frozen graph-imputation + median configuration | **ARI 0.883712** |

The three hierarchy ablations use the same spectral-start-plus-19-random-start budget as the final fit; the shuffled control uses the correspondingly shuffled spectral start. All reported starts are selected by the RF objective, not ARI. The initial six-start baselines are also preserved under each baseline's `initial_6_restart/` directory; these explain different early numbers in progress logs.

Same-stage hierarchies become substantially closer:

| Mean normalized interval-RF distance | Raw | Imputed |
|---|---:|---:|
| Within stage | 0.7769 | 0.4421 |
| Between stages | 0.7884 | 0.6038 |

Normalization divides symmetric difference by the sum of the two interval counts. Labels are used only to summarize the distances. The resulting separation is consistent with the requested mechanism: cells with similar measured contacts receive similar hierarchies, and median clustering can use that similarity.

## Robustness and limits

The fixed configuration was reproduced through a separate standalone entry point. Its 0.883712 solution has the lowest objective across 20 starts and is also recovered by two random starts. Individual starts span ARI 0.343–0.884, so restart selection matters.

Nearby graph configurations, all with k=5, self retention 0.05 and six starts:

| Neighbors | 3 diffusion steps | 5 steps | 8 steps |
|---|---:|---:|---:|
| 8 | 0.6000 | 0.8198 | 0.8664 |
| 10 | 0.8875 | **0.8837** | 0.5396 |
| 12 | 0.8862 | 0.5396 | 0.5398 |

Five of nine settings exceed 0.8, but over-pooling can destroy the separation. We froze the first successful 10-neighbor/5-step configuration instead of replacing it with the largest observed ARI.

Additional checks:

- **Chromosome-disjoint signal and hierarchy construction:** graph learned from chr1,3,…,19 and hierarchies inferred from chr2,4,…,18,X gives ARI **0.8518**. Reversing the chromosome halves gives **0.6660**. These use the same cells, so they are not independent-cell validation.
- **Automatic k, frozen graph:** RF silhouette over k=2…9 chooses k=4, ARI **0.7084**. See `validation/`. Thus the 0.884 result retains a known-five-cluster assumption.
- **Automatic graph configuration:** maximum hierarchy silhouette across the exploratory candidates selects ARI **0.5398**. Three 50/50 binomial contact splits, refitting PCA and the graph on training counts and selecting by held-out cross entropy, choose 8 neighbors/3 steps: ARI **0.6000**. See `reconstruction/`.
- **Development search:** 140 initial feature/clustering configurations, 552 refined feature configurations, and 27 hierarchy configurations including controls are recorded. These are exploratory comparisons on the same cells, not independent replicates. No held-out-cell or external-dataset ARI claim is made.
- **Old result comparison:** the inspected `results_updated/scptr_runs_all_cells/250kb/all/tol_0.json` contains 35 evaluated cells and ARI 0.9029. The new run has 508 cells and uses a different resolution, domain caller, and metric, so these values are not a controlled head-to-head comparison.

The practical result is a reproducible stage-blind fitting pipeline above the requested ARI target. Fully automatic selection that consistently achieves that target remains unresolved.

## Reproduce the frozen result

From the repository root:

```bash
bash scptr_robustad/results_unsupervised_median/run.sh
```

This uses the existing `hic_analysis` Python environment, keeps caches here, runs the fit first, then evaluates it separately. Override the interpreter with `EXPERIMENT_PYTHON=/path/to/python`. Versions are recorded in `requirements.txt`; no installation or changes to existing environments were performed.

The fitting entry point does not import evaluation code or read stage metadata. For an automatic-k run:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export NUMBA_CACHE_DIR="$PWD/scptr_robustad/results_unsupervised_median/cache"
/Users/PSLabMonisha/miniforge3/envs/hic_analysis/bin/python -B \
  scptr_robustad/results_unsupervised_median/run_pipeline.py \
  --k auto --output final_auto
/Users/PSLabMonisha/miniforge3/envs/hic_analysis/bin/python -B \
  scptr_robustad/results_unsupervised_median/evaluate_final.py --output final_auto
```

`extract.py` rebuilds the cached features; `run_pipeline.py` invokes it if the feature cache is absent. `raw_profiles.py`, `signals.py`, and `refine_signals.py` reproduce the exploratory feature screens; the raw fine-distance screen is not needed for the frozen result. `validate.py`, `controls.py`, `select_by_reconstruction.py`, and `fair_baselines.py` reproduce the corresponding validation experiments. `report.py` regenerates the combined figures once those experiment outputs exist.

## Outputs

- `final/run.json`: parameters, objective trace, all restart objectives and cluster sizes.
- `final/assignments.tsv`: cell IDs and inferred clusters, without stage labels.
- `final/metrics.json` and `final/evaluation.json`: post-fit stage agreement and diagnostics.
- `final/imputed_hierarchies.tsv.gz`: every cell's inferred hierarchy, as chromosome/start/end in **0-based, half-open base-pair coordinates**, clipped to mm10 chromosome lengths.
- `final/median_hierarchies.tsv`: median hierarchy for each cluster, in the same coordinate convention.
- `final/fit.npz`, `final/trees.npz`, `final/intervals.json`: numerical fit, sparse cell-by-interval matrix, and its column mapping. Internal intervals are half-open 1 Mb bin indices.
- `final/benchmark.png`, `final/benchmark.pdf`, `final/embedding.png`: actual-cell plots; no synthetic or unselected display points.
- `experiment_summary.tsv`, `summary.json`: complete hierarchy comparisons and concise numerical results.

Five unit tests check laminarity, exact median optimality against exhaustive small examples, objective monotonicity and nearest-median assignments, and order-invariant stochastic imputation weights. `check_inputs.py` checks the raw-count provenance. Existing scripts outside this folder were left unchanged.

## Per-chromosome ARI

The original 0.883712 result combines all chromosomes. Separate per-chromosome fits are now saved in [the per-chromosome table](per_chromosome/ari_by_chromosome.tsv) and [plot](per_chromosome/ari_by_chromosome.png). Each fit uses all 508 cells, k=5, 1 Mb hierarchies, and the same 20-start objective-based selection.

Two definitions are reported to distinguish the sources of information:

- **Chromosome-only:** PCA, neighbor graph, imputation, spectral initialization, and median clustering all use just the stated chromosome. This is the independent per-chromosome result. ARIs range from 0.2025 (chr19) to 0.6178 (chr5).
- **Shared graph:** restrict the saved genome-wide-imputed hierarchies to one chromosome, then refit its median clusters. Imputation and spectral initialization still use the genome-wide contact representation. These ARIs range from 0.4433 (chr16) to 0.8794 (chr13), and must not be described as independent chromosome-only evidence.

Neither mode reads stage labels during fitting; evaluation is a separate script. No chromosome-specific parameter tuning was performed. These results show that the high combined ARI relies substantially on information shared across chromosomes.

To reproduce, run `per_chromosome.py` followed by `evaluate_per_chromosome.py` in the same Python environment and with the cache/thread environment variables described above. The fit script resumes completed chromosomes. Predictions, medians, interval mappings, and all restart objectives are preserved under `per_chromosome/{chromosome_only,shared_graph}/{chrom}/`.
