# Ablation: TAD distance vs imputation in the embryo clustering

**Question (professor, 2026-10-27).** TADLineage clusters the 508 embryo cells far better than
scHiCluster, Higashi and scHiCPTR, even genome-wide (ARI 0.887–0.894 vs at most 0.70). Is that
because of the TAD distance and median, or because of the imputation?

Two experiments were requested:

1. cluster with the TAD distance and the median, **without imputation**;
2. cluster with the TAD distance and **k-means, without the median**.

## Short answer

- **The imputation carries the result, not the TAD distance or the median.** Without imputation,
  clustering by TAD distance alone reaches genome-wide ARI 0.44–0.53 (median) or 0.32–0.43
  (k-means), at or below the baselines (0.58–0.70). On a single chromosome it drops to mean
  ARI 0.05–0.16, below every baseline.
- **The median is not needed.** With imputation, k-means under the same TAD distance matches the
  median everywhere (genome-wide 0.884–0.888 vs 0.884–0.894; single chromosome within 0.03).
- **The imputation's power comes from the cell–cell contact graph it is built on.** That graph
  alone (spectral clustering of the 10-PC contact embedding, no TADs at all) gives genome-wide
  ARI 0.84 and single-chromosome mean 0.51–0.54. The TAD hierarchies of the imputed maps add
  +0.04–0.05 genome-wide and +0.0–0.09 per chromosome on top.
- **But the TAD distance does contain the stage signal, even without imputation.** Started from
  the true stages (an oracle diagnostic, not a method), the EM on *raw* hierarchies stays at ARI
  0.88–0.95, and its objective is as low as or lower than anything the stage-blind starts find.
  The raw TAD objective prefers the stage partition; random starts cannot find it. Imputation
  makes that optimum reachable without labels, at the price of pulling it toward the contact
  graph's partition (oracle on imputed hierarchies: 0.89–0.90).
- **The paper's start set also carries the graph.** Without imputation but with the spectral
  start (which is the contact graph's clustering), the median EM still reaches 0.86–0.89
  genome-wide. So "no imputation" is only a real ablation if the starts are TAD-only as well.

## What was compared

Everything upstream is reused from `../results_unsupervised_median` (the pipeline behind the
paper's Fig. 3); only the step under test changes.

| Factor | Levels |
|---|---|
| Hierarchies | **imputed**: the paper's trees, built from each cell's map pooled over its graph neighbours, W = 0.05 I + 0.95 P^5 (reused unchanged; `build_raw.py --check` rebuilds them bit-identically). **raw**: same tree builder, smoothing and 2–40-bin span limits, but W = I (each cell's own contacts only). |
| Clustering | **median**: the paper's EM, exact median (`fast_median`, = `nussinov.py` cost). **k-means**: Lloyd's k-means on the binary [T \| S \| E] indicator vectors. With tolerance 0, the TAD distance \|TΔT'\| + \|SΔS'\| + \|EΔE'\| *is* the squared Euclidean distance between these vectors, so k-means is exactly "same TAD distance, mean centroid instead of median". |
| Starts | Both methods start from the **same** partitions: the paper's spectral start (10-neighbour spectral clustering of the contact embedding) plus 20 k-means++ starts drawn under the TAD distance. The lowest objective is kept, never the highest ARI. **TAD-only starts** = best of the 20 k-means++ starts (contacts enter only through the hierarchies). **Paper's start set** = best of spectral + k-means++ 1–4 (exactly what the paper ran). |
| Scope | genome-wide (one fit over 20 chromosomes; the cell graph is the 1 Mb one at every bin size, as in the paper) and each of the 20 chromosomes alone (chromosome-only graph and hierarchies), at 250 kb, 500 kb and 1 Mb |

References with no TAD hierarchy at all (`contact_references.py`, genome-wide 1 Mb): spectral
clustering of the contact embedding (= the paper's spectral start), k-means of the same
embedding, and spectral / k-means of the PCA of the imputed contacts.

The paper's numbers are reproduced: the spectral-started median EM gives the objectives of the
paper's fits exactly at 1 Mb, and ARI 0.887 / 0.894 / 0.888 genome-wide and single-chromosome
means 0.565 / 0.565 / 0.487.

## Results

### Genome-wide (ARI vs embryo stage)

| Hierarchies + clustering | starts | 250 kb | 500 kb | 1 Mb |
|---|---|---:|---:|---:|
| imputed + median (paper) | TAD-only | 0.888 | 0.894 | 0.884 |
| imputed + median (paper) | paper's | 0.888 | 0.894 | 0.887 |
| imputed + k-means (exp. 2) | TAD-only | 0.884 | 0.888 | 0.888 |
| imputed + k-means (exp. 2) | paper's | 0.888 | 0.888 | 0.884 |
| no imputation + median (exp. 1) | TAD-only | **0.439** | **0.525** | **0.471** |
| no imputation + median (exp. 1) | paper's | 0.885 | 0.880 | 0.858 |
| no imputation + k-means | TAD-only | 0.340 | 0.321 | 0.431 |
| no imputation + k-means | paper's | 0.879 | 0.879 | 0.872 |
| *contact graph alone, spectral, no TADs* | – | 0.840 | 0.840 | 0.840 |
| *contact embedding, k-means, no TADs* | – | – | – | 0.629 |
| *imputed contacts, PCA, spectral / k-means* | – | – | – | 0.552 / 0.551 |
| best of scHiCluster / Higashi / scHiCPTR | – | 0.702 | 0.584 | 0.599 |

![genome-wide](figures/ablation_genome_wide.png)

Panel (c) shows every TAD-only start. With imputation, several random starts land at 0.85–0.90;
without it, no start of the 20 exceeds 0.60.

**Why the spectral start rescues the raw hierarchies.** Without imputation, the
spectral-started solution has the *lowest* TAD objective of all 21 starts, at every bin size and
for both methods (e.g. 1 Mb median: 2,305,847 vs 2,306,917 for the best k-means++ start), but
by only 0.05–0.3%. The TAD objective
weakly prefers the stage partition, and none of the 20 starts drawn under the TAD distance finds
it. The imputation is what makes that optimum reachable: on imputed hierarchies, random starts
reach it.

### Single chromosome (mean ARI over the 20 chromosomes)

| Hierarchies + clustering | starts | 250 kb | 500 kb | 1 Mb |
|---|---|---:|---:|---:|
| imputed + median (paper) | TAD-only | 0.627 | 0.564 | 0.494 |
| imputed + median (paper) | paper's | 0.565 | 0.565 | 0.487 |
| imputed + k-means | TAD-only | 0.615 | 0.558 | 0.489 |
| imputed + k-means | paper's | 0.560 | 0.538 | 0.496 |
| no imputation + median | TAD-only | **0.155** | **0.100** | **0.048** |
| no imputation + median | paper's | 0.262 | 0.164 | 0.064 |
| no imputation + k-means | TAD-only | 0.201 | 0.140 | 0.096 |
| no imputation + k-means | paper's | 0.520 | 0.432 | 0.239 |
| *contact graph alone (that chromosome's spectral start)* | – | 0.535 | 0.524 | 0.509 |
| scHiCluster / Higashi / scHiCPTR (paper, Fig. 3c) | – | ≤ 0.186 / 0.109 / 0.292 | | |

![per chromosome](figures/ablation_per_chromosome.png)

On one chromosome without imputation, the median EM moves *away* from the contact-graph start
(0.535 → 0.262 at 250 kb): the single-chromosome raw hierarchies are too noisy for the objective
to keep the stage partition.

### Oracle diagnostic: does the TAD objective prefer the stage partition?

`oracle_diagnostic.py` reads the stage labels **on purpose** and is not a method. For each
genome-wide hierarchy set it computes the objective of the true stage partition and runs the EM
from it (`oracle_diagnostic.tsv`).

| bin | hierarchies | method | oracle EM ARI | oracle EM objective | best stage-blind objective | best TAD-only objective |
|---|---|---|---:|---:|---:|---:|
| 250 kb | raw | median | 0.934 | 9,415,667 | 9,416,637 | 9,438,811 |
| 250 kb | raw | k-means | 0.946 | 6,311,956 | 6,312,197 | 6,331,357 |
| 500 kb | raw | median | 0.934 | 4,623,849 | 4,624,113 | 4,626,180 |
| 500 kb | raw | k-means | 0.940 | 3,113,153 | 3,113,323 | 3,122,936 |
| 1 Mb | raw | median | 0.881 | 2,305,862 | 2,305,847 | 2,306,917 |
| 1 Mb | raw | k-means | 0.932 | 1,553,591 | 1,553,751 | 1,557,337 |
| 250 kb | imputed | median | 0.888 | 4,799,045 | 4,799,045 | 4,799,045 |
| 500 kb | imputed | median | 0.894 | 2,270,027 | 2,270,027 | 2,270,027 |
| 1 Mb | imputed | median | 0.903 | 1,125,980 | 1,125,895 | 1,125,900 |

(Imputed k-means rows behave like the imputed median rows; see the TSV.) On raw hierarchies the
stage-near optimum (ARI 0.88–0.95) has a lower objective than every TAD-only solution, and the
lowest of all solutions in five of six cases; the spectral start reaches the same basin. On
imputed hierarchies the oracle EM converges to the solution that the stage-blind starts already
find. The search, not the distance, is what fails without imputation.

## Interpretation for the paper

The comparison with scHiCluster / Higashi / scHiCPTR is not a comparison of TAD hierarchies with
contact embeddings. Most of TADLineage's genome-wide ARI comes from the cell-to-cell contact graph
(1 Mb, sqrt-normalized contacts of all chromosomes, 10 PCs, 10-neighbour graph). The same graph
drives both the imputation and the spectral start. The TAD representation adds a small, consistent
gain on top (+0.04–0.05 genome-wide). The baselines lose mainly on how they compare cells: plain
k-means on our own embedding gives 0.63, about where they are. A fair statement is that
TADLineage's clustering = graph-based imputation of contacts + TAD-hierarchy clustering, and the
first part is necessary. The oracle diagnostic supports the second part: the TAD distance itself
ranks a near-stage partition (ARI up to 0.95) above every partition the stage-blind search finds,
so the TAD representation is informative; imputation is how the method finds that partition
without labels.

Related earlier finding (memory, 2026-10-11): sequencing depth differs by stage, and when all
cells are thinned to equal depth the contact graph alone drops to 0.71 while the imputed median
EM still reaches 0.89. That is the one setting where the TAD step clearly adds more than a few
hundredths.

## Files

| file | what |
|---|---|
| `common.py` | input paths (all read-only from `../results_unsupervised_median`), constants |
| `build_raw.py` | raw (W = I) hierarchies; `--check` rebuilds the paper's imputed trees bit-identically |
| `cluster.py` | median EM and k-means under the TAD distance from identical starts; no labels |
| `contact_references.py` | TAD-free references on the contact embedding; no labels |
| `score.py` | the only fitting-stage script that reads stage labels; writes `summary.tsv`, `per_start.tsv` |
| `oracle_diagnostic.py` | oracle check (reads labels on purpose): EM started from the true stages |
| `plot.py` | `figures/ablation_genome_wide.*`, `figures/ablation_per_chromosome.*` |
| `plot_comparison.py` | one figure per arm in the style of Fig. 3b,c, next to scHiCluster / Higashi / scHiCPTR: `figures/compare_{imputed,raw}_{median,kmeans}.*` (TAD-only starts) and `..._paper_starts.*` |
| `Snakefile` | the whole grid |
| `runs/{bin}/{all,chrN}/{raw,imputed}/` | `result.npz` (labels and objective of every start, both methods), `run.json` |

Run from this folder:

```bash
/Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/snakemake --cores 6 -p --rerun-triggers mtime --scheduler greedy
```

(`--scheduler greedy` because the default ILP scheduler needs `cbc`, which is not installed.)
Runtime: about one hour on 6 cores; the 250 kb raw fits are the slowest.
