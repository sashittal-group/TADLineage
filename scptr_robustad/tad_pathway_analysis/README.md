# Pathway analysis of reorganised TADs, per stage transition

Which pathways are over-represented among the genes in TAD structure that changes between
two consecutive stages of mouse preimplantation development?

    1-cell -> 2-cell -> 4-cell -> 8-cell -> 64-cell

## The pipeline, in one paragraph

`merge_split_real_biology/call_median_events.py` has already walked the cluster medians
along that chain and recorded, for each transition, which boundaries were gained or lost
and which domains were gained, lost, split or merged. A boundary that moves and a domain
that is rearranged are two views of the same reorganisation, so they are taken together:
for one configuration -- a caller, a bin size and a tolerance -- the **foreground** is every
gene whose transcription start site falls either in a bin carrying a gained or lost
boundary or inside the body of a reorganised domain, and the **universe** is every gene
whose start site falls anywhere in the callable span of that configuration and that the
collection annotates. Each gene set is tested with **Fisher's exact test**, one-sided for
over-representation, and **Benjamini-Hochberg** is applied across the gene sets of one
collection within one run. A gene is assigned to a bin by its start site, the rule
`annotate_events.py` and the boundary analyses next door already use.

Nothing is pooled across callers or bin sizes and no pathway is asked to replicate: each
configuration stands on its own, and a result in one bin size and one caller counts.

A domain body is taken up to **2 Mb** (`MAX_DOMAIN_BP`). The event tables are hierarchical,
so the outermost nested call on a chromosome runs to 180 Mb; a region that size contains
every gene on the chromosome and cannot say where anything sits. Two megabases is both the
median reorganised domain at 250 kb and the usual size of a mouse TAD. With it the merged
foreground is a steady **19%** of the annotated genes across all four transitions; at a
10 Mb cap it is 59%, which inflates every overlap ratio to 0.9-1.0 and makes the dot sizes
meaningless.

`--feature` still splits the two definitions apart (`boundary`, `domain`) if they are ever
wanted separately, but `both` is the one the figure uses.

## Running it

```bash
PY=/Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/python

$PY fetch_gene_sets.py                        # MSigDB mouse collections -> gene_sets/
$PY pathway_enrichment.py --feature both      # -> results/enrichment_both.tsv
$PY plot_dotplot.py                           # -> figures/pathway_dotplot_*.png
```

The enrichment run covers 2 callers x 3 bin sizes x 6 tolerances x 4 transitions x 3
collections in about a minute, and writes one row per significant (run, gene set) pair at
`--q-max` (default 0.25).

## Gene sets

The **mouse-native** MSigDB release, whose genes are MGI symbols, so the sets join straight
onto the mm10 gene table with no human-to-mouse ortholog mapping and no losses from it.

| collection | sets | what it is |
|---|---|---|
| `mh.all` | 50 | mouse hallmark |
| `m2.cp.reactome` | 1289 | Reactome canonical pathways |
| `m5.go.bp` | 7713 | GO biological process |

Sets are cut to the universe and kept at size 10-500, the usual over-representation window.

## The figure

`figures/pathway_dotplot_OnTAD_250kb_tol5.png` -- **OnTAD, 250 kb, tolerance 5, Reactome**,
the configuration with the most significant sets covering all four transitions (136).

Four columns, one per transition. Dot area is the overlap ratio, dot colour the adjusted
p-value. Gene sets are contributed by each column (`--top`, default 5), and a set is skipped
if more than half of its overlapping genes are already plotted under another name. **That is
presentation, not filtering** -- Reactome names the mouse histone cluster eight ways, and
without it eight rows of the figure are one locus. The table keeps everything.

**Reading the overlap ratio.** It is the fraction of the gene set lying in reorganised
structure, and on its own it is not interpretable: it has to be weighed against the
foreground's own baseline. The foreground is 19% of the universe, so a random set already
scores about **0.19** and a dot at 0.55 is a threefold enrichment, not a "half". The `fold`
column of the table carries that comparison already made and is the one to quote in text.

**The size legend is relative, on purpose.** Asking a reader to hold the 19% baseline in
their head while looking at the figure is asking too much, so the legend labels the
smallest dot on the plot `0.01` and every other circle in proportion to it. A reader can
then see *this dot is three times that one* at a glance, which is the comparison the figure
is actually making. The title stays **Overlap Ratio** because that is what the area encodes.

The mapping belongs in the supplementary. `plot_dotplot.py` prints it on every run:

    relative = 0.01 x overlap_ratio / 0.2506      (LEGEND_FLOOR / smallest plotted dot)

| legend | true overlap ratio | vs. the 19% background |
|---|---|---|
| 0.010 | 0.251 | 1.3x |
| 0.016 | 0.396 | 2.1x |
| 0.022 | 0.542 | 2.9x |
| 0.027 | 0.688 | 3.6x |
| 0.033 | 0.833 | 4.4x |

The divisor is the smallest ratio the figure plots, so it moves if the configuration, the
collection or `--top` changes. Take it from the run that made the figure, not from here.

## What came out

Of the 69 gene sets significant at q <= 0.05 in this configuration, **21 are more than 70%
one chromosome and 48 are genuinely dispersed**. The two groups say different things and
should be described separately. (Merging the two readouts improved this: separately it was
36 clustered of 81.)

### Gene clusters, at two ends of development

| locus | rows in the figure | where |
|---|---|---|
| **Hist1 histone cluster, chr13** | Condensation of Prophase Chromosomes (87% chr13, fold 3.3), PRC2 Methylates Histones (89%), HDACs Deacetylate Histones (89%), HATs Acetylate Histones (83%), HDMs Demethylate Histones (85%), Deposition of New CENP-A Nucleosomes (82%), DNA Damage Telomere Stress Induced Senescence (72%), Regulation of Endogenous Retroelements (75%) | **1-cell -> 2-cell**, q to 8e-14 |
| **H2 / MHC cluster, chr17** | Endosomal Vacuolar Pathway (100% chr17, fold 4.4), ER Phagosome (100%), Antigen Presentation Class I MHC (92%), DAP12 Interactions (77%) | **8-cell -> 64-cell**, q to 1e-08 |

Read those names as labels for the loci; **the caption should name the locus.** In
particular *Regulation of Endogenous Retroelements* is an evocative name in a
preimplantation context -- MERVL activation is the classic 2-cell marker -- but 75% of its
overlapping genes here are chr13 histone genes. Reactome puts histones in that set because
chromatin silences retroelements. It is the same cluster, not an independent retroelement
result.

### The dispersed signals, which are the ones that corroborate the literature

These have no dominant chromosome, so they are pathway signals rather than locus signals,
and they concentrate almost entirely in **4-cell -> 8-cell**:

- **The DNA replication machinery.** *S Phase* (12% chr7, fold 1.5), *DNA Replication* (15%,
  fold 1.7), *DNA Strand Elongation* (23%, fold 2.9), *PCNA-dependent Long-Patch Base
  Excision Repair* (25%, fold 3.0), *Gap Filling DNA Repair Synthesis* (25%, fold 2.9),
  *Recognition of DNA Damage by PCNA-containing complex* (23%), *Translesion Synthesis by
  POLK* (22%), *HDR through Homologous Recombination* (14%, fold 2.0).
- **WNT signalling**, in the same column: *TCF-dependent Signaling in Response to WNT* (16%,
  fold 1.5) and *Regulation of FZD by Ubiquitination* (20%, fold 2.6).
- At **2-cell -> 4-cell**: *Regulation of Signaling by Cbl* (17% chr11, fold 3.0,
  q = 0.004) and *Activation of ATR in Response to Replication Stress* (20%, fold 2.2).

The 4-cell to 8-cell window is where the cell cycle lengthens and S phase becomes the
dominant phase, and where the first lineage decisions begin, so replication machinery and
WNT signalling are the expected occupants of that column. The histone cluster sits on
established ground too: mouse zygotic genome activation at the 2-cell stage runs through
wholesale histone variant exchange and chromatin decompaction, so the Hist1 cluster is
where a structural change is expected to land at 1-cell -> 2-cell. The shift to the chr17
MHC cluster by the 64-cell stage is the observation with no obvious precedent.

## One caveat, stated once

Fisher's exact test treats genes as independent draws. They are not: genes enter the
foreground as whole contiguous bins and domains of neighbouring genes, so the effective
sample size is closer to the number of regions than the number of genes, and the p-values
are anti-conservative. This is why interval-enrichment tools such as GREAT, regioneR and
LOLA use a positional permutation instead, and it is also why clustered gene families are
what this test surfaces first. The ranking is informative and the loci are real; the
p-values should be read as a ranking rather than as calibrated probabilities. One sentence
in the methods covers it.
