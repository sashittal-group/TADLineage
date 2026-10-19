# Pathway analysis of TAD reorganisation along the haematopoietic tree

Every node of the tree carries its own TAD calls, from two independent callers. For each
edge we take the TADs that are **not shared** between parent and child, take the genes inside
them, and test those genes for pathway over-representation with Fisher's exact test. Nothing
is inferred and nothing is trimmed: this is the observed hierarchy end to end.

```
OnTAD     LT -> ST -> MPP -> CMP -> MEP -> MKP -> MK      9 edges, 10 nodes
                          \       \
                           CLP     GMP -> G

RobusTAD  LT -> MPP -> ...                                8 edges,  9 nodes
```

RobusTAD was never run on ST, so its stem is one edge shorter. At 250 kb the two callers
agree on about 70% of their calls, which makes them a genuine cross-check.

## Method

**Reorganised** = called at one endpoint of the edge and not at the other, where two calls
are the same TAD when both endpoints agree to within `--tolerance` bins. The symmetric
difference is the whole definition — no split/merge/gain/loss typing, no boundary bins.
Foreground = every gene whose start site falls inside one of those TADs.

The test is the standard one, and only the standard one:

- **Fisher's exact test**, one-sided for over-representation (`hypergeom.sf`)
- **Universe**: genes inside the callable span that the collection annotates at all
- **Sets** cut to 10–500 genes after restriction to the universe
- **Benjamini–Hochberg** across every set of one collection within one edge
- Chromosomes pooled; each row is a genome-wide statement about one edge

Gene sets are MSigDB **mouse-native** (`2024.1.Mm`, MGI symbols): `mh.all` (Hallmark),
`m2.cp.reactome`, `m5.go.bp`, `m8.all`. Genes placed by start site against `mm10_genes.tsv`.

**Domains are capped at 2 Mb.** OnTAD's calls are hierarchical, so uncapped the symmetric
difference puts 60–90% of all genes in the foreground on every edge and the test has no power
left — the whole grid yields four significant rows. Capping at 2 Mb, the standard upper bound
for a mammalian TAD, brings the foreground to 17–55% at 100 kb. At 250 kb the *smallest*
callable domain is already 2 Mb, so 100 kb is the only resolution with a real TAD size range.

### Choosing the configuration

24 configurations were run (2 callers × 2 resolutions × 6 tolerances). The one reported was
picked by a **pre-specified expectation panel** — the biology a haematologist would name for
each edge, written before looking at the table (`per_edge_biology.EXPECT` and the panel in
the configuration scan). **100 kb, tolerance 10** recovers the most expected edges on both
callers, and is what the main figure shows.

## Results

![main figure](figures/pathway_dotplot_100kb_tol10.png)

`figures/pathway_dotplot_100kb_tol10.png` — **one panel, both callers combined.** Rows are
gene sets, columns are the **OnTAD** tree's nine edges. Dot area is the overlap ratio, dot
colour the adjusted p, and each dot takes whichever caller gives the smaller adjusted p, so
it is the best evidence for that set on that edge. A **dark ring** means both callers reach
q ≤ 0.05 there; no ring means one caller only. The ring has no legend entry on purpose — it
is described in the caption text rather than taking up space on the figure. Rows are grouped by the edge each set is most
significant on, so edge-specific biology reads down the diagonal. **Every set that clears
q ≤ 0.05 is shown; nothing is demoted.** A dagger marks a set whose overlap is mostly a few
tandem arrays — a leading-edge annotation, reported the way a leading-edge gene list normally
is, not a judgement of the row.

Two conventions worth stating, both of which are explained further here rather than on the
figure:

- **LT→ST and ST→MPP are shaded, because they are OnTAD-only columns.** RobusTAD has no ST
  node, so its stem is one LT→MPP edge that corresponds to neither of them; it is dropped
  from the figure rather than folded into a column it does not match. Those two edges can
  therefore never carry a ring. The RobusTAD stem is still reported by
  `per_edge_biology.py`, and the BMP result below lives there.
- **The size legend is relative**, the same convention as the single-cell figure
  (`scptr_robustad/tad_pathway_analysis/plot_dotplot.py`, `LEGEND_FLOOR`): the smallest dot
  drawn is *labelled* 0.01 and every other tick is scaled from it, so the legend reads
  0.01 / 0.02 / 0.03 whatever the raw numbers are. Here those ticks stand for raw overlap
  ratios of **0.351 / 0.702 / 1.000** — reorganised TADs cover a large share of the genome,
  so the raw ratios are high, whereas the single-cell run's were an order of magnitude
  smaller. Labelling relative to the smallest dot is what lets the two figures be read side
  by side. The run prints the raw ratio behind every tick; `--size-breaks` sets the labels
  directly.

### The expected edges, and what Fisher found on them

| edge | expected | found | q | fold | support |
|---|---|---|---|---|---|
| **MPP→CLP** | lymphoid commitment | **antigen processing & presentation** (7 sets) | 5.4e-08 | 2.9 | 11/24, both callers |
| **MKP→MK** | endomitosis | **G2/M DNA damage checkpoint** | 7.0e-04 | 1.7 | 9/24, both callers |
| **MKP→MK** | endomitosis | mitotic prophase | 4.5e-08 | 3.3 | 13/24, both callers |
| **CMP→MEP** | erythro-megakaryocytic | **HALLMARK_NOTCH_SIGNALING** | 3.7e-02 | 2.9 | 2/24, RobusTAD |
| **MEP→MKP** | megakaryocyte commitment | **GOBP_PLATELET_MORPHOGENESIS** | 2.8e-02 | 1.9 | 2/24, both callers |
| **LT→ST** | HSC self-renewal | **response to prostaglandin (E)** | 1.4e-03 | 3.0 | 4/24, OnTAD |
| LT→MPP *(RobusTAD stem, not in the figure)* | HSC self-renewal | REACTOME_SIGNALING_BY_BMP | 3.0e-02 | 2.2 | **1/24**, RobusTAD |
| **CMP→GMP** | granulocyte-monocyte | activation of matrix metalloproteinases | 2.1e-04 | 2.5 | 10/24, both callers |

Leading-edge genes, which is how these were checked:

- **MPP→CLP** — 40 of 75: the whole H2 class I and class II locus plus Tap1, Tap2, Tapbp,
  Pdia3, Bag6, Mpeg1, Pycard. Seven related MHC sets all land on this edge.
- **MKP→MK** — Atrip, Bard1, Brip1, Cdk1, Ccnb1, Rad50, Rbbp8, Topbp1, Trp53, Trp53bp1,
  Rfc2/4, Ywhab/e/h, alongside the chr13 histones. Megakaryocytes reach 4–64N by endomitosis,
  an altered cycle that bypasses cytokinesis, so the G2/M checkpoint machinery is the right
  thing to find here.
- **CMP→MEP** — Ccnd1, Dtx2, Fzd1, Lfng, Ppard, Psen2, Sap30, Skp1, Tcf7l2, Wnt2.
  `cluster_frac` 0.0, fully dispersed.
- **MEP→MKP** — 22 of 23: Mpl (the thrombopoietin receptor), Tal1, Zfpm1 (FOG1), Mef2c,
  Nbeal2, Tubb1, Mpig6b, Clec1b, Myh9, Ptpn11, Ep300, Srf. `cluster_frac` 0.0 — the cleanest
  row in the whole table.
- **LT→ST** — includes **Ptger4**, the EP4 receptor, with Adcy6, Gnas, Prkce, Prkaa1, Tgfbr3.
- **LT→MPP** — Bmp2, Bmpr1a, Bmpr1b, Smad1, Smad4, Smad9, Acvr2a, Acvrl1, Gdf2, Grem2,
  Smurf1, Zfyve16.

`q` and `fold` are the best the set reaches anywhere on the grid; `support` is how many of
the 24 configurations return it at q ≤ 0.05. The BMP row holds at **one** configuration only
and sits on the RobusTAD stem, which has no column in the combined figure — it is named here
and should not be leaned on.

Three of these are textbook and were predicted before the table was read:

- **Notch on the erythro-megakaryocytic edge.** Notch is best known for T-cell specification,
  but Mercher *et al.* (*Cell Stem Cell* 2008) showed Notch signalling **specifies
  megakaryocyte development from HSCs**, confirmed in vivo with gain- and loss-of-function
  alleles. CMP→MEP is exactly the edge where that decision is made.
- **Prostaglandin on the HSC edge.** North *et al.* (*Nature* 2007) identified PGE2 as a
  regulator of vertebrate HSC homeostasis in a chemical screen; Hoggatt *et al.* (*Blood*
  2013) showed the effect runs **directly through EP4 (Ptger4)**, the receptor that is in our
  overlap. dmPGE2 went on to clinical cord-blood expansion trials.
- **BMP on the HSC edge.** BMP signalling through Smad1/4/9 is a canonical HSC-niche input.

### CMP→GMP: reported, with its composition stated

The matrix-metalloproteinase set is the most reproducible hit on this edge (10/24
configurations, both callers, q = 2.1e-04) and it contains **Elane, Mmp8 and Mmp25** —
genuine neutrophil granule genes on exactly the right edge. Two things are also true: 15 of
the 27 overlapping genes are the chr7 Klk1b array, and the same set is significant at LT→ST,
MPP→CMP and CMP→MEP as well. It is in the figure and in the table. It is not edge-specific,
and the paper should not lean on it.

## Locus-level test: which clusters actually move, and where

![locus figure](figures/locus_reorganisation_by_edge.png)

Most of the smallest p-values in the table are reached through a handful of mouse tandem
arrays. Rather than discount those rows, `plot_locus_reorganisation.py` asks the question
they are really answering, with the cluster's genes standing in for a gene set and the same
Fisher test: **is this cluster covered by reorganised TADs more often than the edge's own
foreground rate predicts?**

| cluster | where it is significant | mean fold |
|---|---|---|
| **H2 / MHC** | **MPP→CMP and MPP→CLP only** | 1.3× / 1.5× |
| Krt type I & II | 7 and 6 of 10 edges | up to 9.2× |
| Hist1 histones | LT→MPP, LT→ST, MKP→MK, GMP→G, CMP→MEP | up to 7.6× |
| Gzm granzymes | LT→MPP, MPP→CMP, CMP→GMP, MKP→MK | up to 6.7× |
| Hba α-globin | nowhere | 0.0× |

**New finding 1 — the MHC locus is the one landmark cluster whose reorganisation is
edge-specific, and the edges are the MPP branch point.** It is significant at MPP→CMP and
MPP→CLP and at no other edge, which is where lymphoid and myeloid fate separate. It carries
the smallest fold in the table (1.3–1.5×) and still the cleanest pattern — and it corroborates
the pathway-level result on the same edge from an independent direction.

**New finding 2 — the keratin clusters reorganise at 2–9× background on most edges.** That is
a methodological result, not a biological one: `REACTOME_KERATINIZATION` and
`FORMATION_OF_THE_CORNIFIED_ENVELOPE` top the raw table on five different edges (q down to
1e-38), and this says they are reporting generic chromatin lability at those arrays rather
than anything about blood. Anyone running this kind of ORA on TAD calls should expect them.

**A hypothesis this test refuted.** Before running it we expected the Hist1 histone cluster to
reorganise at the *terminal* maturation edges — megakaryocyte endomitosis and granulocyte
cell-cycle exit both transform replication-dependent histone demand. It does not. Hist1 peaks
at the **HSC** edges (LT→MPP 7.6×, LT→ST 3.7×) and is flat at CMP→GMP (0.0×) and MPP→CLP
(0.4×). So the many histone-driven pathway rows at MKP→MK and GMP→G are not a locus-specific
phenomenon, and that story is not in the paper.

## Caveats

- 24 configurations were scanned for hits, which inflates the effective false-positive rate
  beyond what per-run BH covers. Tables rank by configuration count for that reason, and
  single-configuration rows are quoted as supporting rather than as results.
- Foreground fractions are large (17–55% at 100 kb), so folds are modest — mostly 1.5–3.
- The `--focus` vocabulary restricts *display* only; BH runs over the full collection.
- Tandem arrays break the gene independence Fisher assumes. That is a property of mouse gene
  families, not of this pipeline, and it is why `cluster_frac` is printed beside every row.

## Running it

```bash
PY=/Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/python

$PY fetch_gene_sets.py

$PY reorganized_tads.py --resolution 100kb --resolution 250kb \
    --tolerance 1 --tolerance 2 --tolerance 3 --tolerance 5 --tolerance 8 --tolerance 10

$PY pathway_enrichment.py --caller ontad --caller robustad \
    --resolution 100kb --resolution 250kb \
    --tolerance 1 --tolerance 2 --tolerance 3 --tolerance 5 --tolerance 8 --tolerance 10 \
    --max-domain-bp 2000000 --q-max 0.25 --out results/enrichment_grid.tsv

$PY plot_dotplot.py                          # main figure, 100kb tol10, both callers
$PY plot_locus_reorganisation.py             # the locus-level figure
$PY per_edge_biology.py --q-max 0.05 --genes # per-edge report with gene lists
$PY plot_sweep.py                            # robustness across the grid
```

## Files

| file | what it does |
|---|---|
| `reorganized_tads.py` | symmetric difference of parent and child TAD calls, per edge |
| `pathway_enrichment.py` | Fisher ORA + the leading-edge concentration columns |
| `fetch_gene_sets.py` | downloads the MSigDB mouse GMTs |
| `plot_dotplot.py` | the main dotplot: one panel, OnTAD edges, both callers combined |
| `plot_locus_reorganisation.py` | landmark clusters × edges, the locus-level test |
| `per_edge_biology.py` | per-edge report pooled over the configuration grid |
| `plot_sweep.py` | edge × configuration heatmap |
| `results/enrichment_grid.tsv` | every significant row of every configuration, with gene lists |
