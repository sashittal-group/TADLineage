# A biology-first follow-up to DiffDomain

The supplied paper, Hua *et al.*, **DiffDomain enables identification of
structurally reorganized topologically associating domains** (Nature
Communications 2024, DOI: 10.1038/s41467-024-44782-6), makes one point that
changes the priority of this analysis: a TAD can reorganize without gaining,
losing, splitting, or merging a boundary.  It calls these **strength-change**
TADs.  Therefore, a gene list from sparse merge/split calls is not the main
biological test of the Nussinov/clustering medians.

## Result to lead with now

**The stage medians progressively acquire boundaries at canonical,
CTCF-associated positions after the 2-cell stage.**  In RobusTAD calls, the
CTCF-location enrichment at median boundaries is lowest at 2-cell and rises
through 64-cell at every resolution: mean per-chromosome Spearman rho is
0.51, 0.52, and 0.45 at 250 kb, 500 kb, and 1 Mb, respectively.  The trend is
positive on 18/20, 17/20, and 16/20 chromosomes (sign-test p = 7.6e-5,
7.3e-4, and 0.00235).  It reproduces with OnTAD (rho 0.58, 0.49, and 0.38).

This is a real developmental result: it recovers the progressive establishment
of TAD architecture during mouse preimplantation development from the
single-cell-derived medians.  The supporting data and plot are in
`results/stage/`, `results/stage_ontad/`, and
`results/figures/embryo_boundaries.png`.

Careful wording matters.  The CTCF peaks are from mouse ES cells, not from the
embryos.  Thus the data show increasing **alignment to genomic positions that
are CTCF-rich in a related pluripotent cell type**; they do not demonstrate
that CTCF occupancy itself rises in the embryo.

## Event-based result that is defensible

**Boundary turnover, rather than the label “split” or “merge”, marks
regulatory-dense genomic positions.**  Relative to boundaries that persist in
the same pair of medians, domain-gain/loss boundaries have 1.26-fold more
CTCF peaks and 1.34-fold more genes in RobusTAD, and 1.15-fold and 1.17-fold
in OnTAD.  The CTCF and gene contrasts are positive on 19–20 of 20
chromosomes for each caller (sign-test p <= 4e-5).

This is the strongest result that comes directly from the median transition
analysis.  It does **not** identify a particular gene or transcription factor:
the high-turnover loci fail to overlap across the two callers, and genes at
turnover boundaries are not more transition-responsive after matching their
local gene density.  It should be presented as a genomic-property result, not
as a target-gene discovery.

## Do not claim a merge/split-specific biology yet

Merge and split calls are only about 3% of events (2,402 of 82,771 in the
RobusTAD grid).  More importantly, their loci overlap across RobusTAD and
OnTAD at chance (fold 0.94–0.96 under a chromosome-preserving null).  The two
callers also disagree on whether these boundaries are relatively CTCF-rich.
No named merge/split gene, including the exploratory PRC1.6 observation,
should be promoted to a biological result from these data.

## The next experiment to run: strength change in stable domains

This is the direct, feasible experiment inspired by DiffDomain and is the
best route to a specific gene or regulatory programme.

1. Create depth-matched pseudo-bulk contact maps from the raw GSE129029 cells
   for each stage.  Resample the same number of cells per stage and repeat the
   resampling, rather than treating the existing stage labels as replicates.
2. Start with the 2-cell -> 4-cell and 4-cell -> 8-cell transitions, where
   zygotic-genome activation and early lineage preparation are relevant.  Use
   TADs whose two boundaries persist in both median callers; this explicitly
   separates contact-strength change from boundary turnover.
3. For each shared TAD, compare the distance-normalized intra-TAD contact
   matrix between stages (DiffDomain if installed; otherwise a prespecified
   within-TAD contact-enrichment statistic).  Call a domain only if it passes
   FDR control in held-out pseudo-bulk resamples and has the same direction in
   RobusTAD- and OnTAD-defined domains.
4. Test whether strength-gaining and strength-losing domains contain genes
   whose expression changes in the matched Deng transition.  Match null
   domains on chromosome, width, gene count, and baseline expression.  Report
   a named gene only after this domain-level association and cross-resample
   reproducibility both pass.

The decisive outcome would be: “during the 2-cell-to-4-cell transition,
stable-boundary TADs that gain internal contact strength are enriched for
transitionally induced genes.”  The opposite direction or a null result is
also informative, but it avoids the unsupported assumption that a structural
change must look like a split or merge.

## A second, optional DiffDomain-style experiment

Estimate **cell-to-median TAD variability** within each developmental stage:
compare every cell's contact map (or its called TADs, if maps are too sparse)
with a depth-matched pseudo-bulk stage reference, then test whether variability
falls from 2-cell to 64-cell.  This tests the paper's single-cell insight that
TADs differ in cell-to-population variability, and it provides a mechanistic
companion to the boundary-maturation result.  Depth and cell-count matching is
essential, since later stages otherwise have more cells and contacts.

## What data would make a TF-level result credible

The local mESC tracks already in `reference/` are useful only for generating
hypotheses.  A claim about CTCF, cohesin, OCT4, NANOG, or H3K27me3 *changing*
over embryo development requires stage-matched embryo CUT&Tag/ChIP-seq (or a
published, matched embryo atlas).  Once such data are available, test
differential occupancy within the reproducible strength-change TADs, not in
the current caller-specific merge/split locus list.
