# Merge and split events between stage medians, and what biology they support

This folder takes the merge/split machinery that `hemato_cells/` runs along the edges of
the hematopoietic tree and applies it to the single-cell embryo clustering of
`scptr_robustad/`, then asks whether the events it finds line up with anything known about
mouse preimplantation development: transcription, imprinting, X inactivation, CTCF.

The short answer is in [`results/REPORT.md`](results/REPORT.md), which is generated from
the tables and reports the diagnostics before the biology, and it has two halves.

**The event-based analysis is negative.** The events are largely not reproducible across
bin sizes, no transition-specific transcriptional signal survives a test that respects how
the runs are nested, and merge and split boundaries are indistinguishable from every other
boundary of the same medians by a test that could only have caught a departure of 46 per
cent or more. The pipeline, the coordinates, the annotated loci and the negative controls are
all here and are the useful part; the locus list should be read as candidates, not findings.

**The stage medians themselves are positive, and that is the result to quote** (section 3
of the report, `stage_boundary_maturation.py`, `results/figures/embryo_boundaries.png`, panels a and b).
Scoring each stage median on its own rather than scoring the differences between medians,
CTCF density and gene density at a median's boundaries fall to a minimum at the 2-cell
stage and climb monotonically to the 64-cell stage. The trend holds at all three bin sizes
separately, on both annotations, at every boundary tolerance and under all three relocation
pools, on 17 to 19 of 20 chromosomes. This is the progressive establishment of TADs over
preimplantation development that Du et al. 2017 and Ke et al. 2017 report from bulk
allele-resolved Hi-C, recovered here from single-cell calls and a clustering alone. It also
explains the negative result above: the event boundaries are pooled over all stages, so
they are dominated by the depleted early ones.

**Each caller against its own held-out cells: the class is real, the loci are not**
(section 4 of the report, `merge_split_locus_validation.py`,
`merge_split_class_summary.py`, `merge_split_cluster_enrichment.py`,
`results/figures/robustad_merge_split.png` and `results/figures/ontad_merge_split.png`).
Section 2 rejects merge and split as a separate class using agreement between the two
callers, and that criterion is arguable: two TAD callers carry different positional biases
and cannot be asked to put a boundary in the same 250 kb bin. This section drops it. Each
caller is scored against **its own** held-out cells and the two are never compared by
position -- a run uses a few dozen cells, the per-cell call tree holds several hundred, and
everything the runs did not touch is a test set.

At 500 kb, **split is the half that replicates on both callers**: RobusTAD +0.032 on 17 of
18 chromosomes (p = 1e-4), OnTAD +0.029 on 18 of 20 (p = 4e-4). **Merge is weaker and only
clears significance on OnTAD** (-0.021, 16 of 20, p = 0.012); RobusTAD puts it in the right
direction at every bin size but never significantly. Broken down by transition, every row
points the predicted way on both callers at 500 kb -- 8 of 8 on OnTAD, 6 of 6 on RobusTAD --
so no single transition carries it. Two configurations drop out for coverage rather than
signal, and neither is a failure: the RobusTAD 1 Mb clustering uses **all 330 cells of the
pool** so it has no held-out set at all, and RobusTAD leaves too few held-out 1-cell cells
to score the first transition. RobusTAD at 250 kb is testable and null, and the figures show
every bin size including those.

Individual loci are a different matter. **None of the 2551 RobusTAD or 1088 OnTAD merge/split
bins clears a 10 per cent false discovery rate**, and the reason is not caller disagreement,
because neither test leaves its own caller. The background of per-bin penetrance change is
heavy tailed -- a few per cent of *all* bins move as much as the reported ones do with no
event called on them -- so the strongest locus is an ordinary member of that distribution.
Two traps are recorded in the scripts so they are not repeated: a 2x2 Fisher test on these
bins assumes the cells are exchangeable and reports hundreds of false loci at q down to
1e-9, and a curated gene set counted by *gene* rather than by *cluster* reports a 2.45-fold
imprinting enrichment for OnTAD merges that is entirely the fact that 20 of the 125
imprinted genes are one locus on chr7. Counted by cluster, no set on either caller beats
chance.

**Two things that were asked of the reorganised boundaries and did not survive**
(`boundary_gene_activity.py`, quoted in section 5). Section 1 finds that a reorganised
boundary carries more genes than a stable boundary of the same medians; the genes are not
*different* genes -- matched on bin size, on gene count and on neighbourhood gene density,
no expression property separates the two classes on both callers -- and they are not the
genes that change at the transition the boundary moved at. Nor can the loci be named: the
two callers' most reorganised bins overlap *below* chance, so a gene list read off either
one would be that caller's noise. That is why this folder names no genes.

## Where the events come from

Each clustering run in `scptr_robustad/results_updated/scptr_runs_all_cells/` produces five
cluster medians, one consensus TAD set per cluster, and each cluster is matched to the
developmental stage most of its cells came from. Walking those medians along

```text
1-cell -> 2-cell -> 4-cell -> 8-cell -> 64-cell
```

and transforming each into the next with `tad_sequence.transform_sequence` gives the domain
rearrangements the clustering implies. This is the same transformation, with the same
tolerance semantics, that `hemato_cells/analyze_edge_merge_split_events.py` uses on tree
edges, so the event definitions match between the two analyses.

Clusters are matched to stages exactly as the paper's alluvial figure matches them, by the
assignment that puts the most cells on the diagonal, so a median called "4-cell" here is
the one whose ribbon flows to 4-cell in that figure.

Events come in four kinds. A **split** is one domain becoming two, a **merge** is the
reverse, and a **domain gain** or **domain loss** is a domain appearing or disappearing
without a partner. Only splits and merges whose two domains are adjacent to within a bin
are treated as a single boundary appearing or disappearing; the rest replace a domain with
two distant pieces and are kept but flagged by `gap_bins`.

## Data

- **Hi-C / TAD calls**: RobusTAD on the single-cell Hi-C of Collombet et al. 2020, *Nature*
  580:142 (GEO GSE129029), stages 1-, 2-, 4-, 8- and 64-cell, assembly **mm10**.
- **Genes**: UCSC `mm10.refGene`, collapsed to one span and one TSS per symbol.
- **Expression**: Deng et al. 2014, *Science* 343:193 (GEO GSE45719), single-cell RNA-seq
  of mouse preimplantation development, averaged per stage on a log scale. Its stages line
  up with the Hi-C stages, which is what makes a transition-matched comparison possible.
  The mapping is in `gene_sets.STAGE_TO_DENG`: the 2-cell Hi-C samples are read against the
  mid and late 2-cell transcriptomes, after major zygotic genome activation, and the
  64-cell samples against the early and mid blastocyst.
- **CTCF**: ENCODE ENCSR179IQN, CTCF ChIP-seq in C57BL/6J mouse embryonic stem cells, mm10.
  A cell line, not an embryo, so it says where CTCF tends to bind, not where it was bound
  in these cells.
- **Curated gene sets**: `gene_sets/*.txt`, each carrying its source and its reasoning in
  the file header. Imprinted genes from Geneimprint; the non-canonical H3K27me3-imprinted
  subset from Inoue et al. 2017, *Nature* 547:419; X-inactivation escapees from Berletch
  et al.; the Hox clusters, the pluripotency and totipotency network, the first lineage
  split, and housekeeping genes as a positive control.

Everything except the curated sets is downloaded by `fetch_reference_data.py` into
`reference/`, which is gitignored.

## Coordinates

Bin indices are turned back into mm10 base pairs by `coordinates.py`. The per-chromosome
runs use 1-based bins, so a TAD `[a, b]` covers `[(a-1) * res, b * res)`, and the mapping is
exact.

The genome-wide (`all`) runs are **not** used here. They lay the chromosomes end to end
using `genome.chrom.sizes` at the repository root, which is an mm9 file, while the data are
mm10. mm10 is longer than mm9 on eleven chromosomes, so a TAD near the end of one of them
can be encoded into the bin range of the next, and several chromosomes do reach that far.
`call_median_events.py` refuses those runs rather than producing coordinates that are
quietly wrong near chromosome ends.

## How the tests are built

Every positional test compares an observed set of windows with the same windows relocated.
The windows keep their count, their widths and their chromosomes, and each slides to a
random position inside the **callable region** of its chromosome, the stretch the medians of
that run actually cover. A whole window must fit inside that region: letting a wide window
hang off the end would score a truncated window against an untruncated observation and
manufacture enrichment. That artefact was real here, and correcting it removed almost all of
what first looked like domain-scale enrichment.

Annotation works at two scales and they are not interchangeable. The **boundary scale** is
the single bin carrying a boundary that appeared, disappeared or survived, and it is what
every test is anchored on. The **domain scale** is the whole domain, which is useful for the
splits and merges and much less so for the rest: the medians contain domains spanning most
of a chromosome, and `annotate_events.py` counts the genes in a domain wider than 20 Mb
without listing them, because naming the eight hundred genes inside one says nothing about
any of them.

Two further guards matter for reading the output.

- **`all_genes` is a density read, not a hypothesis.** A window set that is gene poor
  overall will look depleted for every gene set and one that is gene dense will look
  enriched for every gene set, so every other row has to be read against it.
- **Transition matching separates signal from gene density.** Each window set is scored
  against the transition it was called at *and* against the other three. A window set that
  merely sits in a transcriptionally busy region scores the same on all four. The reported
  contrast, `matched_minus_other`, is what remains.

Tolerances 0 to 5 of one bin size are six re-analyses of the same clustering of the same
cells and are not independent. Anywhere results are pooled, tolerances are averaged within a
bin size first and the test runs over the (bin size, transition) cells that remain. Pooling
over all 18 configurations instead roughly doubles the apparent significance of everything.

## Running it

Everything runs in the `snakemake-tutorial` environment.

```bash
cd scptr_robustad/merge_split_real_biology
PY=/Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial/bin/python

$PY fetch_reference_data.py --with-ctcf     # ~110 MB, once
$PY call_median_events.py --all             # 360 runs, ~15 s
$PY annotate_events.py --all
$PY enrichment.py --all --permutations 5000
$PY expression_analysis.py --all --permutations 5000
$PY ctcf_analysis.py
$PY consensus_events.py
$PY chrx_analysis.py
$PY summarise_results.py
$PY plot_summary.py
```

or the whole thing with `snakemake -s Snakefile --cores 4`.

Any script also runs on one configuration, which is the way to look at a single case:

```bash
$PY call_median_events.py --resolution 500kb --chrom chr1 --tolerance 1
$PY enrichment.py --resolution 500kb --tolerance 1
$PY consensus_events.py --event-type split --min-resolutions 2
```

## Scripts

| script | what it does |
| --- | --- |
| `coordinates.py` | bin indices to mm10 base pairs, for both run layouts |
| `fetch_reference_data.py` | downloads refGene, the Deng expression atlas and CTCF peaks |
| `gene_sets.py` | loads gene coordinates, curated sets, expression; derives the per-transition up/down modules |
| `call_median_events.py` | median-to-median splits, merges, domain gains and losses, plus the boundaries each changes and the ones that survive |
| `annotate_events.py` | genes at each event, at the domain scale and the boundary scale |
| `enrichment.py` | gene-set overlap against the relocation null |
| `expression_analysis.py` | how much the genes at an event move, matched transition against the other three |
| `ctcf_analysis.py` | CTCF peak density at the boundaries |
| `consensus_events.py` | events called at more than one bin size, with their genes |
| `chrx_analysis.py` | chrX boundary loss against the autosomes; distance to escapees |
| `boundary_profile.py` | CTCF and gene density as a function of distance from a call |
| `tolerance_sweep.py` | what positional tolerance does to the CTCF, gene set and reproducibility results |
| `stage_boundary_maturation.py` | scores each stage median's own boundaries for CTCF and gene density, stage by stage, under three relocation pools |
| `boundary_anchor_resolution.py` | whether a sharper anchor definition (RAD21, CTCF+RAD21 co-bound, distance rather than count) makes these boundaries look like anchors, and whether the 2-cell dip is anchor-specific or territory |
| `merge_split_context.py` | merge and split boundaries against the other boundaries of the same medians, split persistence, and the curated sets at the reproducible loci |
| `candidate_loci.py` | writes `results/CANDIDATES.md`, the short list with literature notes |
| `splithalf_penetrance.py` | cuts the held-out cells in two and scores every merge/split bin on each half, so one half can rank a locus and the other judge it |
| `merge_split_weighted_biology.py` | the alpha sweep: ranks merge/split calls from our own held-out evidence to the CTCF/cohesin anchor and scores each mixture on three readouts |
| `boundary_territory_turnover.py` | log2(feature density at the boundaries a transition gains / the ones it loses), paired inside a transition |
| `plot_weighted_biology.py` | writes `results/figures/weighted_biology.png` |
| `summarise_results.py` | writes `results/REPORT.md` |
| `plot_summary.py` | writes `results/figures/summary.png` |
| `plot_tolerance.py` | writes `results/figures/tolerance.png` |
| `plot_stage_maturation.py` | the stage-trajectory panels, imported by `plot_main_figure.py` |

## Outputs

```text
results/events/<resolution>/<chrom>/tol_<t>_events.tsv       one row per event
                                   tol_<t>_boundaries.tsv    one row per boundary changed or kept
                                   tol_<t>_callable.tsv      where an event could have been called
                                   tol_<t>_manifest.json     stage matching, counts, ARI
results/annotated/    genes at each event
results/anchor/       boundary_anchor.tsv, anchor_dip_tests.tsv
results/enrichment/   gene-set tests
results/expression/   transcriptional response and the transition contrasts
results/ctcf/         CTCF density at boundaries
results/consensus/    loci ranked by how many bin sizes call them
results/chrx/         chrX rate and escapee-distance tables
results/profile/      CTCF and gene density by distance from a call
results/tolerance/    every test repeated across positional tolerances, and the nested sign test
results/stage/        per-stage boundary CTCF and gene density, and the stage trend tests
results/merge_split/  merge/split boundaries against their own medians, persistence, locus gene sets
results/events_ontad/ the same events called from the OnTAD clustering runs
results/stage_ontad/, results/merge_split_ontad/, results/ontad_candidates/
                      the OnTAD replication of sections 9 and 10
results/weighted/<caller>/  the alpha sweep: curve, controls, panel and gene-set folds,
                           and the split-half held-out table it ranks on
results/territory/    what territory each transition trades its boundaries for
results/WEIGHTED_BIOLOGY.md  the alpha sweep written up
results/REPORT.md     the written summary
results/CANDIDATES.md the short list of loci worth checking, with literature notes
results/candidates.tsv the same list with every column
results/figures/      ctcf_boundaries.png, embryo_boundaries.png, merge_split.png
```

## Positional tolerance

Single-cell Hi-C is sparse, so the obvious objection to the null results is that the calls
are jittered: a boundary put at one bin in a median may be the same boundary put two bins
away in the next, and a one-bin window would then miss the CTCF site that is really there.
Three knobs test that, and they are worth keeping straight because they behave differently.

- `--flank-bins` on `enrichment.py`, `ctcf_analysis.py` and `expression_analysis.py` widens
  every boundary window by that many bins on each side. The null windows are widened
  identically, so the comparison stays fair, but a wider window is a blunter one and every
  fold is dragged toward 1 as the flank grows. Read a flanked fold next to the profile.
- `--slop` on `consensus_events.py` and `candidate_loci.py` pads calls in base pairs before
  deciding whether two configurations called the same event. This is the knob that pays:
  loci recovered at more than one bin size go 15, 22, 26, 43 at 0, 0.5, 1 and 2 Mb, while
  the total number of loci falls, so matched calls are merging rather than accumulating.
- `--tolerance` is the *existing*, unrelated parameter: the boundary tolerance the
  clustering itself was run at. It is part of a run's identity, not something applied here.

The verdict is in section 8 of the report. The profile has its minimum at the call itself,
which means the calls are positionally precise, so there is no displaced CTCF signal for a
flank to recover; widening only dilutes. The one place a flank changes a sign is the
imprinted gene set, which goes from depleted at one bin to a few per cent above 1 at ±3
bins or wider at gained and lost boundaries -- far from significant under the nested sign
test, but pointing the same way as two of the candidate loci.

Sparsity does show up, though, on a different axis. The CTCF depletion of section 4 tracks
bin size: 0.84 and 0.83 for gained and lost boundaries at 250 kb, 0.98 and 1.02 at 1 Mb,
where there is no depletion left to explain. Contacts per bin rise fourfold along that
axis, so the depletion is about how reliably a boundary can be placed in thin data rather
than about the genome, and the mechanism is placement bias into quiet territory rather than
displacement of an otherwise good call. The remedy is coarser bins or deeper coverage, not
a wider window.

## The stage trajectory

`stage_boundary_maturation.py` is the only analysis here that scores the medians rather
than the transitions, and it is the only one that comes out positive. For each
(bin size, chromosome, tolerance) run it takes the five stage medians, collects the bins
carrying a domain edge in each, and measures CTCF peak density and gene density in those
bins against the same relocation null the rest of the folder uses.

The design is a within-run comparison: the five stages of a run are scored against
relocations drawn from one pool of callable bins, so a difference between two stages cannot
come from the chromosome, the caller or the region of the genome. Three pools are available
because the obvious objection is that the caller looked in different places at different
stages:

| pool | relocations are drawn from | what it controls for |
| --- | --- | --- |
| `union` | bins covered by any of the five medians | the folder's default |
| `own` | bins covered by that stage's own median | a stage whose domains sit in a systematically different part of the chromosome |
| `common` | bins covered by all five medians, observed boundaries restricted to match | every stage scored on identical territory |

Chromosomes are the replicates. Tolerances are averaged within a (bin size, chromosome)
before testing, because the six tolerances of a bin size are re-analyses of one clustering,
and the three bin sizes are tested separately rather than pooled for the same reason. The
statistic is a Spearman rho of the fold across 2-cell, 4-cell, 8-cell and 64-cell per
chromosome, tested over chromosomes with a Wilcoxon signed-rank.

```bash
python stage_boundary_maturation.py --permutations 5000          # the whole grid, ~45 s
python stage_boundary_maturation.py --resolution 500kb --pools own
```

The zygote is excluded from the trend and reported separately: it sits *above* the 2-cell
stage rather than below it, its medians rest on the fewest cells of any stage, and it is
the stage where the two parental genomes are most distinct and so least well described by
an allele-agnostic call set.

## Merge and split places

`merge_split_context.py` answers the question the folder is named for, and the answer is no.
Section 4 scored merge and split boundaries against relocated windows and found them
depleted; section 9 then made that test uninterpretable, because a median's boundaries are
depleted early in development and not depleted late, so any window set pooled over stages
inherits whichever stages fed it most.

The fix is to change the comparison rather than the null. A merge or split boundary is
scored against **every other boundary of the two medians it was called between**, which
holds the chromosome, the caller, the stage pair and the clustering run fixed. A ratio of
one means an ordinary boundary of its own medians, and one is what comes out, on CTCF and
on gene density, at all three bin sizes. A boundary created by a split is also no more
likely to survive into the 64-cell median than any other boundary of the median it appeared
in.

**Read the `detectable` column before reading the null.** It is the smallest departure from
one this many chromosomes at this spread would have caught at 80% power, and it runs from
0.46 to 1.30. A merge or split boundary would have had to carry half again as much CTCF as
its neighbours to be seen. The null is therefore a statement about how rare adjacent merge
and split events are -- 650 across the whole grid of 360 runs, about five per chromosome and
configuration -- and not about what those events are.

The one apparent signal is a cautionary tale worth keeping. Three subunits of PRC1.6 --
PCGF6, E2F6 and RYBP -- sit under three separate merge loci, at p = 0.0035 against loci slid
to random positions, and find nothing at the split loci. That hypothesis was read off these
same loci, so the only honest test is the prediction it makes about data it was not fitted
to: PRC1.6 silences germline and meiotic genes in the early embryo, so its targets should
sit on merge loci as well as its subunits. None of the forty-five does, against an
expectation of about one. `gene_sets/prc16_subunits.txt` and
`gene_sets/prc16_germline_targets.txt` record both halves so the coincidence is not
rediscovered.

```bash
python merge_split_context.py --permutations 5000
```

## The OnTAD replication

`scptr_ontad/` runs the same clustering on the same cells with OnTAD calls instead of
RobusTAD ones, which makes it the one control this project can run that changes the TAD
caller and nothing else. Every script here takes `--run-dir` and `--events-dir`, so the
whole analysis runs against it unchanged:

```bash
ONTAD=../../scptr_ontad/results_updated/scptr_runs_all_cells
python call_median_events.py --all --run-dir $ONTAD --out-dir results/events_ontad
python stage_boundary_maturation.py --run-dir $ONTAD --out-dir results/stage_ontad
python candidate_loci.py --events-dir results/events_ontad --out-dir results/ontad_candidates
python merge_split_context.py --run-dir $ONTAD --events-dir results/events_ontad \
    --candidates results/ontad_candidates/candidates.tsv --out-dir results/merge_split_ontad
```

The OnTAD grid is smaller and ragged, 127 runs against 360, because not every chromosome
and tolerance was run; the 1 Mb column rests on 12 chromosomes and is the weakest
throughout. Adjacent merge and split events number 227 against 650.

**The replication separates the folder's two results.** The stage trajectory of section 9
reproduces: the 2-cell minimum appears in all six caller-by-bin-size combinations, and the
trend holds at rho +0.52 and +0.43 at 250 kb and 500 kb against +0.51 and +0.52 for
RobusTAD. The absolute level does not reproduce, and should not: RobusTAD sits at 0.69-1.01
and OnTAD at 0.93-1.15, which is section 8's placement-bias argument coming true when a
different caller is put on the same contacts.

The merge and split contrast of section 10 does the opposite. The two callers **disagree in
sign on six of eight comparisons**: RobusTAD makes those boundaries CTCF-depleted at
p = 0.004, OnTAD makes them enriched at p = 0.022, and the OnTAD split rows survive the
matched interior background that killed the RobusTAD ones. Whichever caller had been run
first would have produced a publishable sentence with the opposite meaning. Do not report
anything from section 10 as biology.

## Weighting the calls toward the biology

`merge_split_weighted_biology.py`, written up in
[`results/WEIGHTED_BIOLOGY.md`](results/WEIGHTED_BIOLOGY.md) with
`results/figures/weighted_biology.png`.

The merge/split set need not be taken as fixed. A weight `alpha` slides the *ranking* of the
called events from **our own evidence** at 0 -- the held-out per-cell penetrance shift, Hi-C
from cells no run touched, no annotation -- to **what a boundary is supposed to look like**
at 1, CTCF together with cohesin. The top *K* of each ranking is a set of the same size, so
the sets are comparable, and the two ends are the controls that make the middle readable: a
result is a peak strictly inside `(0, 1)` that clears both ends, the permutation null and
the two shuffled-ranking controls.

**The anchor adds nothing, on three readouts and both callers.** Transcription is not there
to be found in the first place: a transition-matched readout is null at every `alpha`, and
so is the same readout on all 4290 domain-gain and 4259 domain-loss boundaries of the OnTAD
500 kb grid, which is far more than merge and split can offer. Held-out replication is
there, strongly -- ranking on one half of the held-out cells lifts the mean statistic in the
disjoint other half from +0.25 to +1.98 -- but every step toward CTCF costs it, monotonically,
down to +0.42 at `alpha = 1`, which does not clear the null. The machinery is not what is
null: `--self-test` plants a transition-matched signal and recovers it at z = +4.3 to +8.1.

The by-product is a negative control worth keeping. The loci that *do* replicate look
significantly **opposite** on the two callers -- OnTAD's are gene rich at 1.22-fold, p = 0.008,
RobusTAD's gene poor at 0.74-fold, p = 0.002, with every active track following gene density
in both. On one caller alone that reads as "the calls that reproduce sit at active
promoters". It is section 8's placement bias, surviving a selection made on held-out cells.

## What territory a transition trades

`boundary_territory_turnover.py`. Every transition destroys boundaries and creates others,
and comparing the two *inside one transition* holds the chromosome, the caller, the run and
the pair of medians fixed, so `log2(gained / lost)` needs no null at all.

**1-cell to 2-cell trades gene-dense territory for gene-poor, and 2-cell to 4-cell puts it
back** -- same sign on both callers at all three bin sizes, 6 of 6 at each transition, and
significant at 500 kb on both. 4-cell to 8-cell is flat and 8-cell to 64-cell is where the
callers disagree in sign, so nothing should be taken from it. This is the mechanism under
the stage-maturation curve of section 9: the 2-cell minimum is one transition discarding
boundaries in active territory and the next installing them there again.

It is a **territory** result, not a factor result -- CTCF, cohesin, CpG islands, H3K4me3,
H3K27ac, POLR2A, EP300, NANOG and POU5F1 all carry the same numbers as plain gene density.
H3K9me3, the one repressive gene-poor track, is the control and it holds at the zygotic
transition on both callers. Restricted to split-created against merge-destroyed boundaries
the contrast points the same way but rarely reaches significance: **nothing here is specific
to merges and splits**, they follow the same rule as domains appearing and disappearing.

## What to be careful about

- **Resolution.** A bin is 250 kb to 1 Mb. A boundary is located to one bin, so a gene named
  at a boundary is a gene in a window that typically holds several, and no claim about a
  specific promoter or CTCF site is available at this resolution.
- **The medians share about a fifth of their domains between consecutive stages.** The
  clustering separates the stages well *because* of that, which means a high ARI is not
  evidence that any individual rearrangement is real.
- **These boundaries are not canonical boundaries.** They are depleted of genes and of CTCF
  relative to their own chromosomes, where real TAD boundaries are enriched for both. The
  depletion is centred on the call and about eight bins wide, so it is a property of the
  territory the calls land in rather than of the exact bin, and no positional tolerance
  removes it.
- **The cells are a subsample.** Each run clusters a few dozen cells, not the full set, for
  the reasons described in the paper's supplementary.
- **The TAD calls are not allele resolved**, which is why the chrX test is weak: a
  paternal-X effect is diluted by the active X in every cell and by the male embryos.
- **`CANDIDATES.md` is a list of hypotheses.** Its ranking rewards agreement between
  configurations that share input data, none of its loci is individually significant, and
  the genome-wide tests behind it are null. The literature notes in it come from published
  work on those genes, not from this analysis.
- **The curated `imprinted_noncanonical_h3k27me3` set is partial.** Inoue et al. report 76
  genes; only the well-attested members are listed, so an overlap with it is a pointer and
  never a count.
