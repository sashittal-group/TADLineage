#!/usr/bin/env python3
"""Gene coordinates, gene sets and stage expression, loaded from ``reference/``.

Two kinds of gene set are used side by side.

Curated sets (``gene_sets/*.txt``)
    Named loci taken from the literature: imprinted genes, the non-canonical
    H3K27me3-imprinted subset, X-inactivation escapees, the Hox clusters, the
    pluripotency and totipotency network, the first lineage split, and housekeeping genes
    as a positive control. Each file carries its source and its reasoning in the header.

Derived sets (from ``reference/deng2014_stage_expression.tsv``)
    What actually changes transcriptionally at each of the four transitions the events are
    called on, computed from the Deng et al. 2014 single-cell atlas rather than assumed.
    These are the sets that make the comparison specific: an event set called at the
    2-cell to 4-cell transition is asked about the genes that move at *that* transition and
    about the genes that move at the others, so a hit only counts if it is transition
    matched.

The Hi-C stages of Collombet et al. 2020 are matched to Deng stages by
``STAGE_TO_DENG``. The 2-cell Hi-C samples are read against the mid and late 2-cell
transcriptomes, which is after major zygotic genome activation, and the 64-cell samples
against the early and mid blastocyst, which is the closest cell count in the atlas.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE_DIR = HERE / "reference"
CURATED_DIR = HERE / "gene_sets"

STAGE_ORDER = ["1-cell", "2-cell", "4-cell", "8-cell", "64-cell"]
TRANSITIONS = [f"{a}->{b}" for a, b in zip(STAGE_ORDER, STAGE_ORDER[1:])]

# Hi-C stage -> the Deng et al. 2014 stages averaged for it
STAGE_TO_DENG = {
    "1-cell": ["zy"],
    "2-cell": ["mid2cell", "late2cell"],
    "4-cell": ["4cell"],
    "8-cell": ["8cell"],
    "64-cell": ["earlyblast", "midblast"],
}


def load_gene_table(path=None):
    """``{symbol: {"chrom", "start", "end", "strand", "tss"}}`` from the mm10 gene table."""
    path = Path(path or REFERENCE_DIR / "mm10_genes.tsv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run fetch_reference_data.py to build it."
        )
    genes = {}
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: i for i, name in enumerate(header)}
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            genes[fields[index["symbol"]]] = {
                "chrom": fields[index["chrom"]],
                "start": int(fields[index["start"]]),
                "end": int(fields[index["end"]]),
                "strand": fields[index["strand"]],
                "tss": int(fields[index["tss"]]),
            }
    return genes


def genes_by_chromosome(genes):
    """Genes bucketed by chromosome and sorted by TSS, for interval queries."""
    buckets = {}
    for symbol, record in genes.items():
        buckets.setdefault(record["chrom"], []).append((record["tss"], symbol, record))
    for chrom in buckets:
        buckets[chrom].sort()
    return buckets


def load_curated_sets(directory=None, genes=None):
    """``{set name: {symbols}}`` from ``gene_sets/``, restricted to genes in the mm10 table.

    Symbols that mm10 refGene does not carry are dropped and reported by
    ``curated_set_coverage``, so a set is never silently tested with members that could
    never be hit.
    """
    directory = Path(directory or CURATED_DIR)
    known = set(genes) if genes is not None else None
    sets = {}
    for path in sorted(directory.glob("*.txt")):
        members = []
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#"):
                members.append(line)
        name = path.stem
        sets[name] = {s for s in members if known is None or s in known}
    return sets


def curated_set_coverage(directory=None, genes=None):
    """``{set name: (kept, listed)}``: how much of each file the mm10 table could place."""
    directory = Path(directory or CURATED_DIR)
    known = set(genes) if genes is not None else None
    coverage = {}
    for path in sorted(directory.glob("*.txt")):
        members = [line.strip() for line in open(path)
                   if line.strip() and not line.startswith("#")]
        kept = [s for s in members if known is None or s in known]
        coverage[path.stem] = (len(kept), len(members))
    return coverage


def set_descriptions(directory=None):
    """The ``# name:`` line of each curated set file."""
    directory = Path(directory or CURATED_DIR)
    descriptions = {}
    for path in sorted(directory.glob("*.txt")):
        title = path.stem
        for line in open(path):
            if line.startswith("# name:"):
                title = line.split(":", 1)[1].strip()
                break
        descriptions[path.stem] = title
    return descriptions


def load_expression(path=None):
    """``({symbol: {deng stage: mean log2(RPKM + 1)}}, [stages])``."""
    path = Path(path or REFERENCE_DIR / "deng2014_stage_expression.tsv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run fetch_reference_data.py to build it."
        )
    with open(path) as handle:
        stages = handle.readline().rstrip("\n").split("\t")[1:]
        expression = {}
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            expression[fields[0]] = {
                stage: float(value) for stage, value in zip(stages, fields[1:])
            }
    return expression, stages


def stage_level(expression, symbol, stage):
    """Mean expression of one gene at one Hi-C stage, or ``None`` if it is not measured."""
    values = expression.get(symbol)
    if values is None:
        return None
    levels = [values[deng] for deng in STAGE_TO_DENG[stage] if deng in values]
    return sum(levels) / len(levels) if levels else None


def transition_change(expression, symbol, transition):
    """log2 fold change of one gene across one transition, or ``None``."""
    from_stage, to_stage = transition.split("->")
    before = stage_level(expression, symbol, from_stage)
    after = stage_level(expression, symbol, to_stage)
    if before is None or after is None:
        return None
    return after - before


def derive_transition_modules(expression, threshold=1.0, min_level=1.0):
    """``{"up@<transition>" / "down@<transition>": {symbols}}`` from the atlas.

    A gene is in the up set of a transition when its mean expression rises by at least
    ``threshold`` on the log2 scale, i.e. a two-fold change at the default, and reaches
    ``min_level`` at the higher of the two stages, which keeps genes that are near zero at
    both ends out of the set.
    """
    modules = {}
    for transition in TRANSITIONS:
        up, down = set(), set()
        for symbol in expression:
            change = transition_change(expression, symbol, transition)
            if change is None:
                continue
            from_stage, to_stage = transition.split("->")
            before = stage_level(expression, symbol, from_stage)
            after = stage_level(expression, symbol, to_stage)
            if change >= threshold and after >= min_level:
                up.add(symbol)
            elif change <= -threshold and before >= min_level:
                down.add(symbol)
        modules[f"up@{transition}"] = up
        modules[f"down@{transition}"] = down
    return modules


def load_ctcf_peaks(path=None):
    """``{chrom: [(start, end), ...]}`` sorted, or ``None`` when the file was not fetched."""
    path = Path(path or REFERENCE_DIR / "ctcf_mesc_mm10.bed")
    if not path.exists():
        return None
    peaks = {}
    for line in open(path):
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 3:
            continue
        peaks.setdefault(fields[0], []).append((int(fields[1]), int(fields[2])))
    for chrom in peaks:
        peaks[chrom].sort()
    return peaks


# The mESC chromatin panel fetch_reference_data.py --with-chromatin writes. The keys are the
# file stems under reference/; CTCF keeps its original name because it predates the panel.
CHROMATIN_MARKS = {
    "ctcf": ("ctcf_mesc_mm10.bed", "CTCF", "architectural"),
    "rad21": ("rad21_mm10.bed", "RAD21 cohesin (MEL)", "architectural"),
    "h3k4me3": ("h3k4me3_mm10.bed", "H3K4me3", "active"),
    "h3k4me1": ("h3k4me1_mm10.bed", "H3K4me1", "active"),
    "h3k27ac": ("h3k27ac_mm10.bed", "H3K27ac", "active"),
    "h3k9ac": ("h3k9ac_mm10.bed", "H3K9ac", "active"),
    "h3k36me3": ("h3k36me3_mm10.bed", "H3K36me3", "active"),
    "polr2a": ("polr2a_mm10.bed", "POLR2A", "active"),
    "ep300": ("ep300_mm10.bed", "EP300", "active"),
    "h3k27me3": ("h3k27me3_mm10.bed", "H3K27me3", "repressive"),
    "h3k9me3": ("h3k9me3_mm10.bed", "H3K9me3", "repressive"),
    "nanog": ("nanog_mm10.bed", "NANOG", "pluripotency"),
    "pou5f1": ("pou5f1_mm10.bed", "POU5F1", "pluripotency"),
    "cpg_islands": ("cpg_islands_mm10.bed", "CpG islands", "sequence"),
}


def load_mark(name, path=None):
    """``{chrom: [(start, end), ...]}`` for one panel track, or ``None`` if not fetched.

    Every track is mESC (RAD21 is MEL, see ``fetch_reference_data.CHROMATIN_TRACKS``) and
    the same at every embryo stage, so a track can say where a mark sits but can never
    produce a difference between stages on its own.
    """
    entry = CHROMATIN_MARKS.get(name)
    filename = entry[0] if entry else f"{name}_mm10.bed"
    path = Path(path or REFERENCE_DIR / filename)
    if not path.exists():
        return None
    intervals = {}
    for line in open(path):
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 3:
            continue
        try:
            intervals.setdefault(fields[0], []).append((int(fields[1]), int(fields[2])))
        except ValueError:
            continue
    for chrom in intervals:
        intervals[chrom].sort()
    return intervals


def available_marks():
    """The panel entries whose BED file is actually on disk, in panel order."""
    return [name for name in CHROMATIN_MARKS if load_mark(name) is not None]


def main():
    genes = load_gene_table()
    print(f"{len(genes)} genes in the mm10 table")
    print()
    print("curated sets (placed / listed):")
    coverage = curated_set_coverage(genes=genes)
    titles = set_descriptions()
    for name, (kept, listed) in coverage.items():
        print(f"  {name:38} {kept:4}/{listed:<4}  {titles[name]}")
    print()
    expression, stages = load_expression()
    print(f"expression: {len(expression)} genes over {len(stages)} Deng stages")
    modules = derive_transition_modules(expression)
    print("derived transition modules (two-fold, expressed):")
    for name in sorted(modules, key=lambda n: (n.split("@")[1], n)):
        print(f"  {name:28} {len(modules[name]):5} genes")


if __name__ == "__main__":
    main()
