#!/usr/bin/env python3
"""Download the public reference data the biological annotation needs.

Three references, each written once into ``reference/`` and skipped when already there:

``mm10_genes.tsv``
    RefSeq genes from the UCSC ``mm10.refGene`` table, collapsed to one row per gene
    symbol: the span of all its transcripts and the transcription start site of the most
    5' one. mm10 is the assembly of the RobusTAD calls this folder analyses.

``deng2014_stage_expression.tsv``
    Mean RPKM per gene for each stage of Deng et al. 2014 (Science 343:193, GEO
    GSE45719), the single-cell RNA-seq atlas of mouse preimplantation development. Its
    stages line up with the Hi-C stages of Collombet et al. 2020 (GSE129029) that the
    clustering runs on, which is what lets a domain rearrangement be read against the
    transcription happening at the same transition. Cells are averaged per stage after
    a log transform, and the per-stage cell counts are written alongside.

``ctcf_mesc_mm10.bed`` (optional, ``--with-ctcf``)
    CTCF ChIP-seq peaks in mouse embryonic stem cells from ENCODE, used to ask whether the
    boundaries that appear or disappear sit on CTCF sites. Skipped quietly when the
    download fails, since the rest of the analysis does not depend on it.

Usage:
    python fetch_reference_data.py
    python fetch_reference_data.py --with-ctcf --force
"""

from __future__ import annotations

import argparse
import gzip
import io
import re
import shutil
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE_DIR = HERE / "reference"
CACHE_DIR = REFERENCE_DIR / "downloads"

REFGENE_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/database/refGene.txt.gz"
DENG_URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE45nnn/GSE45719/suppl/"
            "GSE45719_RAW.tar")
# ENCODE ENCSR179IQN, CTCF ChIP-seq in C57BL/6J mouse embryonic stem cells, mm10
CTCF_URL = "https://www.encodeproject.org/files/ENCFF625RVT/@@download/ENCFF625RVT.bed.gz"

PRIMARY_CHROMS = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY"]

# the Deng stages, in developmental order; the ones the Hi-C stages map onto are chosen in
# expression_analysis.py, so every stage is kept here
DENG_STAGES = [
    "zy", "early2cell", "mid2cell", "late2cell", "4cell", "8cell", "16cell",
    "earlyblast", "midblast", "lateblast",
]
# technical variants and the non-embryo controls of the series
DENG_EXCLUDE = ("liver", "fibroblast", "split", "pooled", "smartseq", "_0r", "C57twocell")


# Chromatin marks in mouse embryonic stem cells, mm10, ENCODE replicated or IDR peak calls.
# The panel exists to separate three readings of a boundary signal: architectural protein
# (CTCF, and cohesin as far as it can be had), active chromatin (H3K4me3, H3K4me1, H3K27ac,
# H3K9ac, H3K36me3, POLR2A, EP300) and repressive chromatin (H3K27me3, H3K9me3), with the
# pluripotency factors as a fourth, cell-identity, group.
#
# ALL OF THESE ARE mESC, NOT EMBRYO, exactly like the CTCF track the folder already uses.
# They say where a mark sits in a related cell type, not where it sits in a 4-cell embryo,
# so they can support "boundaries align to CTCF-marked positions" and cannot support any
# claim about a mark changing over the stage series -- the tracks are constant across
# stages by construction, which is also why they cannot manufacture a stage trend.
# RAD21 IS THE ONE EXCEPTION AND IS LABELLED: ENCODE has no cohesin ChIP in mouse ES cells,
# so it comes from MEL (erythroleukemia). CTCF-anchored cohesin sites are largely shared
# between cell types, but read that row with more caution than the others.
CHROMATIN_TRACKS = {
    "h3k4me3": ("ENCFF012AZO", "H3K4me3, ES-E14"),
    "h3k4me1": ("ENCFF440FYE", "H3K4me1, ES-E14"),
    "h3k27ac": ("ENCFF194TQD", "H3K27ac, ES-E14"),
    "h3k9ac": ("ENCFF487JDK", "H3K9ac, ES-E14"),
    "h3k27me3": ("ENCFF988MSF", "H3K27me3, E14TG2a.4"),
    "h3k9me3": ("ENCFF090UKL", "H3K9me3, ES-E14"),
    "h3k36me3": ("ENCFF759WKW", "H3K36me3, E14TG2a.4"),
    "polr2a": ("ENCFF041IHJ", "POLR2A, ES-Bruce4"),
    "ep300": ("ENCFF179FJG", "EP300, ES-Bruce4"),
    "nanog": ("ENCFF495GBW", "NANOG, E14TG2a.4"),
    "pou5f1": ("ENCFF609NWN", "POU5F1/Oct4, E14TG2a.4"),
    "rad21": ("ENCFF791EKG", "RAD21 cohesin, MEL -- NOT an ES cell line"),
}
ENCODE_FILE_URL = "https://www.encodeproject.org/files/{0}/@@download/{0}.bed.gz"
CPG_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/database/cpgIslandExt.txt.gz"


def download(url, destination, description):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"have {description}: {destination}")
        return destination
    print(f"downloading {description} from {url}")
    with urllib.request.urlopen(url, timeout=300) as response, \
            open(destination, "wb") as handle:
        shutil.copyfileobj(response, handle)
    print(f"  wrote {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
    return destination


def build_gene_table(force=False):
    """One row per gene symbol: chromosome, span, TSS, strand, transcript count."""
    out_path = REFERENCE_DIR / "mm10_genes.tsv"
    if out_path.exists() and not force:
        print(f"have gene table: {out_path}")
        return out_path

    archive = download(REFGENE_URL, CACHE_DIR / "mm10_refGene.txt.gz", "mm10 refGene")

    genes = {}
    with gzip.open(archive, "rt") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            # bin name chrom strand txStart txEnd cdsStart cdsEnd exonCount ... name2
            if len(fields) < 13:
                continue
            chrom, strand = fields[2], fields[3]
            if chrom not in PRIMARY_CHROMS:
                continue
            symbol = fields[12]
            start, end = int(fields[4]), int(fields[5])
            tss = start if strand == "+" else end

            record = genes.get(symbol)
            if record is None:
                genes[symbol] = {"chrom": chrom, "start": start, "end": end,
                                 "strand": strand, "tss": tss, "transcripts": 1}
                continue
            if record["chrom"] != chrom:
                # a symbol placed on more than one chromosome: keep the first placement,
                # which for refGene is the primary one, and do not let a stray alternative
                # stretch the span across chromosomes
                continue
            record["start"] = min(record["start"], start)
            record["end"] = max(record["end"], end)
            record["transcripts"] += 1
            # the most 5' start on the strand of the gene
            if strand == "+":
                record["tss"] = min(record["tss"], tss)
            else:
                record["tss"] = max(record["tss"], tss)

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        handle.write("symbol\tchrom\tstart\tend\tstrand\ttss\ttranscripts\n")
        for symbol in sorted(genes, key=lambda s: (genes[s]["chrom"], genes[s]["start"])):
            record = genes[symbol]
            handle.write(f"{symbol}\t{record['chrom']}\t{record['start']}\t{record['end']}\t"
                         f"{record['strand']}\t{record['tss']}\t{record['transcripts']}\n")
    print(f"wrote {out_path}: {len(genes)} gene symbols")
    return out_path


def deng_stage_of(name):
    """The developmental stage a GSE45719 file name belongs to, or ``None`` to skip it."""
    if any(token in name for token in DENG_EXCLUDE):
        return None
    trimmed = re.sub(r"^GSM\d+_", "", name)
    for stage in sorted(DENG_STAGES, key=len, reverse=True):
        if trimmed.startswith(stage):
            return stage
    return None


def build_expression_table(force=False):
    """Mean expression per gene per Deng stage, from the per-cell RPKM files."""
    out_path = REFERENCE_DIR / "deng2014_stage_expression.tsv"
    if out_path.exists() and not force:
        print(f"have expression table: {out_path}")
        return out_path

    archive = download(DENG_URL, CACHE_DIR / "GSE45719_RAW.tar",
                       "Deng et al. 2014 single-cell RNA-seq (GSE45719)")

    import math

    sums = defaultdict(lambda: defaultdict(float))
    cells_per_stage = defaultdict(int)
    genes = set()

    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.isfile() or "expression" not in member.name:
                continue
            stage = deng_stage_of(Path(member.name).name)
            if stage is None:
                continue
            payload = tar.extractfile(member)
            if payload is None:
                continue
            with gzip.open(io.BytesIO(payload.read()), "rt") as handle:
                header = handle.readline()
                if "RPKM" not in header:
                    continue
                columns = header.lstrip("#").rstrip("\n").split("\t")
                rpkm_column = columns.index("RPKM")
                for line in handle:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) <= rpkm_column:
                        continue
                    symbol = fields[0]
                    try:
                        value = float(fields[rpkm_column])
                    except ValueError:
                        continue
                    # averaged on a log scale, so one very high cell cannot carry a stage
                    sums[symbol][stage] += math.log2(value + 1.0)
                    genes.add(symbol)
            cells_per_stage[stage] += 1

    if not cells_per_stage:
        raise RuntimeError(f"No per-cell expression files were read out of {archive}")

    stages = [stage for stage in DENG_STAGES if cells_per_stage.get(stage)]
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        handle.write("symbol\t" + "\t".join(stages) + "\n")
        for symbol in sorted(genes):
            row = [f"{sums[symbol][stage] / cells_per_stage[stage]:.4f}" for stage in stages]
            handle.write(symbol + "\t" + "\t".join(row) + "\n")

    counts_path = REFERENCE_DIR / "deng2014_stage_cell_counts.tsv"
    with open(counts_path, "w") as handle:
        handle.write("stage\tcells\n")
        for stage in stages:
            handle.write(f"{stage}\t{cells_per_stage[stage]}\n")

    print(f"wrote {out_path}: {len(genes)} genes over {len(stages)} stages "
          f"({sum(cells_per_stage.values())} cells)")
    print(f"wrote {counts_path}")
    return out_path


def build_ctcf_peaks(force=False):
    """CTCF ChIP-seq peaks in mouse ES cells, as a plain BED of peak intervals."""
    out_path = REFERENCE_DIR / "ctcf_mesc_mm10.bed"
    if out_path.exists() and not force:
        print(f"have CTCF peaks: {out_path}")
        return out_path
    try:
        archive = download(CTCF_URL, CACHE_DIR / "ctcf_mesc_mm10.bed.gz",
                           "ENCODE CTCF ChIP-seq peaks (mESC, mm10)")
    except Exception as error:
        print(f"could not fetch CTCF peaks ({error}); the CTCF step will be skipped")
        return None

    kept = 0
    with gzip.open(archive, "rt") as source, open(out_path, "w") as handle:
        for line in source:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3 or fields[0] not in PRIMARY_CHROMS:
                continue
            handle.write("\t".join(fields[:3]) + "\n")
            kept += 1
    print(f"wrote {out_path}: {kept} peaks")
    return out_path


def build_chromatin_tracks(force=False):
    """One plain BED of peak intervals per mark in ``CHROMATIN_TRACKS``.

    A mark whose download fails is skipped rather than fatal: every analysis that uses these
    treats a missing track as one fewer row, so a partial panel still runs.
    """
    written = {}
    for name, (accession, description) in CHROMATIN_TRACKS.items():
        out_path = REFERENCE_DIR / f"{name}_mm10.bed"
        if out_path.exists() and not force:
            print(f"have {name}: {out_path}")
            written[name] = out_path
            continue
        try:
            archive = download(ENCODE_FILE_URL.format(accession),
                               CACHE_DIR / f"{name}_mm10.bed.gz",
                               f"ENCODE {description} [{accession}]")
        except Exception as error:
            print(f"could not fetch {name} ({error}); it will be skipped")
            continue
        kept = 0
        with gzip.open(archive, "rt") as source, open(out_path, "w") as handle:
            for line in source:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 3 or fields[0] not in PRIMARY_CHROMS:
                    continue
                handle.write("\t".join(fields[:3]) + "\n")
                kept += 1
        print(f"wrote {out_path}: {kept} peaks")
        written[name] = out_path
    return written


def build_cpg_islands(force=False):
    """UCSC CpG islands as a plain BED; a sequence feature, not an assay of any cell type."""
    out_path = REFERENCE_DIR / "cpg_islands_mm10.bed"
    if out_path.exists() and not force:
        print(f"have CpG islands: {out_path}")
        return out_path
    try:
        archive = download(CPG_URL, CACHE_DIR / "cpgIslandExt.txt.gz",
                           "UCSC mm10 CpG islands")
    except Exception as error:
        print(f"could not fetch CpG islands ({error}); it will be skipped")
        return None
    kept = 0
    with gzip.open(archive, "rt") as source, open(out_path, "w") as handle:
        for line in source:
            fields = line.rstrip("\n").split("\t")
            # cpgIslandExt carries a bin column first, so the interval is columns 1-3
            if len(fields) < 4 or fields[1] not in PRIMARY_CHROMS:
                continue
            handle.write("\t".join(fields[1:4]) + "\n")
            kept += 1
    print(f"wrote {out_path}: {kept} islands")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="rebuild tables already present")
    parser.add_argument("--with-ctcf", action="store_true",
                        help="also fetch ENCODE CTCF peaks for the boundary analysis")
    parser.add_argument("--skip-expression", action="store_true",
                        help="skip the 98 MB GSE45719 download")
    parser.add_argument("--with-chromatin", action="store_true",
                        help="also fetch the mESC chromatin mark panel and CpG islands")
    args = parser.parse_args()

    build_gene_table(force=args.force)
    if not args.skip_expression:
        build_expression_table(force=args.force)
    if args.with_ctcf:
        build_ctcf_peaks(force=args.force)
    if args.with_chromatin:
        build_chromatin_tracks(force=args.force)
        build_cpg_islands(force=args.force)


if __name__ == "__main__":
    main()
