#!/usr/bin/env python3
"""Bin indices of a clustering run back to mm10 base pairs.

The clustering represents a cell by TAD intervals over bin indices, and two different
conventions produced those indices, so a genomic coordinate can only be recovered if the
convention is known.

Per-chromosome runs
    ``convert_scptr_robustad_to_ontad_chr_layout.py`` writes 1-based bins per chromosome:
    ``start_bin = startbp // res + 1`` and ``end_bin = (endbp - 1) // res + 1``. Bin ``i``
    therefore covers ``[(i - 1) * res, i * res)`` and a TAD ``[a, b]`` covers
    ``[(a - 1) * res, b * res)``.

Genome-wide ("all") runs
    ``em_cluster_updated_all_chr.py`` lays the chromosomes end to end with
    ``encode_position = offset[chrom] + bp // res``, so the bins are 0-based within a
    chromosome and a TAD ``[a, b]`` covers ``[(a - off) * res, (b - off + 1) * res)``.

    Those offsets are built from ``genome.chrom.sizes`` at the repository root, which is
    an **mm9** file while the data are mm10 (the RobusTAD manifests say
    ``full_cell_hic_matrix_all_mm10_chromosomes``, and TADs on chrX reach 171 Mb, past the
    end of mm9 chrX). mm10 is longer than mm9 on chr2, 3, 4, 6, 9, 10, 11, 13, 15, 19 and
    X, so a TAD at the far end of one of those chromosomes can be encoded into the bin
    range of the next one. ``decode_genome_wide`` reports how far past the recorded end of
    a chromosome a bin falls so that such cases can be dropped; the analyses in this folder
    are run per chromosome, where no offset is involved and the mapping is exact.

Coordinates are bin-resolution, so an event position is only ever known to within one bin
(250 kb, 500 kb or 1 Mb). Everything downstream keeps that uncertainty explicit by working
with intervals rather than points.
"""

from __future__ import annotations

import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# the order em_cluster_updated_all_chr.py lays the chromosomes out in
CHROMS = [f"chr{i}" for i in range(1, 20)] + ["chrX"]

# mm10 (GRCm38) primary chromosome lengths, used for clipping annotations to the assembly
MM10_SIZES = {
    "chr1": 195471971, "chr2": 182113224, "chr3": 160039680, "chr4": 156508116,
    "chr5": 151834684, "chr6": 149736546, "chr7": 145441459, "chr8": 129401213,
    "chr9": 124595110, "chr10": 130694993, "chr11": 122082543, "chr12": 120129022,
    "chr13": 120421639, "chr14": 124902244, "chr15": 104043685, "chr16": 98207768,
    "chr17": 94987271, "chr18": 90702639, "chr19": 61431566, "chrX": 171031299,
    "chrY": 91744698, "chrM": 16299,
}


def resolution_to_bp(resolution):
    """'500kb' -> 500000, '1000kb' -> 1000000, 500000 -> 500000."""
    label = str(resolution).lower().strip()
    if label.endswith("kb"):
        return int(label[:-2]) * 1000
    if label.endswith("mb"):
        return int(float(label[:-2]) * 1_000_000)
    return int(label)


def load_chrom_offsets(genome_sizes_path=None, resolution_bp=500000):
    """The same offsets em_cluster_updated_all_chr.py used to encode the genome-wide runs.

    Returns ``{chrom: (offset_bin, bins_recorded)}`` where ``bins_recorded`` is how many
    bins that chromosome was given, i.e. ``ceil(size / res) + 1``.
    """
    path = Path(genome_sizes_path or REPO_ROOT / "genome.chrom.sizes")
    sizes = {}
    for line in open(path):
        parts = line.split()
        if len(parts) >= 2:
            sizes[parts[0]] = int(parts[1])

    offsets, cursor = {}, 0
    for chrom in CHROMS:
        if chrom not in sizes:
            raise KeyError(f"{chrom} is missing from {path}")
        span = int(math.ceil(sizes[chrom] / resolution_bp)) + 1
        offsets[chrom] = (cursor, span)
        cursor += span
    return offsets


def interval_to_bp(interval, resolution_bp):
    """A per-chromosome TAD ``(a, b)`` in 1-based bins to a half-open ``(start, end)`` in bp."""
    a, b = interval
    return (a - 1) * resolution_bp, b * resolution_bp


def bin_to_bp(bin_index, resolution_bp):
    """The half-open bp span of one 1-based per-chromosome bin."""
    return (bin_index - 1) * resolution_bp, bin_index * resolution_bp


def decode_genome_wide(bin_index, offsets, resolution_bp):
    """A global bin of an "all" run to ``(chrom, start_bp, end_bp, overflow_bins)``.

    ``overflow_bins`` is how far past the bin span the chromosome was allotted the index
    falls; anything above zero sits in a stretch that mm9 sizes did not reserve, so the
    chromosome assignment there cannot be trusted.
    """
    for chrom in CHROMS:
        offset, span = offsets[chrom]
        if offset <= bin_index < offset + span:
            local = bin_index - offset
            return chrom, local * resolution_bp, (local + 1) * resolution_bp, 0
    # past the end of the last chromosome
    chrom = CHROMS[-1]
    offset, span = offsets[chrom]
    local = bin_index - offset
    return chrom, local * resolution_bp, (local + 1) * resolution_bp, local - span + 1


def clip_to_chromosome(chrom, start, end):
    """Keep an interval inside the mm10 assembly; returns ``None`` if it falls off the end."""
    size = MM10_SIZES.get(chrom)
    if size is None:
        return int(start), int(end)
    start, end = max(0, int(start)), min(int(size), int(end))
    return (start, end) if end > start else None
