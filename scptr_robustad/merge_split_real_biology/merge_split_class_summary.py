#!/usr/bin/env python3
"""The aggregate that survives when no single locus does.

``merge_split_locus_validation.py`` scores every OnTAD merge and split bin against held-out cells
and finds that none of them clears FDR individually. This asks the class-level version of
the same question, which is a different and much better powered test: taken together, do
the bins OnTAD calls *split* gain penetrance in held-out cells, and do the bins it calls
*merge* lose it?

The replicate is the chromosome, as everywhere else in this folder, and the test is the
exact two-sided sign test on the per-chromosome mean of the referenced delta. The
transition-by-class breakdown is reported alongside because eight rows all pointing the
predicted way is a stronger statement than either pooled number.

Usage:
    python merge_split_class_summary.py --caller RobusTAD
"""

from __future__ import annotations

import argparse
import collections
import csv
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LOCI = HERE / "results" / "merge_split"

EXPECTED = {"split": +1, "merge": -1}
TRANSITIONS = ["1-cell->2-cell", "2-cell->4-cell", "4-cell->8-cell", "8-cell->64-cell"]
# a class needs this many loci on a chromosome before that chromosome is a replicate
MIN_LOCI = 3

FIELDS = ["resolution", "transition", "event_type", "expected", "loci", "chromosomes",
          "agreeing", "mean_delta", "mean_referenced_delta", "sign_p"]


def sign_test(values):
    values = [v for v in values if v == v and v != 0.0]
    n = len(values)
    if n == 0:
        return 1.0
    positive = sum(1 for v in values if v > 0)
    k = min(positive, n - positive)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def summarise(rows, resolution, event_type, transition=None):
    picked = [r for r in rows
              if r["resolution"] == resolution and r["event_type"] == event_type
              and (transition is None or r["transition"] == transition)]
    if len(picked) < MIN_LOCI:
        return None
    by_chrom = collections.defaultdict(list)
    for row in picked:
        by_chrom[row["chrom"]].append(float(row["referenced_delta"]))
    means = [float(np.mean(v)) for v in by_chrom.values() if len(v) >= 1]
    if len(means) < 5:
        return None
    direction = EXPECTED[event_type]
    return {
        "resolution": resolution,
        "transition": transition or "all",
        "event_type": event_type,
        "expected": "up" if direction > 0 else "down",
        "loci": len(picked),
        "chromosomes": len(means),
        "agreeing": sum(1 for m in means if m * direction > 0),
        "mean_delta": round(float(np.mean([float(r["delta"]) for r in picked])), 5),
        "mean_referenced_delta": round(float(np.mean(means)), 5),
        "sign_p": round(sign_test([m * direction for m in means]), 5),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--caller", default="OnTAD")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    loci_dir = LOCI / args.caller
    out_dir = Path(args.out_dir or loci_dir)

    with open(loci_dir / "locus_validation.tsv") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    out = []
    for resolution in sorted({r["resolution"] for r in rows}):
        for event_type in ("split", "merge"):
            record = summarise(rows, resolution, event_type)
            if record:
                out.append(record)
            for transition in TRANSITIONS:
                record = summarise(rows, resolution, event_type, transition)
                if record:
                    out.append(record)

    path = out_dir / "class_summary.tsv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(out)

    for record in out:
        if record["transition"] == "all":
            print(f"{record['resolution']:7s} {record['event_type']:6s} "
                  f"{record['loci']:4d} loci  mean {record['mean_referenced_delta']:+.4f}  "
                  f"{record['agreeing']:2d}/{record['chromosomes']:<2d} chromosomes  "
                  f"p = {record['sign_p']:.5f}")
    print(f"wrote {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
