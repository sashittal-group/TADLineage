#!/usr/bin/env python3
"""Download the mouse-native MSigDB collections used for the pathway analysis.

MSigDB publishes mouse (``Mm``) releases whose genes are MGI symbols, so the gene sets
join straight onto the mm10 gene table with no human-to-mouse ortholog mapping and no
losses from that step.  Three collections are fetched:

    mh.all          the 50 mouse hallmark sets, the collection most papers lead with
    m2.cp.reactome  Reactome canonical pathways
    m5.go.bp        GO biological process

Usage:
    python fetch_gene_sets.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "gene_sets"
RELEASE = "2024.1.Mm"
BASE = f"https://data.broadinstitute.org/gsea-msigdb/msigdb/release/{RELEASE}"

COLLECTIONS = ("mh.all", "m2.cp.reactome", "m5.go.bp")


def fetch(name):
    """Download one collection's symbol GMT into ``gene_sets/`` and report its size."""
    fname = f"{name}.v{RELEASE}.symbols.gmt"
    dest = OUT_DIR / fname
    if dest.exists():
        print(f"{fname}: already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    url = f"{BASE}/{fname}"
    print(f"{fname}: downloading")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:  # noqa: BLE001 - report and continue with the others
        print(f"{fname}: FAILED ({exc})", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        return None
    n = sum(1 for _ in dest.open())
    print(f"{fname}: {n} gene sets")
    return dest


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = [fetch(name) for name in COLLECTIONS]
    if not all(ok):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
