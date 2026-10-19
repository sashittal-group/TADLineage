#!/usr/bin/env python3
"""Download the mouse-native MSigDB collections used for the haematopoietic pathway analysis.

MSigDB publishes mouse (``Mm``) releases whose genes are MGI symbols, so the sets join
straight onto the mm10 gene table with no human-to-mouse ortholog mapping and no losses
from that step.  Four collections:

    mh.all          the 50 mouse hallmark sets, the collection most papers lead with
    m2.cp.reactome  Reactome canonical pathways
    m5.go.bp        GO biological process
    m8.all          mouse cell-type signature sets

``m8.all`` is here because this is a lineage: it carries signatures of sorted marrow
populations that map almost node for node onto the tree, so an edge can be asked whether
the structure that reorganises along it carries the signature of the cell type it leads to.

Usage:
    python fetch_gene_sets.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "gene_sets"
SHARED = HERE.parent.parent / "scptr_robustad" / "tad_pathway_analysis" / "gene_sets"
RELEASE = "2024.1.Mm"
BASE = f"https://data.broadinstitute.org/gsea-msigdb/msigdb/release/{RELEASE}"

COLLECTIONS = ("mh.all", "m2.cp.reactome", "m5.go.bp", "m8.all")


def fetch(name):
    """Get one collection's symbol GMT into ``gene_sets/``, reusing the single-cell copy."""
    fname = f"{name}.v{RELEASE}.symbols.gmt"
    dest = OUT_DIR / fname
    if dest.exists():
        print(f"{fname}: already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    shared = SHARED / fname
    if shared.exists():
        dest.write_bytes(shared.read_bytes())
        print(f"{fname}: copied from the single-cell folder "
              f"({sum(1 for _ in dest.open())} gene sets)")
        return dest

    print(f"{fname}: downloading")
    try:
        urllib.request.urlretrieve(f"{BASE}/{fname}", dest)
    except Exception as exc:  # noqa: BLE001 - report and continue with the others
        print(f"{fname}: FAILED ({exc})", file=sys.stderr)
        dest.unlink(missing_ok=True)
        return None
    print(f"{fname}: {sum(1 for _ in dest.open())} gene sets")
    return dest


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return 0 if all(fetch(name) for name in COLLECTIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
