#!/usr/bin/env python3
"""Run RobusTAD on the shared simulated matrices to produce called hierarchies.

This is the RobusTAD analogue of ``using_ontad/run_ontad.py``. RobusTAD reads a
cooler, so each dense simulated matrix is first materialized as a single-
chromosome ``.cool`` (unit balancing weights, since the synthetic matrices need
no ICE correction). We then run RobusTAD's single-sample ``boundary`` caller and
the ``assembly`` step, and convert both outputs to bin-indexed ``*_with_bins.tsv``
tables that mirror the real-data RobusTAD layout consumed by the hemato work.

Per region/node it writes, into the shared data tree at
``<outdir>/region_XXX/robustad/``:

  <node>_boundary.bed                 raw RobusTAD boundary scores (left/right)
  <node>_boundary_with_bins.tsv       + integer bin column
  <node>_tads_all.bed / _tads_hq.bed  raw RobusTAD assembled TADs
  <node>_tads_all_with_bins.tsv       + start_bin / end_bin columns
  <node>_tads_hq_with_bins.tsv

These are the observed TAD hierarchies used by the experiment:
  * internal nodes (A, B, C)      -> evaluation ground truth
  * root + leaves (D, E, F, G)    -> reconstruction inputs

RobusTAD must run under an interpreter that has ``cooler`` + RobusTAD's deps
(e.g. the ``refhic`` or ``hic_analysis`` conda env); the RobusTAD package itself
is imported from ``<repo>/RobusTAD`` via PYTHONPATH.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import cooler

SCRIPT_DIR = Path(__file__).resolve().parent
SIM_ROOT = SCRIPT_DIR.parent          # simulation/ (shared config.py + data/)
REPO_ROOT = SIM_ROOT.parent           # repo root (holds RobusTAD/)
sys.path.insert(0, str(SIM_ROOT))     # shared config.py

import config

ROBUSTAD_ROOT = REPO_ROOT / "RobusTAD"
CHROM = "chr1"                         # synthetic single-chromosome label


# ---------------------------------------------------------------------------
# dense matrix -> cooler
# ---------------------------------------------------------------------------
def matrix_to_cooler(matrix_file: Path, cool_file: Path, resol: int) -> None:
    """Write a dense simulated matrix as a single-chromosome balanced cooler.

    The synthetic counts need no ICE correction, so a constant ``weight`` of 1
    is stored; RobusTAD reads with ``balance=True`` and thus gets the raw
    (unmodified) matrix back.
    """
    mat = np.loadtxt(matrix_file)
    n = mat.shape[0]
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Matrix {matrix_file} is not square: {mat.shape}")

    bins = pd.DataFrame(
        {
            "chrom": [CHROM] * n,
            "start": np.arange(n) * resol,
            "end": (np.arange(n) + 1) * resol,
            "weight": 1.0,
        }
    )
    iu = np.triu_indices(n)
    counts = mat[iu]
    nz = counts > 0
    pixels = pd.DataFrame(
        {"bin1_id": iu[0][nz], "bin2_id": iu[1][nz], "count": counts[nz]}
    )
    cool_file.parent.mkdir(parents=True, exist_ok=True)
    cooler.create_cooler(
        str(cool_file),
        bins=bins,
        pixels=pixels,
        dtypes={"count": "float64"},
        symmetric_upper=True,
        ordered=True,
    )


# ---------------------------------------------------------------------------
# RobusTAD subprocess helpers
# ---------------------------------------------------------------------------
def robustad_env(mplconfigdir: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{ROBUSTAD_ROOT}{os.pathsep}{existing}" if existing else str(ROBUSTAD_ROOT)
    )
    mplconfigdir.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mplconfigdir)
    return env


def robustad_cmd() -> list[str]:
    return [sys.executable, "-m", "robustad.robustad"]


def run(cmd: list[str], env: dict[str, str], timeout: int) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, timeout=timeout)


def run_boundary(cool_file: Path, prefix: Path, resol: int,
                 minw_kb: int, maxw_kb: int, ratio: float, alpha: float,
                 env: dict[str, str], timeout: int) -> Path:
    run(
        robustad_cmd()
        + [
            "boundary",
            "--resol", str(resol),
            "--chr", CHROM,
            "--minW", str(minw_kb),
            "--maxW", str(maxw_kb),
            "--ratio", str(ratio),
            "--alpha", str(alpha),
            str(cool_file),
            str(prefix),
        ],
        env,
        timeout,
    )
    return Path(str(prefix) + ".bed")


def run_assembly(cool_file: Path, boundary_bed: Path, prefix: Path, resol: int,
                 mintad_bp: int, maxtad_bp: int, ratio: float, delta: float,
                 env: dict[str, str], timeout: int) -> None:
    run(
        robustad_cmd()
        + [
            "assembly",
            "--resol", str(resol),
            "--chr", CHROM,
            "--mintad", str(mintad_bp),
            "--maxtad", str(maxtad_bp),
            "--ratio", str(ratio),
            "--delta", str(delta),
            str(cool_file),
            str(boundary_bed),
            str(prefix),
        ],
        env,
        timeout,
    )


# ---------------------------------------------------------------------------
# raw RobusTAD .bed -> bin-indexed *_with_bins.tsv
# ---------------------------------------------------------------------------
def convert_boundary(path: Path, resol: int) -> Path:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "leftScore", "leftBoundary",
               "rightScore", "rightBoundary"],
    )
    for col in ["start", "end", "leftScore", "leftBoundary", "rightScore", "rightBoundary"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["start", "end"]).copy()
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["bin"] = (df["start"] // resol).astype(int)

    out_path = path.with_name(path.stem + "_with_bins.tsv")
    df[["chrom", "start", "end", "bin", "leftScore", "leftBoundary",
        "rightScore", "rightBoundary"]].to_csv(out_path, sep="\t", index=False)
    return out_path


TAD_COLUMNS = ["chrom", "start", "end", "score", "nestedScore", "Hq", "level"]


def convert_tads(path: Path, resol: int) -> Path:
    # RobusTAD's assembly writes a header row for the "all" table but not for
    # the "hq" one, so detect it from the first field rather than assuming.
    with path.open() as fh:
        first = fh.readline().split("\t")
    has_header = first and first[0].strip().lower() in {"chrom", "chr"}
    if has_header:
        df = pd.read_csv(path, sep="\t")
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={"startbp": "start", "endbp": "end"})
    else:
        df = pd.read_csv(path, sep="\t", header=None, names=TAD_COLUMNS)
    for col in ["start", "end", "score", "nestedScore", "Hq", "level"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["start", "end"]).copy()
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["start_bin"] = (df["start"] // resol).astype(int)
    df["end_bin"] = ((df["end"] - 1) // resol).astype(int)

    out_path = path.with_name(path.stem + "_with_bins.tsv")
    cols = ["chrom", "start", "end", "start_bin", "end_bin",
            "score", "nestedScore", "Hq", "level"]
    df[[c for c in cols if c in df.columns]].to_csv(out_path, sep="\t", index=False)
    return out_path


# ---------------------------------------------------------------------------
# per-node driver
# ---------------------------------------------------------------------------
def process_node(region_dir: Path, node: str, resol: int, params: dict,
                 env: dict[str, str], keep_cooler: bool) -> None:
    matrix_file = region_dir / "matrices" / f"{node}.matrix"
    if not matrix_file.exists():
        raise SystemExit(f"Matrix not found: {matrix_file}")

    out_dir = region_dir / "robustad"
    out_dir.mkdir(exist_ok=True)
    cool_file = out_dir / f"{node}.cool"
    boundary_prefix = out_dir / f"{node}_boundary"
    tad_prefix = out_dir / f"{node}_tads"

    matrix_to_cooler(matrix_file, cool_file, resol)
    try:
        boundary_bed = run_boundary(
            cool_file, boundary_prefix, resol,
            params["minW_kb"], params["maxW_kb"], params["ratio"], params["alpha"],
            env, params["timeout"],
        )
        run_assembly(
            cool_file, boundary_bed, tad_prefix, resol,
            params["mintad_bp"], params["maxtad_bp"], params["ratio"], params["delta"],
            env, params["timeout"],
        )
    finally:
        if not keep_cooler:
            cool_file.unlink(missing_ok=True)

    convert_boundary(boundary_bed, resol)
    for kind in ("all", "hq"):
        bed = out_dir / f"{node}_tads_{kind}.bed"
        if bed.exists():
            convert_tads(bed, resol)
        else:
            raise SystemExit(f"RobusTAD did not write expected TAD output: {bed}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=SIM_ROOT / "data")
    p.add_argument("--region-index", type=str, default=None,
                   help="run exactly this region, e.g. 7, 007, or region_007")
    p.add_argument("--node", choices=config.NODE_ORDER, default=None,
                   help="run exactly this node within the selected region(s)")
    p.add_argument("--resolution", type=int, default=config.DEFAULT_PARAMS.resolution_bp,
                   help="bin size in bp for the cooler (default: config resolution_bp)")
    p.add_argument("--minW", type=int, default=80, help="RobusTAD min window in kb")
    p.add_argument("--maxW", type=int, default=2000, help="RobusTAD max window in kb")
    p.add_argument("--ratio", type=float, default=1.2)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--mintad", type=int, default=None,
                   help="min TAD size in bp [default: min_size bins * resolution]")
    p.add_argument("--maxtad", type=int, default=None,
                   help="max TAD size in bp [default: 200 bins * resolution]")
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--keep-cooler", action="store_true",
                   help="keep the intermediate .cool (default: delete to save space)")
    args = p.parse_args()

    resol = args.resolution
    mintad = args.mintad if args.mintad is not None else config.DEFAULT_PARAMS.min_size * resol
    maxtad = args.maxtad if args.maxtad is not None else 200 * resol
    params = {
        "minW_kb": args.minW,
        "maxW_kb": args.maxW,
        "ratio": args.ratio,
        "alpha": args.alpha,
        "mintad_bp": mintad,
        "maxtad_bp": maxtad,
        "delta": args.delta,
        "timeout": args.timeout,
    }

    if args.region_index is None:
        region_dirs = sorted(args.outdir.glob("region_*"))
    else:
        idx = args.region_index
        region_name = idx if idx.startswith("region_") else f"region_{int(idx):03d}"
        region_dirs = [args.outdir / region_name]

    if not region_dirs:
        raise SystemExit(f"No region_* directories under {args.outdir}; generate matrices first")
    missing = [str(rd) for rd in region_dirs if not rd.exists()]
    if missing:
        raise SystemExit(f"Region directory not found: {missing[0]}")

    nodes = [args.node] if args.node else config.NODE_ORDER
    env = robustad_env(SCRIPT_DIR / ".mplcache")

    for region_dir in region_dirs:
        for node in nodes:
            process_node(region_dir, node, resol, params, env, args.keep_cooler)
        print(f"{region_dir.name}: RobusTAD done", flush=True)


if __name__ == "__main__":
    main()
