#!/usr/bin/env python3
"""Run RobusTAD on CMP and MPP chr4 at 250 kb without a reference panel."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import median

import cooler
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBUSTAD_ROOT = REPO_ROOT / "RobusTAD"
HEMATO_ROOT = REPO_ROOT / "hemato_cells"
MCOOL_ROOT = REPO_ROOT / "refhic_predictions" / "converted_mcool"
OUTPUT_ROOT = REPO_ROOT / "robustad_predictions"
TMP_ROOT = REPO_ROOT / "robustad_predictions" / ".mplcache"

CHROM = "chr4"
RESOLUTION_BP = 250_000
RESOLUTION_LABEL = "250kb"
SAMPLES = ("CMP", "MPP")


def read_ontad_domains(sample: str) -> list[dict[str, int]]:
    path = HEMATO_ROOT / sample / CHROM / RESOLUTION_LABEL / "tads.tad"
    domains = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            start_bin, end_bin, level, *_rest = line.split()
            start_bin = int(start_bin)
            end_bin = int(end_bin)
            level = int(level)
            domains.append(
                {
                    "start_bin": start_bin,
                    "end_bin": end_bin,
                    "level": level,
                    "span_bins": end_bin - start_bin,
                }
            )
    return domains


def recommend_windows(samples: tuple[str, ...]) -> dict[str, object]:
    sample_stats = {}
    non_root_spans = []

    for sample in samples:
        domains = read_ontad_domains(sample)
        sample_non_root = [d["span_bins"] for d in domains if d["level"] > 0]
        if not sample_non_root:
            raise ValueError(f"No non-root OnTAD domains found for {sample}")
        sample_stats[sample] = {
            "count_total": len(domains),
            "count_non_root": len(sample_non_root),
            "min_span_bins": min(sample_non_root),
            "median_span_bins": median(sample_non_root),
            "max_span_bins": max(sample_non_root),
        }
        non_root_spans.extend(sample_non_root)

    min_bins = min(non_root_spans)
    max_bins = max(non_root_spans)
    summary = {
        "chrom": CHROM,
        "resolution_bp": RESOLUTION_BP,
        "resolution_label": RESOLUTION_LABEL,
        "samples": sample_stats,
        "recommended": {
            "min_span_bins": min_bins,
            "max_span_bins": max_bins,
            "minW_kb": min_bins * RESOLUTION_BP // 1000,
            "maxW_kb": max_bins * RESOLUTION_BP // 1000,
            "mintad_bp": min_bins * RESOLUTION_BP,
            "maxtad_bp": max_bins * RESOLUTION_BP,
        },
        "notes": [
            "Window recommendations are derived from non-root OnTAD domains only.",
            "The whole-chromosome root domain is excluded because it would dominate max-window selection.",
            "RobusTAD is run with boundary+assembly only; no reference panel is used.",
        ],
    }
    return summary


def robustad_base_cmd() -> list[str]:
    return [sys.executable, "-m", "robustad.robustad"]


def resolve_cooler_chrom_label(coolfile: Path) -> str:
    clr = cooler.Cooler(f"{coolfile}::/resolutions/{RESOLUTION_BP}")
    chromnames = list(clr.chromnames)
    if CHROM in chromnames:
        return CHROM
    if CHROM.startswith("chr") and CHROM[3:] in chromnames:
        return CHROM[3:]
    prefixed = f"chr{CHROM}"
    if prefixed in chromnames:
        return prefixed
    raise ValueError(f"Could not resolve chromosome label for {CHROM} in {coolfile}")


def run_command(cmd: list[str], env: dict[str, str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def normalize_boundary_chrom(boundary_file: Path, chrom_label: str) -> None:
    df = pd.read_csv(boundary_file, sep="\t", header=None)
    df[0] = chrom_label
    df.to_csv(boundary_file, sep="\t", header=False, index=False)


def run_sample(sample: str, params: dict[str, int], env: dict[str, str]) -> None:
    coolfile = MCOOL_ROOT / f"GSE152918_{sample}.mcool"
    outdir = OUTPUT_ROOT / sample / CHROM / RESOLUTION_LABEL
    outdir.mkdir(parents=True, exist_ok=True)
    cooler_chrom = resolve_cooler_chrom_label(coolfile)

    boundary_prefix = outdir / f"{sample}_{CHROM}_{RESOLUTION_LABEL}_boundary"
    tad_prefix = outdir / f"{sample}_{CHROM}_{RESOLUTION_LABEL}_tads"

    boundary_cmd = robustad_base_cmd() + [
        "boundary",
        "--resol",
        str(RESOLUTION_BP),
        "--chr",
        cooler_chrom,
        "--minW",
        str(params["minW_kb"]),
        "--maxW",
        str(params["maxW_kb"]),
        str(coolfile),
        str(boundary_prefix),
    ]
    run_command(boundary_cmd, env=env)

    boundary_file = Path(str(boundary_prefix) + ".bed")
    normalize_boundary_chrom(boundary_file, cooler_chrom)
    assembly_cmd = robustad_base_cmd() + [
        "assembly",
        "--resol",
        str(RESOLUTION_BP),
        "--chr",
        cooler_chrom,
        "--mintad",
        str(params["mintad_bp"]),
        "--maxtad",
        str(params["maxtad_bp"]),
        str(coolfile),
        str(boundary_file),
        str(tad_prefix),
    ]
    run_command(assembly_cmd, env=env)


def main() -> None:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    summary = recommend_windows(SAMPLES)
    params = summary["recommended"]

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{ROBUSTAD_ROOT}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(ROBUSTAD_ROOT)
    )
    env["MPLCONFIGDIR"] = str(TMP_ROOT)

    summary_path = OUTPUT_ROOT / "cmp_mpp_chr4_250kb_window_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved window summary to {summary_path}")

    for sample in SAMPLES:
        run_sample(sample, params, env)


if __name__ == "__main__":
    main()
