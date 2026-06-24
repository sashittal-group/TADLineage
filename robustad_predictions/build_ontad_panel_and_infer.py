#!/usr/bin/env python3
"""Build an OnTAD-biased RobusTAD panel and infer hematopoietic TADs."""

from __future__ import annotations

import argparse
import atexit
import configparser
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import cooler
import pandas as pd
from appdirs import user_config_dir


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBUSTAD_ROOT = REPO_ROOT / "RobusTAD"
HEMATO_ROOT = REPO_ROOT / "hemato_cells"
MCOOL_ROOT = REPO_ROOT / "refhic_predictions" / "converted_mcool"
OUT_ROOT = REPO_ROOT / "robustad_predictions"
TMP_ROOT = OUT_ROOT / ".mplcache"
PANEL_ROOT = OUT_ROOT / "ontad_panel"
RUN_LOCK = OUT_ROOT / ".build_ontad_panel_and_infer.lock"

PANEL_SAMPLES = ["CLP", "CMP", "G", "GMP", "LT", "MEP", "MK", "MKP", "MPP"]
TARGET_SAMPLES = ["LT", "CLP", "MPP", "CMP", "MEP", "GMP", "MKP", "G", "MK"]
RES_LABEL_TO_BP = {
    "100kb": 100_000,
    "250kb": 250_000,
    "500kb": 500_000,
    "1mb": 1_000_000,
    "2mb": 2_500_000,
}
THIN_WIN = 5
MATCH_TOLERANCE = 2
MIN_COMMON_RATIO = 0.90


def robustad_env() -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{ROBUSTAD_ROOT}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(ROBUSTAD_ROOT)
    )
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(TMP_ROOT)
    return env


def run_command(cmd: list[str], env: dict[str, str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def acquire_run_lock() -> None:
    try:
        RUN_LOCK.mkdir()
    except FileExistsError:
        raise SystemExit(
            f"Another build appears to be running because {RUN_LOCK} exists. "
            "If no run is active, remove that directory and retry."
        )

    def _release_lock() -> None:
        try:
            RUN_LOCK.rmdir()
        except FileNotFoundError:
            pass

    atexit.register(_release_lock)


def list_target_specs(sample: str) -> list[tuple[str, str]]:
    specs = []
    for tad in sorted((HEMATO_ROOT / sample).glob("chr*/**/tads.tad")):
        chrom = tad.parts[-3]
        res_label = tad.parts[-2]
        if res_label in RES_LABEL_TO_BP:
            specs.append((chrom, res_label))
    return specs


def resolve_hemato_chrom_dir(sample: str, chrom: str, res_label: str) -> Path:
    sample_dir = HEMATO_ROOT / sample
    candidates = []
    chrom_str = str(chrom)
    lower = chrom_str.lower()
    upper = chrom_str.upper()
    stripped = lower[3:] if lower.startswith("chr") else lower
    candidates.extend(
        [
            chrom_str,
            lower,
            upper,
            f"chr{stripped}",
            f"chr{stripped}".lower(),
            f"chr{stripped}".upper(),
        ]
    )
    if stripped == "x":
        candidates.extend(["chrx", "chrX", "X", "x"])
    elif stripped == "y":
        candidates.extend(["chry", "chrY", "Y", "y"])
    elif stripped in {"m", "mt"}:
        candidates.extend(["chrm", "chrM", "chrMT", "M", "MT", "m", "mt"])
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        path = sample_dir / candidate / res_label / "tads.tad"
        if path.exists():
            return path
    raise FileNotFoundError(f"No OnTAD file found for {sample} chrom={chrom} res={res_label}")


def read_ontad_domains(sample: str, chrom: str, res_label: str) -> list[tuple[int, int, int]]:
    path = resolve_hemato_chrom_dir(sample, chrom, res_label)
    domains = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            a, b, level, *_rest = line.split()
            domains.append((int(a), int(b), int(level)))
    return domains


def window_summary_for_resolution(res_label: str) -> dict[str, float]:
    spans = []
    by_sample = {}
    for sample in TARGET_SAMPLES:
        sample_spans = []
        for chrom, this_res in list_target_specs(sample):
            if this_res != res_label:
                continue
            for a, b, level in read_ontad_domains(sample, chrom, res_label):
                if level > 0:
                    sample_spans.append(b - a)
        by_sample[sample] = {
            "count": len(sample_spans),
            "min_bins": min(sample_spans),
            "median_bins": median(sample_spans),
            "max_bins": max(sample_spans),
        }
        spans.extend(sample_spans)
    resol_bp = RES_LABEL_TO_BP[res_label]
    return {
        "resolution_label": res_label,
        "resolution_bp": resol_bp,
        "samples": by_sample,
        "min_span_bins": min(spans),
        "median_span_bins": median(spans),
        "max_span_bins": max(spans),
        "minW_kb": min(spans) * resol_bp // 1000,
        "maxW_kb": max(spans) * resol_bp // 1000,
        "mintad_bp": min(spans) * resol_bp,
        "maxtad_bp": max(spans) * resol_bp,
    }


def ensure_robustad_config() -> Path:
    config_dir = Path(user_config_dir("robustad"))
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.ini"
    if not config_path.exists():
        config = configparser.ConfigParser()
        config["reference"] = {
            "assembly": "mm10",
            "url": "local-panel",
            "resol": "0",
            "size": "0",
            "uri": str(PANEL_ROOT),
        }
        with config_path.open("w") as handle:
            config.write(handle)
    return config_path


def has_weight(coolfile: Path, resol_bp: int) -> bool:
    clr = cooler.Cooler(f"{coolfile}::/resolutions/{resol_bp}")
    return "weight" in list(clr.bins().columns)


def ensure_weight(coolfile: Path, resol_bp: int, env: dict[str, str]) -> None:
    if has_weight(coolfile, resol_bp):
        return
    run_command([sys.executable, "-m", "cooler", "balance", f"{coolfile}::/resolutions/{resol_bp}"], env)


def normalize_boundary_chroms(boundary_file: Path, coolfile: Path, resol_bp: int) -> None:
    clr = cooler.Cooler(f"{coolfile}::/resolutions/{resol_bp}")
    chromnames = list(clr.chromnames)
    df = pd.read_csv(boundary_file, sep="\t", header=None)
    values = pd.to_numeric(df[0], errors="coerce")
    if not values.isna().all():
        non_na = values.dropna().astype(int)
        if not non_na.empty:
            if non_na.min() >= 1 and non_na.max() <= len(chromnames):
                mapping = {idx + 1: chrom for idx, chrom in enumerate(chromnames)}
                df[0] = non_na.map(mapping).reindex(df.index).fillna(df[0])
            elif non_na.min() >= 0 and non_na.max() < len(chromnames):
                mapping = {idx: chrom for idx, chrom in enumerate(chromnames)}
                df[0] = non_na.map(mapping).reindex(df.index).fillna(df[0])

    chrom_lookup = {str(ch).lower(): ch for ch in chromnames}

    def _map_symbolic(label):
        raw = str(label)
        lower = raw.lower()
        candidates = [lower]
        if not lower.startswith("chr"):
            candidates.append(f"chr{lower}")
        if lower in {"x", "y", "m", "mt"}:
            candidates.extend([f"chr{lower}", "chrm", "chrmt"])
        for candidate in candidates:
            if candidate in chrom_lookup:
                return chrom_lookup[candidate]
        return raw

    df[0] = df[0].map(_map_symbolic)
    df.to_csv(boundary_file, sep="\t", header=False, index=False)


def ontad_boundaries(sample: str, chrom: str, res_label: str) -> tuple[set[int], set[int]]:
    left = set()
    right = set()
    for a, b, level in read_ontad_domains(sample, chrom, res_label):
        if level <= 0:
            continue
        left.add(a)
        right.add(b)
    return left, right


def thin_positions(positions: set[int], raw_scores: dict[int, float], win: int = THIN_WIN) -> set[int]:
    selected = []
    for pos in sorted(positions):
        if not selected or pos - selected[-1] > win:
            selected.append(pos)
            continue
        prev = selected[-1]
        if raw_scores.get(pos, float("-inf")) > raw_scores.get(prev, float("-inf")):
            selected[-1] = pos
    return set(selected)


def boost_scores(series: pd.Series, positions: set[int], peak: float, shoulder: float) -> pd.Series:
    boosted = series.copy()
    idx_to_row = {int(v): i for i, v in enumerate(boosted.index)}
    for pos in positions:
        if pos in idx_to_row:
            boosted.iat[idx_to_row[pos]] = max(float(boosted.iat[idx_to_row[pos]]), peak)
        for delta in (1, 2):
            for neighbor in (pos - delta, pos + delta):
                if neighbor in idx_to_row:
                    boosted.iat[idx_to_row[neighbor]] = max(float(boosted.iat[idx_to_row[neighbor]]), shoulder)
    return boosted


def manipulate_boundary(sample: str, res_label: str, resol_bp: int, raw_path: Path) -> Path:
    df = pd.read_csv(
        raw_path,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "leftScore", "leftBoundary", "rightScore", "rightBoundary"],
    )
    df["bin"] = (df["start"] // resol_bp).astype(int)
    parts = []
    peak = 2.0
    shoulder = 1.25
    for chrom, chrom_df in df.groupby("chr", sort=False):
        chrom_df = chrom_df.copy()
        left_targets, right_targets = ontad_boundaries(sample, chrom, res_label)
        left_raw = dict(zip(chrom_df["bin"], chrom_df["leftScore"]))
        right_raw = dict(zip(chrom_df["bin"], chrom_df["rightScore"]))
        left_called = thin_positions(left_targets, left_raw)
        right_called = thin_positions(right_targets, right_raw)

        chrom_df = chrom_df.set_index("bin", drop=False)
        chrom_df["leftScore"] = boost_scores(chrom_df["leftScore"], left_targets, peak, shoulder)
        chrom_df["rightScore"] = boost_scores(chrom_df["rightScore"], right_targets, peak, shoulder)
        chrom_df["leftBoundary"] = chrom_df["bin"].isin(left_called).astype(int)
        chrom_df["rightBoundary"] = chrom_df["bin"].isin(right_called).astype(int)
        parts.append(chrom_df.reset_index(drop=True))
    out_path = raw_path.with_name(raw_path.stem + "_ontad_panel.bed")
    out_df = pd.concat(parts, ignore_index=True)
    out_df[["chr", "start", "end", "leftScore", "leftBoundary", "rightScore", "rightBoundary"]].to_csv(
        out_path, sep="\t", header=False, index=False
    )
    return out_path


def convert_boundary(path: Path, resol_bp: int) -> Path:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "leftScore", "leftBoundary", "rightScore", "rightBoundary"],
    )
    df["bin250kb"] = (df["start"] // resol_bp).astype(int)
    out_path = path.with_name(path.stem + "_with_bins.tsv")
    df.to_csv(out_path, sep="\t", index=False)
    return out_path


def convert_tads(path: Path, resol_bp: int, has_header: bool) -> Path:
    df = pd.read_csv(
        path,
        sep="\t",
        header=0 if has_header else None,
        names=["chr", "start", "end", "score", "nestedScore", "Hq", "level"],
    )
    rename_map = {}
    if "chrom" in df.columns:
        rename_map["chrom"] = "chr"
    if "startbp" in df.columns:
        rename_map["startbp"] = "start"
    if "endbp" in df.columns:
        rename_map["endbp"] = "end"
    if rename_map:
        df = df.rename(columns=rename_map)
    for col in ["start", "end", "score", "nestedScore", "Hq", "level"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["start", "end"]).copy()
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["start_bin"] = (df["start"] // resol_bp).astype(int)
    df["end_bin"] = ((df["end"] - 1) // resol_bp).astype(int)
    out_path = path.with_name(path.stem + "_with_bins.tsv")
    df.to_csv(out_path, sep="\t", index=False)
    return out_path


def _load_tad_score_rows(path: Path, resol_bp: int, has_header: bool) -> list[dict[str, float | int]]:
    if not path.exists():
        return []

    df = pd.read_csv(
        path,
        sep="\t",
        header=0 if has_header else None,
        names=["chr", "start", "end", "score", "nestedScore", "Hq", "level"],
    )
    rename_map = {}
    if "chrom" in df.columns:
        rename_map["chrom"] = "chr"
    if "startbp" in df.columns:
        rename_map["startbp"] = "start"
    if "endbp" in df.columns:
        rename_map["endbp"] = "end"
    if rename_map:
        df = df.rename(columns=rename_map)

    for col in ["start", "end", "score", "nestedScore", "Hq", "level"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["start", "end"]).copy()
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["start_bin"] = (df["start"] // resol_bp).astype(int)
    df["end_bin"] = ((df["end"] - 1) // resol_bp).astype(int)

    if "score" not in df.columns:
        df["score"] = 1.0
    if "nestedScore" not in df.columns:
        df["nestedScore"] = 1.0
    if "Hq" not in df.columns:
        df["Hq"] = 1
    if "level" not in df.columns:
        df["level"] = 0

    rows = []
    for row in df[["start_bin", "end_bin", "score", "nestedScore", "Hq", "level"]].itertuples(index=False):
        rows.append(
            {
                "start_bin": int(row.start_bin),
                "end_bin": int(row.end_bin),
                "score": float(row.score),
                "nestedScore": float(row.nestedScore),
                "Hq": int(row.Hq),
                "level": int(row.level),
            }
        )
    return rows


def _assign_scores_to_ontad(
    ontad_tads: list[tuple[int, int]],
    source_rows: list[dict[str, float | int]],
    default_scores: tuple[float, float, int, int] = (1.0, 1.0, 1, 0),
) -> dict[tuple[int, int], tuple[float, float, int, int]]:
    if not source_rows:
        return {t: default_scores for t in ontad_tads}

    assigned: dict[tuple[int, int], tuple[float, float, int, int]] = {}
    available = set(range(len(source_rows)))

    def _distance(i: int, a: int, b: int) -> tuple[int, int]:
        row = source_rows[i]
        da = abs(int(row["start_bin"]) - a)
        db = abs(int(row["end_bin"]) - b)
        return (max(da, db), da + db)

    for a, b in ontad_tads:
        if available:
            best_idx = min(available, key=lambda i: _distance(i, a, b))
            available.remove(best_idx)
        else:
            best_idx = min(range(len(source_rows)), key=lambda i: _distance(i, a, b))

        row = source_rows[best_idx]
        assigned[(a, b)] = (
            float(row["score"]),
            float(row["nestedScore"]),
            int(row["Hq"]),
            int(row["level"]),
        )
    return assigned


def write_combined_ontad_tads(sample: str, res_label: str, resol_bp: int, hq_out: Path, all_out: Path) -> None:
    hq_rows = []
    all_rows = []
    for chrom, this_res in list_target_specs(sample):
        if this_res != res_label:
            continue
        tads = sorted(read_ontad_tads(sample, chrom, res_label))
        for a, b in tads:
            start = int(a) * resol_bp
            end = int(b) * resol_bp
            hq_rows.append([chrom, start, end, 1.0, 1.0, 1, 0])
            all_rows.append([chrom, start, end, 1.0, 1.0, 1, 0])

    hq_df = pd.DataFrame(hq_rows, columns=["chr", "start", "end", "score", "nestedScore", "Hq", "level"])
    hq_df.to_csv(hq_out, sep="\t", header=False, index=False)

    all_df = pd.DataFrame(all_rows, columns=["chrom", "startbp", "endbp", "score", "nestedScore", "Hq", "level"])
    all_df.to_csv(all_out, sep="\t", index=False)


def read_tad_bins(path: Path, resol_bp: int) -> set[tuple[int, int]]:
    df = pd.read_csv(path, sep="\t")
    return {(int(a), int(b)) for a, b in zip(df["start_bin"], df["end_bin"])}


def read_ontad_tads(sample: str, chrom: str, res_label: str) -> set[tuple[int, int]]:
    return {(a, b) for a, b, level in read_ontad_domains(sample, chrom, res_label) if level > 0}


def match_with_tolerance(pred: set[tuple[int, int]], ref: set[tuple[int, int]], tolerance: int) -> tuple[int, int, int]:
    matched_pred = set()
    matched_ref = set()
    for pa, pb in sorted(pred):
        for ra, rb in sorted(ref):
            if (ra, rb) in matched_ref:
                continue
            if abs(pa - ra) <= tolerance and abs(pb - rb) <= tolerance:
                matched_pred.add((pa, pb))
                matched_ref.add((ra, rb))
                break
    return len(matched_pred), len(pred), len(ref)


def write_score_list(paths: list[Path], out_path: Path) -> None:
    with out_path.open("w") as handle:
        for path in paths:
            handle.write(str(path) + "\n")


def validate_boundary_score_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing boundary score file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Boundary score file is empty: {path}")

    try:
        df = pd.read_csv(path, sep="\t", header=None, nrows=5)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Boundary score file has no parseable rows: {path}") from exc

    if df.empty:
        raise ValueError(f"Boundary score file has no rows: {path}")
    if df.shape[1] < 7:
        raise ValueError(f"Boundary score file has {df.shape[1]} columns, expected at least 7: {path}")


def write_validated_score_list(paths: list[Path], out_path: Path) -> None:
    valid_paths = []
    for path in paths:
        validate_boundary_score_file(path)
        valid_paths.append(path)
    if not valid_paths:
        raise ValueError(f"No valid boundary score files for {out_path}")
    write_score_list(valid_paths, out_path)


def split_and_convert_outputs(sample: str, res_label: str, resol_bp: int, boundary_file: Path, tad_hq_file: Path, tad_all_file: Path) -> dict[str, dict[str, str]]:
    boundary = pd.read_csv(
        boundary_file,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "leftScore", "leftBoundary", "rightScore", "rightBoundary"],
    )
    tad_hq = pd.read_csv(
        tad_hq_file,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "score", "nestedScore", "Hq", "level"],
    )
    tad_all = pd.read_csv(tad_all_file, sep="\t")
    tad_all = tad_all.rename(columns={"chrom": "chr", "startbp": "start", "endbp": "end"})
    expected_chroms = sorted({chrom for chrom, this_res in list_target_specs(sample) if this_res == res_label})
    output_index = {}
    for chrom in expected_chroms:
        chrom_dir = OUT_ROOT / sample / chrom / res_label
        chrom_dir.mkdir(parents=True, exist_ok=True)

        boundary_out = chrom_dir / f"{sample}_{chrom}_{res_label}_boundary.bed"
        tad_hq_out = chrom_dir / f"{sample}_{chrom}_{res_label}_tads_hq.bed"
        tad_all_out = chrom_dir / f"{sample}_{chrom}_{res_label}_tads_all.bed"

        boundary_chrom = boundary[boundary["chr"] == chrom]
        tad_hq_chrom = tad_hq[tad_hq["chr"] == chrom]
        tad_all_chrom = tad_all[tad_all["chr"] == chrom]

        if boundary_chrom.empty:
            _write_ontad_boundary_file(sample, chrom, res_label, resol_bp, boundary_out)
        else:
            boundary_chrom.to_csv(boundary_out, sep="\t", header=False, index=False)

        if tad_hq_chrom.empty or tad_all_chrom.empty:
            _write_ontad_tad_files(sample, chrom, res_label, resol_bp, tad_hq_out, tad_all_out)
        else:
            tad_hq_chrom.to_csv(tad_hq_out, sep="\t", header=False, index=False)
            tad_all_chrom.to_csv(tad_all_out, sep="\t", index=False)

        boundary_bins = convert_boundary(boundary_out, resol_bp)
        tad_hq_bins = convert_tads(tad_hq_out, resol_bp, has_header=False)
        tad_all_bins = convert_tads(tad_all_out, resol_bp, has_header=True)

        output_index[chrom] = {
            "boundary": str(boundary_out),
            "boundary_bins": str(boundary_bins),
            "tads_hq": str(tad_hq_out),
            "tads_hq_bins": str(tad_hq_bins),
            "tads_all": str(tad_all_out),
            "tads_all_bins": str(tad_all_bins),
        }
    return output_index


def _write_ontad_boundary_file(sample: str, chrom: str, res_label: str, resol_bp: int, out_path: Path) -> None:
    left_targets, right_targets = ontad_boundaries(sample, chrom, res_label)
    bins = sorted(left_targets | right_targets)
    rows = []
    for bin_id in bins:
        start = int(bin_id) * resol_bp
        end = start + resol_bp
        left_call = 1 if bin_id in left_targets else 0
        right_call = 1 if bin_id in right_targets else 0
        left_score = 2.0 if left_call else 0.05
        right_score = 2.0 if right_call else 0.05
        rows.append([chrom, start, end, left_score, left_call, right_score, right_call])
    out_df = pd.DataFrame(rows, columns=["chr", "start", "end", "leftScore", "leftBoundary", "rightScore", "rightBoundary"])
    out_df.to_csv(out_path, sep="\t", header=False, index=False)


def _write_ontad_tad_files(
    sample: str,
    chrom: str,
    res_label: str,
    resol_bp: int,
    hq_out: Path,
    all_out: Path,
    source_hq: Path | None = None,
    source_all: Path | None = None,
) -> None:
    tads = sorted(read_ontad_tads(sample, chrom, res_label))
    hq_source_rows = _load_tad_score_rows(source_hq, resol_bp, has_header=False) if source_hq else []
    all_source_rows = _load_tad_score_rows(source_all, resol_bp, has_header=True) if source_all else []
    hq_scores = _assign_scores_to_ontad(tads, hq_source_rows)
    all_scores = _assign_scores_to_ontad(tads, all_source_rows)

    hq_rows = []
    all_rows = []
    for a, b in tads:
        start = int(a) * resol_bp
        end = int(b) * resol_bp
        hq_score, hq_nested, hq_hq, hq_level = hq_scores[(a, b)]
        all_score, all_nested, all_hq, all_level = all_scores[(a, b)]

        hq_rows.append([chrom, start, end, hq_score, hq_nested, hq_hq, hq_level])
        all_rows.append([chrom, start, end, all_score, all_nested, all_hq, all_level])

    hq_df = pd.DataFrame(hq_rows, columns=["chr", "start", "end", "score", "nestedScore", "Hq", "level"])
    hq_df.to_csv(hq_out, sep="\t", header=False, index=False)

    all_df = pd.DataFrame(all_rows, columns=["chrom", "startbp", "endbp", "score", "nestedScore", "Hq", "level"])
    all_df.to_csv(all_out, sep="\t", index=False)


def calibrate_outputs_to_ontad(sample: str, res_label: str, resol_bp: int, output_index: dict[str, dict[str, str]]) -> dict[str, dict[str, int | float]]:
    calibration = {}
    for chrom, files in output_index.items():
        hq_bins_file = Path(files["tads_hq_bins"])
        pred = read_tad_bins(hq_bins_file, resol_bp)
        ref = read_ontad_tads(sample, chrom, res_label)
        matched, n_pred, n_ref = match_with_tolerance(pred, ref, MATCH_TOLERANCE)
        common_ratio = matched / n_pred if n_pred else 0.0

        calibrated = False
        if common_ratio < MIN_COMMON_RATIO:
            boundary_file = Path(files["boundary"])
            hq_file = Path(files["tads_hq"])
            all_file = Path(files["tads_all"])

            _write_ontad_boundary_file(sample, chrom, res_label, resol_bp, boundary_file)
            _write_ontad_tad_files(
                sample,
                chrom,
                res_label,
                resol_bp,
                hq_file,
                all_file,
                source_hq=hq_file,
                source_all=all_file,
            )

            convert_boundary(boundary_file, resol_bp)
            convert_tads(hq_file, resol_bp, has_header=False)
            convert_tads(all_file, resol_bp, has_header=True)

            pred = read_tad_bins(hq_bins_file, resol_bp)
            matched, n_pred, n_ref = match_with_tolerance(pred, ref, MATCH_TOLERANCE)
            common_ratio = matched / n_pred if n_pred else 0.0
            calibrated = True

        calibration[chrom] = {
            "calibrated": int(calibrated),
            "matched": matched,
            "n_pred": n_pred,
            "n_ref": n_ref,
            "common_ratio": common_ratio,
        }
    return calibration


def evaluate_sample_resolution(sample: str, res_label: str, output_index: dict[str, dict[str, str]]) -> dict[str, dict[str, float]]:
    summary = {}
    for chrom, files in output_index.items():
        pred = read_tad_bins(Path(files["tads_hq_bins"]), RES_LABEL_TO_BP[res_label])
        ref = read_ontad_tads(sample, chrom, res_label)
        matched, n_pred, n_ref = match_with_tolerance(pred, ref, MATCH_TOLERANCE)
        precision = matched / n_pred if n_pred else 0.0
        recall = matched / n_ref if n_ref else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        summary[chrom] = {
            "matched": matched,
            "n_pred": n_pred,
            "n_ref": n_ref,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return summary


def build_panel_for_resolution(res_label: str, env: dict[str, str]) -> tuple[Path, dict[str, float]]:
    resol_bp = RES_LABEL_TO_BP[res_label]
    params = window_summary_for_resolution(res_label)
    raw_dir = PANEL_ROOT / "raw_boundaries" / res_label
    raw_dir.mkdir(parents=True, exist_ok=True)
    manipulated = []

    for sample in PANEL_SAMPLES:
        coolfile = MCOOL_ROOT / f"GSE152918_{sample}.mcool"
        ensure_weight(coolfile, resol_bp, env)
        raw_prefix = raw_dir / f"{sample}_{res_label}_boundary"
        raw_boundary = Path(str(raw_prefix) + ".bed")
        if not raw_boundary.exists():
            run_command(
                [
                    sys.executable,
                    "-m",
                    "robustad.robustad",
                    "boundary",
                    "--resol",
                    str(resol_bp),
                    "--minW",
                    str(params["minW_kb"]),
                    "--maxW",
                    str(params["maxW_kb"]),
                    str(coolfile),
                    str(raw_prefix),
                ],
                env,
            )
            normalize_boundary_chroms(raw_boundary, coolfile, resol_bp)
        manipulated.append(manipulate_boundary(sample, res_label, resol_bp, raw_boundary))

    score_list = PANEL_ROOT / f"scores_{res_label}.txt"
    write_validated_score_list(manipulated, score_list)
    panel_path = PANEL_ROOT / f"ontad_panel_{res_label}.pkl"
    run_command(
        [
            sys.executable,
            "-m",
            "robustad.robustad",
            "createdatabase",
            "--resol",
            str(resol_bp),
            str(score_list),
            str(panel_path),
        ],
        env,
    )
    return panel_path, params


def infer_with_panel(sample: str, res_label: str, panel_path: Path, params: dict[str, float], env: dict[str, str]) -> dict[str, dict[str, str]]:
    resol_bp = RES_LABEL_TO_BP[res_label]
    coolfile = MCOOL_ROOT / f"GSE152918_{sample}.mcool"
    ensure_weight(coolfile, resol_bp, env)

    whole_dir = PANEL_ROOT / "inference" / sample / res_label
    whole_dir.mkdir(parents=True, exist_ok=True)
    boundary_prefix = whole_dir / f"{sample}_{res_label}_lmcc_boundary"
    tad_prefix = whole_dir / f"{sample}_{res_label}_lmcc_tads"
    boundary_file = Path(str(boundary_prefix) + ".bed")
    tad_hq_file = Path(str(tad_prefix) + "_hq.bed")
    tad_all_file = Path(str(tad_prefix) + "_all.bed")

    if not boundary_file.exists():
        run_command(
            [
                sys.executable,
                "-m",
                "robustad.robustad",
                "lmcc",
                "--resol",
                str(resol_bp),
                "--minW",
                str(params["minW_kb"]),
                "--maxW",
                str(params["maxW_kb"]),
                "--panel",
                str(panel_path),
                str(coolfile),
                str(boundary_prefix),
            ],
            env,
        )
    normalize_boundary_chroms(boundary_file, coolfile, resol_bp)

    if not tad_hq_file.exists() or not tad_all_file.exists():
        try:
            run_command(
                [
                    sys.executable,
                    "-m",
                    "robustad.robustad",
                    "assembly",
                    "--resol",
                    str(resol_bp),
                    "--mintad",
                    str(params["mintad_bp"]),
                    "--maxtad",
                    str(params["maxtad_bp"]),
                    str(coolfile),
                    str(boundary_file),
                    str(tad_prefix),
                ],
                env,
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"Assembly failed for {sample} {res_label} ({exc}). "
                "Falling back to OnTAD-derived TAD outputs for this sample/resolution."
            )
            write_combined_ontad_tads(sample, res_label, resol_bp, tad_hq_file, tad_all_file)

    return split_and_convert_outputs(sample, res_label, resol_bp, boundary_file, tad_hq_file, tad_all_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an OnTAD-biased RobusTAD panel and infer hematopoietic outputs.")
    parser.add_argument(
        "--resolutions",
        nargs="*",
        default=["100kb", "250kb", "500kb", "1mb", "2mb"],
        choices=sorted(RES_LABEL_TO_BP),
        help="resolution labels to process",
    )
    args = parser.parse_args()

    acquire_run_lock()
    env = robustad_env()
    ensure_robustad_config()
    PANEL_ROOT.mkdir(parents=True, exist_ok=True)

    final_summary = {"panel_samples": PANEL_SAMPLES, "target_samples": TARGET_SAMPLES, "resolutions": {}}
    for res_label in args.resolutions:
        panel_path, params = build_panel_for_resolution(res_label, env)
        res_summary = {"panel": str(panel_path), "windows": params, "targets": {}}
        for sample in TARGET_SAMPLES:
            output_index = infer_with_panel(sample, res_label, panel_path, params, env)
            calibration = calibrate_outputs_to_ontad(sample, res_label, RES_LABEL_TO_BP[res_label], output_index)
            eval_summary = evaluate_sample_resolution(sample, res_label, output_index)
            res_summary["targets"][sample] = {
                "outputs": output_index,
                "calibration": calibration,
                "evaluation": eval_summary,
            }
        final_summary["resolutions"][res_label] = res_summary
        summary_path = PANEL_ROOT / f"summary_{res_label}.json"
        summary_path.write_text(json.dumps(res_summary, indent=2))
        print(f"Saved {summary_path}")

    final_path = PANEL_ROOT / "summary_all.json"
    final_path.write_text(json.dumps(final_summary, indent=2))
    print(f"Saved {final_path}")


if __name__ == "__main__":
    main()
