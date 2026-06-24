#!/usr/bin/env python3
"""
Build balanced single-resolution .mcool files for selected GSE129029 cells,
run RobusTAD boundary/TAD calling, and maintain one stage-specific reference
panel per developmental stage and resolution.

The workflow is intentionally sequential and restartable:
- one cell is processed at a time
- only one .mcool is created at a time
- existing outputs are reused unless --force is given
- when panels are enabled, each stage/resolution is processed in two passes:
  1. raw boundary scores for panel creation
  2. LMCC boundary refinement with that panel, followed by assembly
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/Users/PSLabMonisha/Desktop/tads")
SCPTR_ROOT = PROJECT_ROOT / "scptr"
SOURCE_ROOT = SCPTR_ROOT / "cell_differentiation"
OUTPUT_ROOT = SCPTR_ROOT / "robustad_scptr_data"
METADATA_FILE = SCPTR_ROOT / "GSE129029_metadata.csv"
ROBUSTAD_ROOT = PROJECT_ROOT / "robustad"
TMP_ROOT = Path("/tmp/robustad_scptr")
STAGES = ["1-cell", "2-cell", "4-cell", "8-cell", "64-cell"]

# Mouse genome mm10 chromosome sizes.
MM10_CHROM_SIZES = {
    "chr1": 195471971,
    "chr2": 182113224,
    "chr3": 160039680,
    "chr4": 156508116,
    "chr5": 151834684,
    "chr6": 149736546,
    "chr7": 145441459,
    "chr8": 129401213,
    "chr9": 124595110,
    "chr10": 130694993,
    "chr11": 122082543,
    "chr12": 120129022,
    "chr13": 120421639,
    "chr14": 124902244,
    "chr15": 104043685,
    "chr16": 98207768,
    "chr17": 94987271,
    "chr18": 90702639,
    "chr19": 61431566,
    "chrX": 171031299,
}

VALID_CHROMS = tuple(MM10_CHROM_SIZES.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RobusTAD-ready mouse reference panels for GSE129029."
    )
    parser.add_argument(
        "--resolution",
        dest="resolutions",
        action="append",
        type=int,
        default=None,
        help="Resolution stored in each .mcool file. Repeat to process multiple resolutions. Default: 250000",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="Rows per chunk while streaming contact files. Default: 1000000",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=["1-cell", "2-cell", "4-cell", "8-cell", "64-cell"],
        help="Optional stage filter. Repeat to process multiple stages.",
    )
    parser.add_argument(
        "--targets-file",
        type=Path,
        default=None,
        help="Optional TSV/CSV file listing specific GSMs to process. Must include a GSM column.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on processed cells after filtering, for testing only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild existing per-cell outputs and stage reference panels.",
    )
    parser.add_argument(
        "--keep-mcool",
        action="store_true",
        help="Keep per-cell .mcool files after successful RobusTAD completion.",
    )
    parser.add_argument(
        "--skip-panels",
        action="store_true",
        help="Skip stage-level reference panel creation and panel-aware boundary refinement.",
    )
    parser.add_argument(
        "--panels-only",
        action="store_true",
        help="Create or refresh stage reference panels from existing/raw boundary scores without running LMCC assembly.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Default: INFO",
    )
    return parser.parse_args()


def selected_resolutions(args: argparse.Namespace) -> List[int]:
    resolutions = args.resolutions if args.resolutions else [250_000]
    unique = []
    seen = set()
    for resolution in resolutions:
        if resolution <= 0:
            raise ValueError(f"Resolution must be positive: {resolution}")
        if resolution not in seen:
            unique.append(resolution)
            seen.add(resolution)
    return unique


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def bootstrap_imports() -> Tuple[object, object, object, object, object, object, object]:
    sys.path.insert(0, str(ROBUSTAD_ROOT))
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(TMP_ROOT / "mplconfig"))

    import cooler
    from cooler import create_cooler
    try:
        from cooler._balance import balance_cooler
    except ModuleNotFoundError:
        from cooler.balance import balance_cooler
    from robustad.assembly import assembly
    from robustad.boundary import boundary
    from robustad.lmcc import lmcc
    from robustad.util import createdatabase

    return cooler, create_cooler, balance_cooler, boundary, lmcc, assembly, createdatabase


def load_target_gsms(targets_file: Path | None) -> set[str] | None:
    if targets_file is None:
        return None
    with targets_file.open() as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "GSM" not in reader.fieldnames:
            raise ValueError(f"Targets file must include a GSM column: {targets_file}")
        return {row["GSM"] for row in reader if row.get("GSM")}


def load_metadata(stage_filter: Iterable[str] | None, target_gsms: set[str] | None) -> List[dict]:
    with METADATA_FILE.open() as handle:
        rows = list(csv.DictReader(handle))

    if stage_filter:
        allowed = set(stage_filter)
        rows = [row for row in rows if row["CELL_STAGE"] in allowed]
    if target_gsms is not None:
        rows = [row for row in rows if row["GSM"] in target_gsms]

    return rows


def contact_path(row: dict) -> Path:
    return SOURCE_ROOT / row["CELL_STAGE"] / f'{row["GSM"]}_{row["CELL_ID"]}.txt'


def resolution_label(resolution: int) -> str:
    if resolution % 1000 != 0:
        raise ValueError(f"Resolution is not an integer kb value: {resolution}")
    return str(resolution // 1000)


def output_root_for_resolution(resolution: int) -> Path:
    return OUTPUT_ROOT / resolution_label(resolution)


def run_summary_path(output_root: Path) -> Path:
    return output_root / "run_summary.tsv"


def resolution_log_paths(output_root: Path) -> Dict[str, Path]:
    return {
        "log": output_root / "full_run.log",
    }


def output_paths(row: dict, output_root: Path, resolution: int) -> Dict[str, Path]:
    cell_dir = output_root / row["CELL_STAGE"] / row["GSM"]
    return {
        "cell_dir": cell_dir,
        "mcool": cell_dir / f'{row["GSM"]}.mcool',
        "boundary_prefix": cell_dir / "robustad_boundaries",
        "boundary_bed": cell_dir / "robustad_boundaries.bed",
        "tad_prefix": cell_dir / "robustad_tads",
        "tad_hq": cell_dir / "robustad_tads_hq.bed",
        "tad_all": cell_dir / "robustad_tads_all.bed",
        "manifest": cell_dir / "cell_metadata.tsv",
        "stage_dir": output_root / row["CELL_STAGE"],
        "panel": output_root / row["CELL_STAGE"] / f"robustad_reference_panel_{resolution}.pkl",
        "boundary_list": output_root / row["CELL_STAGE"] / "boundary_score_files.txt",
    }


def build_bins(resolution: int) -> Tuple[pd.DataFrame, Dict[str, int]]:
    chroms = []
    starts = []
    ends = []
    offsets = {}
    running = 0

    for chrom in VALID_CHROMS:
        offsets[chrom] = running
        size = MM10_CHROM_SIZES[chrom]
        chrom_starts = list(range(0, size, resolution))
        chrom_ends = [min(start + resolution, size) for start in chrom_starts]
        chroms.extend([chrom] * len(chrom_starts))
        starts.extend(chrom_starts)
        ends.extend(chrom_ends)
        running += len(chrom_starts)

    bins = pd.DataFrame({"chrom": chroms, "start": starts, "end": ends})
    return bins, offsets


def chunked_contact_counts(
    contact_file: Path, resolution: int, offsets: Dict[str, int], chunksize: int
) -> Dict[Tuple[int, int], int]:
    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    columns = ["chr1", "pos1", "chr2", "pos2"]

    reader = pd.read_csv(
        contact_file,
        sep="\t",
        header=None,
        usecols=[1, 2, 4, 5],
        names=columns,
        dtype={"chr1": "string", "pos1": np.int64, "chr2": "string", "pos2": np.int64},
        chunksize=chunksize,
    )

    for chunk in reader:
        chunk = chunk[
            chunk["chr1"].isin(VALID_CHROMS)
            & chunk["chr2"].isin(VALID_CHROMS)
            & (chunk["chr1"] == chunk["chr2"])
        ]
        if chunk.empty:
            continue

        chunk["bin1_id"] = (
            chunk["chr1"].map(offsets).astype(np.int64)
            + (chunk["pos1"] // resolution).astype(np.int64)
        )
        chunk["bin2_id"] = (
            chunk["chr2"].map(offsets).astype(np.int64)
            + (chunk["pos2"] // resolution).astype(np.int64)
        )

        low = np.minimum(chunk["bin1_id"].to_numpy(), chunk["bin2_id"].to_numpy())
        high = np.maximum(chunk["bin1_id"].to_numpy(), chunk["bin2_id"].to_numpy())
        pair_df = pd.DataFrame({"bin1_id": low, "bin2_id": high})
        grouped = pair_df.groupby(["bin1_id", "bin2_id"], sort=False).size()
        for (bin1_id, bin2_id), count in grouped.items():
            counts[(int(bin1_id), int(bin2_id))] += int(count)

    return counts


def counts_to_pixels(counts: Dict[Tuple[int, int], int]) -> pd.DataFrame:
    if not counts:
        return pd.DataFrame({"bin1_id": [], "bin2_id": [], "count": []})

    records = [
        {"bin1_id": bin1_id, "bin2_id": bin2_id, "count": count}
        for (bin1_id, bin2_id), count in counts.items()
    ]
    pixels = pd.DataFrame.from_records(records)
    pixels.sort_values(["bin1_id", "bin2_id"], inplace=True, ignore_index=True)
    return pixels


def manifest_rows(
    row: dict,
    source_file: Path,
    resolution: int,
    boundary_mode: str,
    panel_file: Path | None,
) -> List[Tuple[str, str]]:
    return [
        ("gsm", row["GSM"]),
        ("stage", row["CELL_STAGE"]),
        ("cell_id", row["CELL_ID"]),
        ("source_contact_file", str(source_file)),
        ("resolution_bp", str(resolution)),
        ("boundary_mode", boundary_mode),
        ("panel_file", str(panel_file) if panel_file is not None else ""),
    ]


def ensure_manifest(
    row: dict,
    paths: Dict[str, Path],
    source_file: Path,
    resolution: int,
    boundary_mode: str,
    panel_file: Path | None,
) -> None:
    text = "\n".join(f"{key}\t{value}" for key, value in manifest_rows(row, source_file, resolution, boundary_mode, panel_file)) + "\n"
    paths["manifest"].write_text(text)


def load_manifest(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    values: Dict[str, str] = {}
    with path.open() as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                values[parts[0]] = parts[1]
    return values


def outputs_match_request(
    row: dict,
    paths: Dict[str, Path],
    source_file: Path,
    resolution: int,
    boundary_mode: str,
    panel_file: Path | None,
    require_tads: bool,
) -> bool:
    required = [paths["boundary_bed"], paths["manifest"]]
    if require_tads:
        required.extend([paths["tad_hq"], paths["tad_all"]])
    if not all(path.exists() for path in required):
        return False

    manifest = load_manifest(paths["manifest"])
    expected = dict(manifest_rows(row, source_file, resolution, boundary_mode, panel_file))
    for key, value in expected.items():
        if manifest.get(key, "") != value:
            return False
    return True


def create_balanced_mcool(
    cooler_module,
    create_cooler_fn,
    balance_cooler_fn,
    source_file: Path,
    mcool_path: Path,
    bins: pd.DataFrame,
    offsets: Dict[str, int],
    resolution: int,
    chunksize: int,
    force: bool,
) -> None:
    uri = f"{mcool_path}::/resolutions/{resolution}"
    if mcool_path.exists() and not force:
        logging.info("Reuse existing mcool: %s", mcool_path)
        return

    if mcool_path.exists():
        mcool_path.unlink()

    counts = chunked_contact_counts(source_file, resolution, offsets, chunksize)
    pixels = counts_to_pixels(counts)
    if pixels.empty:
        raise RuntimeError(f"No cis contacts survived filtering in {source_file}")

    create_cooler_fn(
        cool_uri=uri,
        bins=bins,
        pixels=pixels,
        ordered=True,
        symmetric_upper=True,
        dtypes={"count": np.int32},
    )

    clr = cooler_module.Cooler(uri)
    balance_cooler_fn(clr, cis_only=True, store=True, chunksize=max(chunksize, 5_000_000))


def run_robustad_for_cell(
    boundary_cmd,
    lmcc_cmd,
    assembly_cmd,
    mcool_path: Path,
    paths: Dict[str, Path],
    resolution: int,
    panel_file: Path | None,
    run_assembly: bool,
) -> None:
    if panel_file is None:
        boundary_cmd.callback(
            coolfile=str(mcool_path),
            prefix=str(paths["boundary_prefix"]),
            resol=resolution,
            minw=max(100, resolution // 1000),
            maxw=3000,
            alpha=0.05,
            ratio=1.2,
            chr=None,
            mind=max(100, resolution // 1000),
        )
    else:
        lmcc_cmd.callback(
            coolfile=str(mcool_path),
            prefix=str(paths["boundary_prefix"]),
            alpha=0.05,
            chr=None,
            ratio=1.2,
            resol=resolution,
            minw=max(100, resolution // 1000),
            maxw=3000,
            ds=-1,
            mind=max(100, resolution // 1000),
            panel=str(panel_file),
        )

    if not run_assembly:
        return

    assembly_cmd.callback(
        delta=0.1,
        ratio=1.2,
        resol=resolution,
        maxtad=3_000_000,
        mintad=resolution,
        boundaryfile=str(paths["boundary_bed"]),
        coolfile=str(mcool_path),
        prefix=str(paths["tad_prefix"]),
        chr=None,
    )


def process_cell(
    row: dict,
    output_root: Path,
    bins: pd.DataFrame,
    offsets: Dict[str, int],
    resolution: int,
    chunksize: int,
    force: bool,
    keep_mcool: bool,
    cooler_module,
    create_cooler_fn,
    balance_cooler_fn,
    boundary_cmd,
    lmcc_cmd,
    assembly_cmd,
    panel_file: Path | None,
    run_assembly: bool,
) -> Dict[str, Path]:
    source_file = contact_path(row)
    paths = output_paths(row, output_root=output_root, resolution=resolution)
    paths["cell_dir"].mkdir(parents=True, exist_ok=True)

    if not source_file.exists():
        raise FileNotFoundError(f"Missing source file: {source_file}")

    boundary_mode = "lmcc" if panel_file is not None else "boundary"
    if not force and not run_assembly and paths["boundary_bed"].exists():
        logging.info("Reuse existing boundary scores: %s", paths["boundary_bed"])
        return paths

    if not force and outputs_match_request(
        row=row,
        paths=paths,
        source_file=source_file,
        resolution=resolution,
        boundary_mode=boundary_mode,
        panel_file=panel_file,
        require_tads=run_assembly,
    ):
        logging.info("Reuse existing RobusTAD outputs: %s", paths["cell_dir"])
        if paths["mcool"].exists() and not keep_mcool:
            paths["mcool"].unlink()
        return paths

    create_balanced_mcool(
        cooler_module=cooler_module,
        create_cooler_fn=create_cooler_fn,
        balance_cooler_fn=balance_cooler_fn,
        source_file=source_file,
        mcool_path=paths["mcool"],
        bins=bins,
        offsets=offsets,
        resolution=resolution,
        chunksize=chunksize,
        force=force,
    )
    run_robustad_for_cell(
        boundary_cmd,
        lmcc_cmd,
        assembly_cmd,
        paths["mcool"],
        paths,
        resolution,
        panel_file,
        run_assembly,
    )
    ensure_manifest(row, paths, source_file, resolution, boundary_mode, panel_file)
    if paths["mcool"].exists() and not keep_mcool:
        paths["mcool"].unlink()
    return paths


def build_stage_reference_panel(
    output_root: Path,
    stage: str,
    boundary_files: List[Path],
    createdatabase_cmd,
    resolution: int,
    force: bool,
) -> None:
    stage_dir = output_root / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    list_file = stage_dir / "boundary_score_files.txt"
    panel_file = stage_dir / f"robustad_reference_panel_{resolution}.pkl"

    list_file.write_text("\n".join(str(path) for path in sorted(boundary_files)) + "\n")
    if panel_file.exists() and not force:
        logging.info("Reuse existing stage panel: %s", panel_file)
        return

    createdatabase_cmd.callback(
        resol=resolution,
        output=str(panel_file),
        scores=str(list_file),
    )


def process_stage_with_panels(
    rows: List[dict],
    output_root: Path,
    bins: pd.DataFrame,
    offsets: Dict[str, int],
    resolution: int,
    chunksize: int,
    force: bool,
    keep_mcool: bool,
    cooler_module,
    create_cooler_fn,
    balance_cooler_fn,
    boundary_cmd,
    lmcc_cmd,
    assembly_cmd,
    createdatabase_cmd,
    panels_only: bool,
) -> List[Tuple[str, str]]:
    failures: List[Tuple[str, str]] = []

    grouped_rows: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped_rows[row["CELL_STAGE"]].append(row)

    for stage in STAGES:
        stage_rows = grouped_rows.get(stage, [])
        if not stage_rows:
            continue
        logging.info("Preparing reference panel for %s at %s kb", stage, resolution_label(resolution))
        boundary_files: List[Path] = []
        panel_file = output_root / stage / f"robustad_reference_panel_{resolution}.pkl"

        for row in stage_rows:
            label = f'{row["CELL_STAGE"]} {row["GSM"]} (panel pass)'
            try:
                paths = process_cell(
                    row=row,
                    output_root=output_root,
                    bins=bins,
                    offsets=offsets,
                    resolution=resolution,
                    chunksize=chunksize,
                    force=force,
                    keep_mcool=keep_mcool,
                    cooler_module=cooler_module,
                    create_cooler_fn=create_cooler_fn,
                    balance_cooler_fn=balance_cooler_fn,
                    boundary_cmd=boundary_cmd,
                    lmcc_cmd=lmcc_cmd,
                    assembly_cmd=assembly_cmd,
                    panel_file=None,
                    run_assembly=False,
                )
                boundary_files.append(paths["boundary_bed"])
            except Exception as exc:
                logging.exception("Failed on %s", label)
                failures.append((row["GSM"], str(exc)))

        successful_rows = [row for row in stage_rows if row["GSM"] not in {gsm for gsm, _ in failures}]
        if not boundary_files:
            continue

        try:
            build_stage_reference_panel(
                output_root=output_root,
                stage=stage,
                boundary_files=boundary_files,
                createdatabase_cmd=createdatabase_cmd,
                resolution=resolution,
                force=force,
            )
        except Exception as exc:
            logging.exception("Failed to create reference panel for %s", stage)
            for row in successful_rows:
                failures.append((row["GSM"], f"panel creation failed for {stage}: {exc}"))
            continue

        if panels_only:
            continue

        for row in successful_rows:
            label = f'{row["CELL_STAGE"]} {row["GSM"]} (LMCC pass)'
            try:
                process_cell(
                    row=row,
                    output_root=output_root,
                    bins=bins,
                    offsets=offsets,
                    resolution=resolution,
                    chunksize=chunksize,
                    force=force,
                    keep_mcool=keep_mcool,
                    cooler_module=cooler_module,
                    create_cooler_fn=create_cooler_fn,
                    balance_cooler_fn=balance_cooler_fn,
                    boundary_cmd=boundary_cmd,
                    lmcc_cmd=lmcc_cmd,
                    assembly_cmd=assembly_cmd,
                    panel_file=panel_file,
                    run_assembly=True,
                )
            except Exception as exc:
                logging.exception("Failed on %s", label)
                failures.append((row["GSM"], str(exc)))

    return failures


def process_without_panels(
    rows: List[dict],
    output_root: Path,
    bins: pd.DataFrame,
    offsets: Dict[str, int],
    resolution: int,
    chunksize: int,
    force: bool,
    keep_mcool: bool,
    cooler_module,
    create_cooler_fn,
    balance_cooler_fn,
    boundary_cmd,
    lmcc_cmd,
    assembly_cmd,
) -> List[Tuple[str, str]]:
    failures: List[Tuple[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        label = f'{row["CELL_STAGE"]} {row["GSM"]} ({idx}/{len(rows)})'
        logging.info("Processing %s", label)
        try:
            process_cell(
                row=row,
                output_root=output_root,
                bins=bins,
                offsets=offsets,
                resolution=resolution,
                chunksize=chunksize,
                force=force,
                keep_mcool=keep_mcool,
                cooler_module=cooler_module,
                create_cooler_fn=create_cooler_fn,
                balance_cooler_fn=balance_cooler_fn,
                boundary_cmd=boundary_cmd,
                lmcc_cmd=lmcc_cmd,
                assembly_cmd=assembly_cmd,
                panel_file=None,
                run_assembly=True,
            )
        except Exception as exc:
            logging.exception("Failed on %s", label)
            failures.append((row["GSM"], str(exc)))
    return failures


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    cooler_module, create_cooler_fn, balance_cooler_fn, boundary_cmd, lmcc_cmd, assembly_cmd, createdatabase_cmd = (
        bootstrap_imports()
    )

    target_gsms = load_target_gsms(args.targets_file)
    rows = load_metadata(args.stage, target_gsms)
    if args.limit is not None:
        rows = rows[: args.limit]

    logging.info("Metadata rows selected: %d", len(rows))
    if not rows:
        logging.error("No cells selected after applying filters.")
        return 1

    exit_code = 0
    for resolution in selected_resolutions(args):
        output_root = output_root_for_resolution(resolution)
        output_root.mkdir(parents=True, exist_ok=True)
        logging.info("Processing resolution %s kb into %s", resolution_label(resolution), output_root)
        bins, offsets = build_bins(resolution)

        if args.skip_panels:
            failures = process_without_panels(
                rows=rows,
                output_root=output_root,
                bins=bins,
                offsets=offsets,
                resolution=resolution,
                chunksize=args.chunksize,
                force=args.force,
                keep_mcool=args.keep_mcool,
                cooler_module=cooler_module,
                create_cooler_fn=create_cooler_fn,
                balance_cooler_fn=balance_cooler_fn,
                boundary_cmd=boundary_cmd,
                lmcc_cmd=lmcc_cmd,
                assembly_cmd=assembly_cmd,
            )
        else:
            failures = process_stage_with_panels(
                rows=rows,
                output_root=output_root,
                bins=bins,
                offsets=offsets,
                resolution=resolution,
                chunksize=args.chunksize,
                force=args.force,
                keep_mcool=args.keep_mcool,
                cooler_module=cooler_module,
                create_cooler_fn=create_cooler_fn,
                balance_cooler_fn=balance_cooler_fn,
                boundary_cmd=boundary_cmd,
                lmcc_cmd=lmcc_cmd,
                assembly_cmd=assembly_cmd,
                createdatabase_cmd=createdatabase_cmd,
                panels_only=args.panels_only,
            )

        summary_file = run_summary_path(output_root)
        summary_lines = ["gsm\tstatus\terror"]
        failed = {gsm: error for gsm, error in failures}
        for row in rows:
            gsm = row["GSM"]
            summary_lines.append(f"{gsm}\t{'failed' if gsm in failed else 'ok'}\t{failed.get(gsm, '')}")
        summary_file.write_text("\n".join(summary_lines) + "\n")

        logging.info(
            "Completed resolution %s kb with %d failures. Summary: %s",
            resolution_label(resolution),
            len(failures),
            summary_file,
        )
        if failures:
            exit_code = 2

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
