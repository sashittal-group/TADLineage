#!/usr/bin/env python3
"""Run RobusTAD on full per-cell Hi-C matrices and prepare all-chromosome TAD files."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_robustad_panels import (  # noqa: E402
    MM10_CHROM_SIZES,
    STAGES,
    bootstrap_imports,
    build_bins,
    build_stage_reference_panel,
    contact_path,
    create_balanced_mcool,
    load_metadata,
    load_target_gsms,
    resolution_label,
    run_robustad_for_cell,
)


CHROMS = list(MM10_CHROM_SIZES.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolution",
        dest="resolutions",
        action="append",
        type=int,
        default=None,
        help="Resolution in bp. Repeat for multiple resolutions. Default: 1000000",
    )
    parser.add_argument(
        "--work-base-dir",
        type=Path,
        default=SCRIPT_DIR / "results_updated" / "scptr_merged_all_chr",
        help="Directory to update with all-chromosome RobustAD outputs.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=STAGES,
        help="Optional stage filter. Repeat to process multiple stages.",
    )
    parser.add_argument(
        "--targets-file",
        type=Path,
        default=None,
        help="Optional CSV/TSV containing a GSM column.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cell limit for testing.")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--force", action="store_true", help="Rebuild existing outputs.")
    parser.add_argument("--keep-mcool", action="store_true", help="Keep generated .mcool files.")
    parser.add_argument(
        "--skip-panels",
        action="store_true",
        help="Run RobustAD boundary/assembly directly without stage reference panels.",
    )
    parser.add_argument(
        "--panels-only",
        action="store_true",
        help="Build panel boundary scores and stage panels, but do not run final assembly.",
    )
    parser.add_argument(
        "--top-tads-per-cell",
        type=int,
        default=20,
        help="Keep this many highest-scoring TADs in tads.tad; <=0 keeps all.",
    )
    parser.add_argument(
        "--boundary-alpha",
        type=float,
        default=0.05,
        help=("RobusTAD boundary/LMCC significance cutoff. Larger values are more "
              "permissive; default: 0.05."),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def selected_resolutions(values: Iterable[int] | None) -> List[int]:
    resolutions = list(values) if values else [1_000_000]
    unique = []
    seen = set()
    for resolution in resolutions:
        if resolution <= 0:
            raise ValueError(f"Resolution must be positive: {resolution}")
        if resolution not in seen:
            unique.append(resolution)
            seen.add(resolution)
    return unique


def resolution_dir(resolution: int) -> str:
    return f"{resolution // 1000}kb" if resolution % 1000 == 0 else str(resolution)


def full_matrix_cell_dir(work_base_dir: Path, resolution: int, stage: str, gsm: str) -> Path:
    label = resolution_dir(resolution)
    return work_base_dir / label / stage / gsm / "all" / label


def robustad_paths(work_base_dir: Path, row: dict, resolution: int) -> Dict[str, Path]:
    cell_dir = full_matrix_cell_dir(work_base_dir, resolution, row["CELL_STAGE"], row["GSM"])
    return {
        "cell_dir": cell_dir,
        "mcool": cell_dir / f'{row["GSM"]}.full_matrix.mcool',
        "boundary_prefix": cell_dir / "robustad_boundaries",
        "boundary_bed": cell_dir / "robustad_boundaries.bed",
        "tad_prefix": cell_dir / "robustad_tads",
        "tad_hq": cell_dir / "robustad_tads_hq.bed",
        "tad_all": cell_dir / "robustad_tads_all.bed",
        "tads_tad": cell_dir / "tads.tad",
        "boundaries_bed": cell_dir / "boundaries.bed",
        "manifest": cell_dir / "full_matrix_robustad_metadata.tsv",
    }


def stage_panel_path(work_base_dir: Path, resolution: int, stage: str) -> Path:
    return work_base_dir / resolution_dir(resolution) / stage / f"full_matrix_reference_panel_{resolution}.pkl"


def stage_boundary_list_path(work_base_dir: Path, resolution: int, stage: str) -> Path:
    return work_base_dir / resolution_dir(resolution) / stage / "full_matrix_boundary_score_files.txt"


def global_offsets(resolution: int) -> Dict[str, int]:
    offsets = {}
    cursor = 0
    for chrom in CHROMS:
        offsets[chrom] = cursor
        cursor += int(math.ceil(MM10_CHROM_SIZES[chrom] / resolution)) + 1
    return offsets


def encode_position(chrom: str, position_bp: str, offsets: Dict[str, int], resolution: int) -> int:
    return offsets[chrom] + int(float(position_bp)) // resolution


def write_encoded_tads(
    robustad_tads: Path,
    output_tads: Path,
    offsets: Dict[str, int],
    resolution: int,
    top_tads_per_cell: int,
) -> int:
    records = []
    with robustad_tads.open() as handle:
        for line in handle:
            cols = line.strip().split()
            if len(cols) < 4 or cols[0] == "chrom":
                continue
            chrom = cols[0]
            if chrom not in offsets:
                continue
            try:
                start = encode_position(chrom, cols[1], offsets, resolution)
                end = encode_position(chrom, cols[2], offsets, resolution)
                score = float(cols[3])
                nested_score = float(cols[4]) if len(cols) > 4 else score
                hq = float(cols[5]) if len(cols) > 5 else 1.0
                level = float(cols[6]) if len(cols) > 6 else 0.0
            except Exception:
                continue
            if end > start:
                records.append((start, end, level, score, nested_score, hq))

    records.sort(key=lambda item: item[3], reverse=True)
    if top_tads_per_cell and top_tads_per_cell > 0:
        records = records[:top_tads_per_cell]

    output_tads.parent.mkdir(parents=True, exist_ok=True)
    with output_tads.open("w") as handle:
        for row in records:
            handle.write("\t".join(str(value) for value in row) + "\n")
    return len(records)


def write_encoded_boundaries(
    robustad_boundaries: Path,
    output_boundaries: Path,
    offsets: Dict[str, int],
    resolution: int,
) -> int:
    count = 0
    output_boundaries.parent.mkdir(parents=True, exist_ok=True)
    with robustad_boundaries.open() as source, output_boundaries.open("w") as out:
        for line in source:
            cols = line.strip().split()
            if len(cols) < 6:
                continue
            chrom = cols[0]
            if chrom not in offsets:
                continue
            try:
                start = encode_position(chrom, cols[1], offsets, resolution)
                end = encode_position(chrom, cols[2], offsets, resolution)
                values = [start, end] + cols[3:]
            except Exception:
                continue
            out.write("\t".join(str(value) for value in values) + "\n")
            count += 1
    return count


def write_manifest(
    path: Path,
    row: dict,
    source_file: Path,
    resolution: int,
    boundary_mode: str,
    panel_file: Path | None,
    tads_written: int,
    boundaries_written: int,
    boundary_alpha: float,
) -> None:
    values = [
        ("gsm", row["GSM"]),
        ("stage", row["CELL_STAGE"]),
        ("cell_id", row["CELL_ID"]),
        ("source_contact_file", str(source_file)),
        ("resolution_bp", str(resolution)),
        ("input_scope", "full_cell_hic_matrix_all_mm10_chromosomes"),
        ("boundary_mode", boundary_mode),
        ("boundary_alpha", str(boundary_alpha)),
        ("panel_file", str(panel_file) if panel_file is not None else ""),
        ("tads_written", str(tads_written)),
        ("boundaries_written", str(boundaries_written)),
    ]
    path.write_text("\n".join(f"{key}\t{value}" for key, value in values) + "\n")


def outputs_ready(paths: Dict[str, Path], force: bool) -> bool:
    if force:
        return False
    return paths["tads_tad"].exists() and paths["boundaries_bed"].exists() and paths["manifest"].exists()


def convert_robustad_outputs(
    paths: Dict[str, Path],
    row: dict,
    source_file: Path,
    resolution: int,
    panel_file: Path | None,
    top_tads_per_cell: int,
    boundary_alpha: float,
) -> None:
    offsets = global_offsets(resolution)
    tads_written = write_encoded_tads(
        paths["tad_all"],
        paths["tads_tad"],
        offsets,
        resolution,
        top_tads_per_cell,
    )
    boundaries_written = write_encoded_boundaries(
        paths["boundary_bed"],
        paths["boundaries_bed"],
        offsets,
        resolution,
    )
    write_manifest(
        paths["manifest"],
        row,
        source_file,
        resolution,
        "lmcc" if panel_file is not None else "boundary",
        panel_file,
        tads_written,
        boundaries_written,
        boundary_alpha,
    )


def run_cell(
    row: dict,
    work_base_dir: Path,
    bins,
    offsets,
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
    top_tads_per_cell: int,
    boundary_alpha: float,
) -> Dict[str, Path]:
    paths = robustad_paths(work_base_dir, row, resolution)
    paths["cell_dir"].mkdir(parents=True, exist_ok=True)
    source_file = contact_path(row)
    if not source_file.exists():
        raise FileNotFoundError(f"Missing source contact file: {source_file}")

    if run_assembly and outputs_ready(paths, force):
        logging.info("Reuse full-matrix outputs: %s", paths["cell_dir"])
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
        boundary_cmd=boundary_cmd,
        lmcc_cmd=lmcc_cmd,
        assembly_cmd=assembly_cmd,
        mcool_path=paths["mcool"],
        paths=paths,
        resolution=resolution,
        panel_file=panel_file,
        run_assembly=run_assembly,
        boundary_alpha=boundary_alpha,
    )

    if run_assembly:
        convert_robustad_outputs(
            paths,
            row,
            source_file,
            resolution,
            panel_file,
            top_tads_per_cell,
            boundary_alpha,
        )

    if paths["mcool"].exists() and not keep_mcool:
        paths["mcool"].unlink()
    return paths


def build_full_matrix_stage_panel(
    work_base_dir: Path,
    resolution: int,
    stage: str,
    boundary_files: List[Path],
    createdatabase_cmd,
    force: bool,
) -> Path:
    list_file = stage_boundary_list_path(work_base_dir, resolution, stage)
    panel_file = stage_panel_path(work_base_dir, resolution, stage)
    list_file.parent.mkdir(parents=True, exist_ok=True)
    list_file.write_text("\n".join(str(path) for path in sorted(boundary_files)) + "\n")
    if panel_file.exists() and not force:
        logging.info("Reuse full-matrix stage panel: %s", panel_file)
        return panel_file
    build_stage_reference_panel(
        output_root=work_base_dir / resolution_dir(resolution),
        stage=stage,
        boundary_files=boundary_files,
        createdatabase_cmd=createdatabase_cmd,
        resolution=resolution,
        force=force,
    )
    default_panel = work_base_dir / resolution_dir(resolution) / stage / f"robustad_reference_panel_{resolution}.pkl"
    if default_panel.exists() and default_panel != panel_file:
        default_panel.replace(panel_file)
    return panel_file


def process_resolution(args: argparse.Namespace, rows: List[dict], resolution: int, tools: Tuple[object, ...]) -> List[Tuple[str, str]]:
    (
        cooler_module,
        create_cooler_fn,
        balance_cooler_fn,
        boundary_cmd,
        lmcc_cmd,
        assembly_cmd,
        createdatabase_cmd,
    ) = tools

    bins, bin_offsets = build_bins(resolution)
    failures: List[Tuple[str, str]] = []
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["CELL_STAGE"]].append(row)

    if args.skip_panels:
        ordered_rows = [row for stage in STAGES for row in grouped.get(stage, [])]
        for idx, row in enumerate(ordered_rows, start=1):
            logging.info(
                "Full-matrix RobustAD %s %s (%d/%d)",
                row["CELL_STAGE"],
                row["GSM"],
                idx,
                len(ordered_rows),
            )
            try:
                run_cell(
                    row,
                    args.work_base_dir,
                    bins,
                    bin_offsets,
                    resolution,
                    args.chunksize,
                    args.force,
                    args.keep_mcool,
                    cooler_module,
                    create_cooler_fn,
                    balance_cooler_fn,
                    boundary_cmd,
                    lmcc_cmd,
                    assembly_cmd,
                    panel_file=None,
                    run_assembly=True,
                    top_tads_per_cell=args.top_tads_per_cell,
                    boundary_alpha=args.boundary_alpha,
                )
            except Exception as exc:
                logging.exception("Failed on %s", row["GSM"])
                failures.append((row["GSM"], str(exc)))
        return failures

    for stage in STAGES:
        stage_rows = grouped.get(stage, [])
        if not stage_rows:
            continue

        boundary_files: List[Path] = []
        logging.info("Building full-matrix boundary scores for %s at %s", stage, resolution_dir(resolution))
        for row in stage_rows:
            try:
                paths = run_cell(
                    row,
                    args.work_base_dir,
                    bins,
                    bin_offsets,
                    resolution,
                    args.chunksize,
                    args.force,
                    args.keep_mcool,
                    cooler_module,
                    create_cooler_fn,
                    balance_cooler_fn,
                    boundary_cmd,
                    lmcc_cmd,
                    assembly_cmd,
                    panel_file=None,
                    run_assembly=False,
                    top_tads_per_cell=args.top_tads_per_cell,
                    boundary_alpha=args.boundary_alpha,
                )
                boundary_files.append(paths["boundary_bed"])
            except Exception as exc:
                logging.exception("Failed boundary pass on %s", row["GSM"])
                failures.append((row["GSM"], str(exc)))

        successful_rows = [row for row in stage_rows if row["GSM"] not in {gsm for gsm, _ in failures}]
        if not boundary_files:
            continue

        try:
            panel_file = build_full_matrix_stage_panel(
                args.work_base_dir,
                resolution,
                stage,
                boundary_files,
                createdatabase_cmd,
                args.force,
            )
        except Exception as exc:
            logging.exception("Failed panel creation for %s", stage)
            for row in successful_rows:
                failures.append((row["GSM"], f"panel creation failed: {exc}"))
            continue

        if args.panels_only:
            continue

        for row in successful_rows:
            logging.info("Running full-matrix LMCC/assembly for %s %s", stage, row["GSM"])
            try:
                run_cell(
                    row,
                    args.work_base_dir,
                    bins,
                    bin_offsets,
                    resolution,
                    args.chunksize,
                    args.force,
                    args.keep_mcool,
                    cooler_module,
                    create_cooler_fn,
                    balance_cooler_fn,
                    boundary_cmd,
                    lmcc_cmd,
                    assembly_cmd,
                    panel_file=panel_file,
                    run_assembly=True,
                    top_tads_per_cell=args.top_tads_per_cell,
                    boundary_alpha=args.boundary_alpha,
                )
            except Exception as exc:
                logging.exception("Failed LMCC pass on %s", row["GSM"])
                failures.append((row["GSM"], str(exc)))

    return failures


def write_summary(work_base_dir: Path, resolution: int, rows: List[dict], failures: List[Tuple[str, str]]) -> None:
    summary_path = work_base_dir / resolution_dir(resolution) / "full_matrix_robustad_summary.tsv"
    failed = {gsm: error for gsm, error in failures}
    lines = ["gsm\tstage\tstatus\terror"]
    for row in rows:
        gsm = row["GSM"]
        lines.append(f"{gsm}\t{row['CELL_STAGE']}\t{'failed' if gsm in failed else 'ok'}\t{failed.get(gsm, '')}")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n")
    logging.info("Summary written: %s", summary_path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/robustad_scptr") / "mplconfig"))

    target_gsms = load_target_gsms(args.targets_file)
    rows = load_metadata(args.stage, target_gsms)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        logging.error("No cells selected.")
        return 1

    args.work_base_dir.mkdir(parents=True, exist_ok=True)
    tools = bootstrap_imports()

    exit_code = 0
    for resolution in selected_resolutions(args.resolutions):
        logging.info("Processing full matrices at %s into %s", resolution_dir(resolution), args.work_base_dir)
        failures = process_resolution(args, rows, resolution, tools)
        write_summary(args.work_base_dir, resolution, rows, failures)
        if failures:
            exit_code = 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
