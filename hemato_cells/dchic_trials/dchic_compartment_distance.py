#!/usr/bin/env python3
"""Run core dcHiC for pairwise compartment distances.

For each cell pair and one chromosome/resolution, this script:

1. Converts dense Hi-C matrices into dcHiC sparse matrix + BED input.
2. Runs dcHiC's R workflow: cis -> select -> analyze.
3. Reads dcHiC's differential compartment bedGraph.
4. Writes per-bin compartment differences and one pairwise distance summary.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import shutil
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_HEMATO_DIR = SCRIPT_DIR.parent
DEFAULT_DCHIC = REPO_ROOT / "dcHiC" / "dchicf.r"
DEFAULT_CELLS = ["LT", "ST", "MPP", "CMP", "CLP", "MEP", "GMP", "MKP", "G", "MK"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dcHiC and summarize pairwise compartment distances.")
    parser.add_argument("--chrom", required=True, help="Chromosome, e.g. chr19")
    parser.add_argument("--resolution", required=True, help="Resolution label, e.g. 1mb")
    parser.add_argument("--hemato-dir", type=Path, default=DEFAULT_HEMATO_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-bin-dir", type=Path)
    parser.add_argument("--work-dir", type=Path, default=SCRIPT_DIR / "dchic_runs")
    parser.add_argument("--cells", nargs="+", default=DEFAULT_CELLS)
    parser.add_argument("--matrix-name", default="hic_matrix.cleaned.txt")
    parser.add_argument("--matrix1", type=Path, help="Optional direct matrix path for one pair.")
    parser.add_argument("--matrix2", type=Path, help="Optional direct matrix path for one pair.")
    parser.add_argument("--cell1", help="Cell/sample name for --matrix1.")
    parser.add_argument("--cell2", help="Cell/sample name for --matrix2.")
    parser.add_argument("--dchic-script", type=Path, default=DEFAULT_DCHIC)
    parser.add_argument("--genome", default="mm10")
    parser.add_argument("--gfolder", type=Path, help="dcHiC goldenPath folder for select step.")
    parser.add_argument("--conda-env", help="Optional conda env used to run Rscript, e.g. dchic-lite.")
    parser.add_argument("--pc", type=int, default=3)
    parser.add_argument("--minc", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Remove existing pair work dirs before rerun.")
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep dcHiC pair working directories for debugging. By default they are deleted after CSV extraction.",
    )
    return parser.parse_args()


def resolution_bp(label: str) -> int:
    text = str(label).strip().lower()
    if text.endswith("kb"):
        return int(float(text[:-2]) * 1_000)
    if text.endswith("mb"):
        return int(float(text[:-2]) * 1_000_000)
    if text.endswith("bp"):
        return int(float(text[:-2]))
    return int(float(text))


def chrom_size(chrom: str) -> int:
    sizes = REPO_ROOT / "genome.chrom.sizes"
    with sizes.open() as handle:
        for line in handle:
            name, size = line.rstrip("\n").split("\t")[:2]
            if name == chrom:
                return int(size)
    raise ValueError(f"{chrom} not found in {sizes}")


def read_matrix(path: Path) -> list[list[float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append([float(value) for value in stripped.split()])
    if not rows or any(len(row) != len(rows) for row in rows):
        width = len(rows[0]) if rows else 0
        raise ValueError(f"{path} is not square: {len(rows)}x{width}")
    return rows


def write_bed(path: Path, chrom: str, n_bins: int, res_bp: int) -> None:
    size = chrom_size(chrom)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for idx in range(n_bins):
            start = idx * res_bp
            end = min(start + res_bp, size)
            writer.writerow([chrom, start, end, idx + 1])


def write_sparse_upper(path: Path, matrix: list[list[float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        n = len(matrix)
        for i in range(n):
            for j in range(i, n):
                value = matrix[i][j]
                if value != 0 and math.isfinite(value):
                    writer.writerow([i + 1, j + 1, f"{value:.8g}"])


def matrix_path(hemato_dir: Path, cell: str, chrom: str, resolution: str, matrix_name: str) -> Path:
    return hemato_dir / cell / chrom / resolution / matrix_name


def rscript_cmd(args: argparse.Namespace) -> list[str]:
    if args.conda_env:
        return ["conda", "run", "-n", args.conda_env, "Rscript"]
    return ["Rscript"]


def run_cmd(command: list[str], cwd: Path) -> None:
    print("Running:", " ".join(str(part) for part in command))
    subprocess.run(command, cwd=cwd, check=True)


def goldenpath_name(genome: str, res_bp: int) -> str:
    return f"{genome}_{int(res_bp)}_goldenpathData"


def goldenpath_ready(folder: Path, genome: str) -> bool:
    return (
        folder.exists()
        and (folder / f"{genome}.chrom.sizes").exists()
        and (folder / f"{genome}.refGene.gtf.gz").exists()
        and ((folder / f"{genome}.fa").exists() or (folder / f"{genome}.fa.gz").exists())
    )


def link_goldenpath_for_analyze(run_dir: Path, shared_gfolder: Path, genome: str, res_bp: int) -> None:
    local_folder = run_dir / goldenpath_name(genome, res_bp)
    if local_folder.exists() or local_folder.is_symlink():
        if local_folder.resolve() == shared_gfolder.resolve():
            return
        if local_folder.is_symlink():
            local_folder.unlink()
        else:
            shutil.rmtree(local_folder)
    local_folder.symlink_to(shared_gfolder.resolve(), target_is_directory=True)


def install_shared_goldenpath(run_dir: Path, shared_gfolder: Path, genome: str, res_bp: int) -> None:
    local_folder = run_dir / goldenpath_name(genome, res_bp)
    if goldenpath_ready(shared_gfolder, genome):
        link_goldenpath_for_analyze(run_dir, shared_gfolder, genome, res_bp)
        return
    if not local_folder.exists():
        raise FileNotFoundError(
            f"dcHiC did not create {local_folder}, and shared goldenPath is not ready: {shared_gfolder}"
        )
    shared_gfolder.parent.mkdir(parents=True, exist_ok=True)
    if shared_gfolder.exists():
        shutil.rmtree(local_folder)
    else:
        shutil.move(str(local_folder), str(shared_gfolder))
    link_goldenpath_for_analyze(run_dir, shared_gfolder, genome, res_bp)


def prepare_pair_run(
    run_dir: Path,
    chrom: str,
    res_bp: int,
    cell1: str,
    cell2: str,
    matrix1_path: Path,
    matrix2_path: Path,
    force: bool,
) -> Path:
    if run_dir.exists() and force:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".Rprofile").write_text("options(bigstatsr.ncores.max = 1)\n")

    matrix1 = read_matrix(matrix1_path)
    matrix2 = read_matrix(matrix2_path)
    shape1 = (len(matrix1), len(matrix1[0]))
    shape2 = (len(matrix2), len(matrix2[0]))
    if shape1 != shape2:
        raise ValueError(f"Matrix shapes differ for {cell1}/{cell2}: {shape1} vs {shape2}")

    bed = run_dir / f"{chrom}_{res_bp}.bed"
    mat1 = run_dir / f"{cell1}_{chrom}_{res_bp}.matrix"
    mat2 = run_dir / f"{cell2}_{chrom}_{res_bp}.matrix"
    write_bed(bed, chrom, len(matrix1), res_bp)
    write_sparse_upper(mat1, matrix1)
    write_sparse_upper(mat2, matrix2)

    manifest = run_dir / "input.dchic.txt"
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([mat1.name, bed.name, f"{cell1}_rep1", cell1])
        writer.writerow([mat2.name, bed.name, f"{cell2}_rep1", cell2])
    return manifest


def run_dchic_pair(
    args: argparse.Namespace,
    run_dir: Path,
    manifest: Path,
    cell1: str,
    cell2: str,
    res_bp: int,
    shared_gfolder: Path,
) -> Path:
    base = rscript_cmd(args) + [str(args.dchic_script.resolve())]
    common = ["--file", manifest.name, "--dirovwt", "T"]

    run_cmd(
        base
        + common
        + [
            "--pcatype",
            "cis",
            "--cthread",
            "1",
            "--pthread",
            "1",
            "--sthread",
            "1",
            "--pc",
            str(args.pc),
            "--minc",
            str(args.minc),
        ],
        cwd=run_dir,
    )

    select_cmd = base + common + ["--pcatype", "select", "--genome", args.genome, "--pc", str(args.pc)]
    if goldenpath_ready(shared_gfolder, args.genome):
        select_cmd += ["--gfolder", str(shared_gfolder.resolve())]
    run_cmd(select_cmd, cwd=run_dir)
    install_shared_goldenpath(run_dir, shared_gfolder, args.genome, res_bp)

    diffdir = f"{cell1}_vs_{cell2}_dchic"
    run_cmd(base + common + ["--pcatype", "analyze", "--diffdir", diffdir], cwd=run_dir)
    return run_dir / "DifferentialResult" / diffdir / "fdr_result" / "differential.intra_sample_group.pcQnm.bedGraph"


def read_dchic_bins(path: Path, cell1: str, cell2: str) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            v1 = float(row[cell1])
            v2 = float(row[cell2])
            delta = v2 - v1
            abs_delta = abs(delta)
            rows.append(
                {
                    "chr": row["chr"],
                    "start": row["start"],
                    "end": row["end"],
                    "cell1": cell1,
                    "cell2": cell2,
                    "compartment_score_cell1": v1,
                    "compartment_score_cell2": v2,
                    "signed_difference": delta,
                    "abs_difference": abs_delta,
                    "dchic_sample_maha": float(row["sample_maha"]),
                    "dchic_pval": float(row["pval"]),
                    "dchic_padj": float(row["padj"]),
                    "compartment_switch": int((v1 < 0 < v2) or (v2 < 0 < v1)),
                }
            )
    max_abs = max((float(row["abs_difference"]) for row in rows), default=0.0)
    for row in rows:
        row["aggregate_bin_score"] = float(row["abs_difference"]) / max_abs if max_abs > 0 else 0.0
    return rows


def summarize_pair(rows: list[dict[str, object]], cell1: str, cell2: str) -> dict[str, object]:
    abs_diff = [float(row["abs_difference"]) for row in rows]
    agg = [float(row["aggregate_bin_score"]) for row in rows]
    maha = [float(row["dchic_sample_maha"]) for row in rows]
    switches = [int(row["compartment_switch"]) for row in rows]
    rms = math.sqrt(sum(value * value for value in abs_diff) / len(abs_diff)) if abs_diff else 0.0
    return {
        "cell1": cell1,
        "cell2": cell2,
        "num_bins": len(rows),
        "mean_abs_compartment_diff": f"{sum(abs_diff) / len(abs_diff):.8g}" if abs_diff else "0",
        "sum_abs_compartment_diff": f"{sum(abs_diff):.8g}" if abs_diff else "0",
        "rms_compartment_diff": f"{rms:.8g}" if abs_diff else "0",
        "compartment_switch_count": sum(switches),
        "mean_aggregate_bin_score": f"{sum(agg) / len(agg):.8g}" if agg else "0",
        "sum_aggregate_bin_score": f"{sum(agg):.8g}" if agg else "0",
        "mean_dchic_sample_maha": f"{sum(maha) / len(maha):.8g}" if maha else "0",
        "max_dchic_sample_maha": f"{max(maha):.8g}" if maha else "0",
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def failed_pairs_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.failed_pairs.csv")


def main() -> None:
    args = parse_args()
    res_bp = resolution_bp(args.resolution)
    shared_gfolder = args.gfolder or (SCRIPT_DIR / "goldenpath" / goldenpath_name(args.genome, res_bp))

    if args.matrix1 or args.matrix2:
        if not (args.matrix1 and args.matrix2 and args.cell1 and args.cell2):
            raise ValueError("--matrix1/--matrix2 mode also requires --cell1 and --cell2")
        pairs = [(args.cell1, args.cell2, args.matrix1, args.matrix2)]
    else:
        cells = [
            cell
            for cell in args.cells
            if matrix_path(args.hemato_dir, cell, args.chrom, args.resolution, args.matrix_name).exists()
        ]
        pairs = [
            (
                cell1,
                cell2,
                matrix_path(args.hemato_dir, cell1, args.chrom, args.resolution, args.matrix_name),
                matrix_path(args.hemato_dir, cell2, args.chrom, args.resolution, args.matrix_name),
            )
            for cell1, cell2 in itertools.combinations(cells, 2)
        ]

    summary_rows = []
    failed_rows = []
    for cell1, cell2, path1, path2 in pairs:
        pair_name = f"{cell1}_vs_{cell2}"
        run_dir = args.work_dir / args.chrom / args.resolution / pair_name
        try:
            manifest = prepare_pair_run(run_dir, args.chrom, res_bp, cell1, cell2, path1, path2, force=True)
            dchic_output = run_dchic_pair(args, run_dir, manifest, cell1, cell2, res_bp, shared_gfolder)
            rows = read_dchic_bins(dchic_output, cell1, cell2)
            summary_rows.append(summarize_pair(rows, cell1, cell2))

            if args.per_bin_dir:
                per_bin_path = args.per_bin_dir / f"bin_scores_{args.chrom}_{args.resolution}_{cell1}_vs_{cell2}.csv"
                write_csv(per_bin_path, rows, list(rows[0].keys()) if rows else [])
        except subprocess.CalledProcessError as error:
            failed_rows.append(
                {
                    "chrom": args.chrom,
                    "resolution": args.resolution,
                    "cell1": cell1,
                    "cell2": cell2,
                    "returncode": error.returncode,
                    "failed_command": " ".join(str(part) for part in error.cmd),
                }
            )
            print(
                "WARNING: dcHiC failed for "
                f"{args.chrom} {args.resolution} {cell1}_vs_{cell2}; "
                "skipping this pair and recording it in the failed-pairs CSV."
            )
        finally:
            if not args.keep_work_dir and run_dir.exists():
                shutil.rmtree(run_dir)

    write_csv(
        args.output,
        summary_rows,
        [
            "cell1",
            "cell2",
            "num_bins",
            "mean_abs_compartment_diff",
            "sum_abs_compartment_diff",
            "rms_compartment_diff",
            "compartment_switch_count",
            "mean_aggregate_bin_score",
            "sum_aggregate_bin_score",
            "mean_dchic_sample_maha",
            "max_dchic_sample_maha",
        ],
    )
    write_csv(
        failed_pairs_path(args.output),
        failed_rows,
        ["chrom", "resolution", "cell1", "cell2", "returncode", "failed_command"],
    )


if __name__ == "__main__":
    main()
