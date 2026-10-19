#!/usr/bin/env python3
"""
Run scHiCluster-style clustering on scP/T cell differentiation contacts.

The workflow follows the scHiCluster clustering steps for one chromosome and resolution:
1. convert per-cell raw contact files to sparse scHiCluster matrices,
2. run `hicluster impute-cell` for each cell,
3. run `hicluster embed-concatcell-chr`,
4. run `hicluster embed-mergechr`,
5. cluster/visualize the resulting embedding against the known cell stages.

The cloned scHiCluster repository is used via PYTHONPATH so the local checkout
does not need to be pip-installed with its optional heavy dependencies.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
WORK_DIR = ROOT / "work"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = WORK_DIR / "python_cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_DIR / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))
for _cache_key in ("MPLCONFIGDIR", "NUMBA_CACHE_DIR", "XDG_CACHE_HOME"):
    Path(os.environ[_cache_key]).mkdir(parents=True, exist_ok=True)

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize

try:
    from umap import UMAP

    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


SCHICLUSTER_REPO = ROOT / "scHiCluster"
SOURCE_DIR = PROJECT_ROOT / "scptr" / "cell_differentiation"
DEFAULT_CHROM = "chr1"
DEFAULT_RESOLUTION = 1_000_000
CHROMS = [f"chr{i}" for i in range(1, 20)] + ["chrX"]
MODE = "pad1_std1_rp0.5_sqrtvc"
CONDA_PYTHON = Path("/Users/PSLabMonisha/miniforge3/envs/schiccluster/bin/python")
PLOT_COLORS = np.array(
    [
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#0082c8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#d2f53c",
        "#fabebe",
        "#008080",
        "#e6beff",
        "#aa6e28",
        "#800000",
        "#aaffc3",
        "#808000",
        "#ffd8b1",
        "#000080",
        "#808080",
        "#fffac8",
        "#000000",
    ]
)


def ensure_local_caches() -> None:
    for key in ("MPLCONFIGDIR", "NUMBA_CACHE_DIR", "XDG_CACHE_HOME"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def discover_cells(limit: int | None = None) -> pd.DataFrame:
    rows = []
    for label_dir in sorted(SOURCE_DIR.iterdir()):
        if not label_dir.is_dir():
            continue
        for contact_path in sorted(label_dir.glob("*.txt")):
            cell_name = contact_path.stem
            rows.append(
                {
                    "cell_id": cell_name,
                    "cell_type": label_dir.name,
                    "contact_path": str(contact_path.resolve()),
                }
            )
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise FileNotFoundError(f"No .txt contact files found under {SOURCE_DIR}")
    return pd.DataFrame(rows)


def normalize_chrom(chrom: str) -> str:
    chrom = chrom.strip()
    if not chrom:
        raise ValueError("Chromosome cannot be empty")
    if chrom.lower() in {"all", "genome"}:
        return "all"
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def resolution_label(resolution: int) -> str:
    if resolution % 1_000 == 0:
        return f"{resolution // 1_000}kb"
    return f"{resolution}bp"


def run_tag(chrom: str, resolution: int) -> str:
    return f"{chrom}_{resolution_label(resolution)}"


def run_work_dir(chrom: str, resolution: int) -> Path:
    return WORK_DIR / resolution_label(resolution) / chrom


def run_results_dir(chrom: str, resolution: int) -> Path:
    return RESULTS_DIR / resolution_label(resolution) / chrom


def sorted_chroms(chroms: list[str]) -> list[str]:
    def key(chrom: str) -> tuple[int, str]:
        suffix = chrom.replace("chr", "")
        if suffix == "X":
            return (23, suffix)
        if suffix == "Y":
            return (24, suffix)
        try:
            return (int(suffix), suffix)
        except ValueError:
            return (99, suffix)

    return sorted(chroms, key=key)


def write_chrom_sizes(out_path: Path, chrom: str) -> None:
    chrom_sizes = PROJECT_ROOT / "genome.chrom.sizes"
    size = None
    with chrom_sizes.open() as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if row and row[0] == chrom:
                size = int(row[1])
                break
    if size is None:
        raise ValueError(f"{chrom} not found in {chrom_sizes}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"{chrom}\t{size}\n")


def get_chrom_size(chrom: str) -> int:
    chrom_sizes = PROJECT_ROOT / "genome.chrom.sizes"
    with chrom_sizes.open() as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if row and row[0] == chrom:
                return int(row[1])
    raise ValueError(f"{chrom} not found in {chrom_sizes}")


def convert_contact_file(contact_path: Path, out_path: Path, chrom: str, resolution: int, max_bin: int) -> int:
    counts: Counter[tuple[int, int]] = Counter()
    with contact_path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 6:
                continue
            chrom1, pos1, chrom2, pos2 = fields[1], fields[2], fields[4], fields[5]
            if chrom1 != chrom or chrom2 != chrom:
                continue
            bin1 = int(pos1) // resolution
            bin2 = int(pos2) // resolution
            bin1 = min(bin1, max_bin)
            bin2 = min(bin2, max_bin)
            if bin1 == bin2:
                continue
            if bin2 < bin1:
                bin1, bin2 = bin2, bin1
            counts[(bin1, bin2)] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for (bin1, bin2), count in sorted(counts.items()):
            handle.write(f"{bin1}\t{bin2}\t{count}\n")
    return sum(counts.values())


def prepare_matrices(cells: pd.DataFrame, chrom: str, resolution: int) -> pd.DataFrame:
    rows = []
    work_dir = run_work_dir(chrom, resolution)
    raw_chr_dir = work_dir / "cell_matrix"
    max_bin = get_chrom_size(chrom) // resolution
    for _, row in cells.iterrows():
        out_path = raw_chr_dir / f"{row.cell_id}_{chrom}.txt"
        contacts = convert_contact_file(Path(row.contact_path), out_path, chrom, resolution, max_bin)
        rows.append({**row.to_dict(), "matrix_path": str(out_path), "contacts": contacts})
    prepared = pd.DataFrame(rows)
    prepared.to_csv(work_dir / "cell_metadata.tsv", sep="\t", index=False)
    return prepared


def read_prepared_metadata(chrom: str, resolution: int) -> pd.DataFrame:
    metadata_path = run_work_dir(chrom, resolution) / "cell_metadata.tsv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Prepared metadata not found: {metadata_path}. Run without --analysis-only first."
        )
    return pd.read_csv(metadata_path, sep="\t")


def read_all_chrom_metadata(resolution: int) -> pd.DataFrame:
    combined = None
    missing = []
    for chrom in CHROMS:
        metadata_path = run_work_dir(chrom, resolution) / "cell_metadata.tsv"
        if not metadata_path.exists():
            missing.append(str(metadata_path))
            continue
        metadata = pd.read_csv(metadata_path, sep="\t")
        if combined is None:
            combined = metadata[["cell_id", "cell_type", "contact_path"]].copy()
            combined["contacts"] = 0
        if not combined["cell_id"].equals(metadata["cell_id"]):
            raise ValueError(f"Cell order in {metadata_path} does not match other chromosomes")
        combined["contacts"] += metadata["contacts"].to_numpy()
    if missing:
        raise FileNotFoundError("Prepared metadata missing for all-chromosome analysis:\n" + "\n".join(missing))
    if combined is None:
        raise FileNotFoundError(f"No prepared metadata found under {WORK_DIR / resolution_label(resolution)}")
    return combined


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCHICLUSTER_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_cmd(args: list[str], dry_run: bool = False) -> None:
    print(" ".join(args), flush=True)
    if dry_run:
        return
    subprocess.run(args, cwd=str(ROOT), env=command_env(), check=True)


def impute_cells(cells: pd.DataFrame, chrom_file: Path, python_bin: Path, chrom: str, resolution: int, dry_run: bool = False) -> None:
    work_dir = run_work_dir(chrom, resolution)
    raw_dir = work_dir / "cell_matrix"
    out_dir = work_dir / "imputed_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, row in cells.iterrows():
        expected = out_dir / f"{row.cell_id}_{chrom}_{MODE}.hdf5"
        if expected.exists():
            continue
        print(f"[{index + 1}/{len(cells)}] imputing {row.cell_id}", flush=True)
        run_cmd(
            [
                str(python_bin),
                "-m",
                "schicluster",
                "impute-cell",
                "--indir",
                str(raw_dir.resolve()) + "/",
                "--outdir",
                str(out_dir.resolve()) + "/",
                "--cell",
                row.cell_id,
                "--chrom",
                chrom,
                "--res",
                str(resolution),
                "--pad",
                "1",
                "--std",
                "1",
                "--rp",
                "0.5",
                "--mode",
                MODE,
                "--chrom_file",
                str(chrom_file.resolve()),
            ],
            dry_run=dry_run,
        )


def write_impute_filelist(cells: pd.DataFrame, chrom: str, resolution: int) -> Path:
    work_dir = run_work_dir(chrom, resolution)
    filelist_dir = work_dir / "filelist"
    impute_dir = work_dir / "imputed_matrix"
    filelist_dir.mkdir(parents=True, exist_ok=True)
    path = filelist_dir / f"imputelist_{MODE}.txt"
    with path.open("w") as handle:
        for cell_id in cells["cell_id"]:
            impute_path = impute_dir / f"{cell_id}_{chrom}_{MODE}.hdf5"
            handle.write(str(impute_path.resolve()) + "\n")
    return path


def run_embedding(filelist: Path, python_bin: Path, chrom: str, resolution: int, dry_run: bool = False) -> Path:
    work_dir = run_work_dir(chrom, resolution)
    filelist_dir = work_dir / "filelist"
    embed_dir = work_dir / "merged" / "embed"
    embed_dir.mkdir(parents=True, exist_ok=True)
    outprefix = embed_dir / MODE
    svd_path = Path(f"{outprefix}.svd50.npy")
    if not svd_path.exists():
        run_cmd(
            [
                str(python_bin),
                "-m",
                "schicluster",
                "embed-concatcell-chr",
                "--cell_list",
                str(filelist.resolve()),
                "--outprefix",
                str(outprefix.resolve()),
                "--res",
                str(resolution),
                "--dist",
                "10000000",
                "--dim",
                "50",
            ],
            dry_run=dry_run,
        )

    embed_list = filelist_dir / f"embedlist_{MODE}.txt"
    embed_list.write_text(str(svd_path.resolve()) + "\n")
    merged_prefix = embed_dir / MODE
    merged_path = Path(f"{merged_prefix}.svd20.hdf5")
    if not merged_path.exists():
        run_cmd(
            [
                str(python_bin),
                "-m",
                "schicluster",
                "embed-mergechr",
                "--embed_list",
                str(embed_list.resolve()),
                "--outprefix",
                str(merged_prefix.resolve()),
                "--dim",
                "20",
            ],
            dry_run=dry_run,
        )
    return merged_path


def run_all_chrom_embedding(python_bin: Path, resolution: int, merge_dim: int, dry_run: bool = False) -> Path:
    work_dir = run_work_dir("all", resolution)
    filelist_dir = work_dir / "filelist"
    embed_dir = work_dir / "merged" / "embed"
    filelist_dir.mkdir(parents=True, exist_ok=True)
    embed_dir.mkdir(parents=True, exist_ok=True)

    svd_paths = []
    missing = []
    for chrom in CHROMS:
        svd_path = run_work_dir(chrom, resolution) / "merged" / "embed" / f"{MODE}.svd50.npy"
        if svd_path.exists():
            svd_paths.append(svd_path)
        else:
            missing.append(str(svd_path))
    if missing:
        raise FileNotFoundError(
            "Single-chromosome SVD files missing for all-chromosome merge:\n" + "\n".join(missing)
        )

    embed_list = filelist_dir / f"embedlist_{MODE}.txt"
    embed_list.write_text("\n".join(str(path.resolve()) for path in svd_paths) + "\n")
    outprefix = embed_dir / MODE
    merged_path = Path(f"{outprefix}.svd{merge_dim}.hdf5")
    needs_merge = not merged_path.exists()
    if not needs_merge:
        needs_merge = any(path.stat().st_mtime > merged_path.stat().st_mtime for path in svd_paths)
    if needs_merge:
        run_cmd(
            [
                str(python_bin),
                "-m",
                "schicluster",
                "embed-mergechr",
                "--embed_list",
                str(embed_list.resolve()),
                "--outprefix",
                str(outprefix.resolve()),
                "--dim",
                str(merge_dim),
            ],
            dry_run=dry_run,
        )
    return merged_path


def choose_dimensions(embedding: np.ndarray) -> int:
    if embedding.shape[1] <= 1:
        return embedding.shape[1]
    ndim = 1
    for index in range(embedding.shape[1] - 1):
        pvalue = ks_2samp(embedding[:, index], embedding[:, index + 1]).pvalue
        if pvalue > 0.01:
            break
        ndim = index + 2
    return max(1, min(ndim, embedding.shape[1]))


def make_cluster_features(
    embedding: np.ndarray,
    cells: pd.DataFrame,
    cluster_dims: int | None,
    feature_mode: str,
) -> tuple[np.ndarray, int]:
    if cluster_dims is None:
        ndim = choose_dimensions(embedding)
    else:
        ndim = max(1, min(cluster_dims, embedding.shape[1]))

    used = embedding[:, :ndim]
    if feature_mode == "embedding":
        return used, ndim
    if feature_mode == "embedding-l2":
        return normalize(used), ndim

    depth = np.log1p(cells["contacts"].to_numpy(dtype=float))[:, None]
    if feature_mode == "depth":
        return StandardScaler().fit_transform(depth), ndim
    if feature_mode == "embedding-depth":
        return StandardScaler().fit_transform(np.c_[used, depth]), ndim
    raise ValueError(f"Unknown feature mode: {feature_mode}")


def analyze_embedding(
    embedding_path: Path,
    cells: pd.DataFrame,
    chrom: str,
    resolution: int,
    cluster_dims: int | None,
    feature_mode: str,
    kmeans_n_init: int,
    cluster_method: str,
) -> None:
    with h5py.File(embedding_path, "r") as handle:
        embedding = handle["data"][()]
    labels = cells["cell_type"].to_numpy()
    label_ids = LabelEncoder().fit_transform(labels)
    n_clusters = len(np.unique(label_ids))
    used, ndim = make_cluster_features(embedding, cells, cluster_dims, feature_mode)

    if cluster_method == "kmeans":
        predicted = KMeans(n_clusters=n_clusters, random_state=0, n_init=kmeans_n_init).fit_predict(used)
    elif cluster_method == "ward":
        predicted = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(used)
    else:
        raise ValueError(f"Unknown cluster method: {cluster_method}")
    ari = adjusted_rand_score(label_ids, predicted)
    nmi = normalized_mutual_info_score(label_ids, predicted)
    sil = silhouette_score(used, predicted) if len(np.unique(predicted)) > 1 else float("nan")
    depth_only = StandardScaler().fit_transform(np.log1p(cells["contacts"].to_numpy(dtype=float))[:, None])
    depth_predicted = KMeans(n_clusters=n_clusters, random_state=0, n_init=kmeans_n_init).fit_predict(depth_only)
    depth_ari = adjusted_rand_score(label_ids, depth_predicted)
    depth_nmi = normalized_mutual_info_score(label_ids, depth_predicted)

    results_dir = run_results_dir(chrom, resolution)
    results_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = results_dir / "schiccluster_clusters.tsv"
    cluster_table = cells[["cell_id", "cell_type", "contacts"]].copy()
    cluster_table["predicted_cluster"] = predicted
    cluster_table.to_csv(assignment_path, sep="\t", index=False)

    result_path = results_dir / "schiccluster_results.txt"
    with result_path.open("w") as handle:
        handle.write(f"scHiCluster {chrom} {resolution_label(resolution)} clustering results\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"embedding\t{embedding_path}\n")
        handle.write(f"feature_mode\t{feature_mode}\n")
        handle.write(f"selected_dimensions\t{ndim}\n")
        handle.write(f"cluster_method\t{cluster_method}\n")
        handle.write(f"kmeans_n_init\t{kmeans_n_init}\n")
        handle.write(f"ARI\t{ari:.6f}\n")
        handle.write(f"NMI\t{nmi:.6f}\n")
        handle.write(f"Silhouette\t{sil:.6f}\n")
        handle.write(f"depth_only_ARI\t{depth_ari:.6f}\n")
        handle.write(f"depth_only_NMI\t{depth_nmi:.6f}\n")
        handle.write(f"cluster_table\t{assignment_path}\n")
        handle.write("\ncluster_id\tsize\tmajor_cell_type\n")
        for cluster_id in sorted(np.unique(predicted)):
            mask = predicted == cluster_id
            major = pd.Series(labels[mask]).value_counts().idxmax()
            handle.write(f"{cluster_id}\t{int(mask.sum())}\t{major}\n")
        handle.write("\ncell_id\tpredicted_cluster\tcell_type\tcontacts\n")
        for cell_id, pred, label, contacts in zip(cells["cell_id"], predicted, labels, cells["contacts"]):
            handle.write(f"{cell_id}\t{pred}\t{label}\t{contacts}\n")
    print(f"Wrote {result_path}")
    print(f"Wrote {assignment_path}")

    if HAS_UMAP:
        coords = UMAP(n_neighbors=min(25, len(cells) - 1), random_state=0).fit_transform(used)
    else:
        coords = used[:, :2] if used.shape[1] >= 2 else np.c_[used[:, 0], np.zeros(len(used))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    for axis in axes:
        axis.set_xlabel("UMAP-1" if HAS_UMAP else "Dim 1", fontsize=12)
        axis.set_ylabel("UMAP-2" if HAS_UMAP else "Dim 2", fontsize=12)
        axis.spines["right"].set_visible(False)
        axis.spines["top"].set_visible(False)
        axis.tick_params(axis="both", which="both", length=0)
        axis.set_xticklabels([])
        axis.set_yticklabels([])

    unique_labels = pd.Series(labels).value_counts().index.tolist()
    for index, label in enumerate(unique_labels):
        mask = labels == label
        axes[0].scatter(coords[mask, 0], coords[mask, 1], c=PLOT_COLORS[index % len(PLOT_COLORS)], s=12, edgecolors="none", alpha=0.8, label=label, rasterized=True)
    axes[0].set_title("Cell Type", fontsize=14)
    axes[0].legend(
        markerscale=3,
        prop={"size": 8},
        bbox_to_anchor=(1, 1),
        loc="upper left",
        labelspacing=1.3,
        handletextpad=1.0,
        borderpad=0.7,
    )

    unique_clusters = sorted(np.unique(predicted))
    for index, cluster_id in enumerate(unique_clusters):
        mask = predicted == cluster_id
        axes[1].scatter(coords[mask, 0], coords[mask, 1], c=PLOT_COLORS[index % len(PLOT_COLORS)], s=12, edgecolors="none", alpha=0.8, label=f"Cluster {cluster_id}", rasterized=True)
    axes[1].set_title("scHiCluster clustering", fontsize=14)
    plt.tight_layout()
    fig_path = results_dir / "schiccluster_figure.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {fig_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=CONDA_PYTHON, help="Python executable from the schiccluster conda env")
    parser.add_argument("--chrom", default=DEFAULT_CHROM, help="Chromosome to process, e.g. chr1, 1, or all")
    parser.add_argument("--res", "--resolution", dest="resolution", type=int, default=DEFAULT_RESOLUTION, help="Bin size in bp, e.g. 1000000 for 1000kb")
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N cell limit for a smoke test")
    parser.add_argument("--prepare-only", action="store_true", help="Only convert contacts and write command filelists")
    parser.add_argument("--analysis-only", action="store_true", help="Reuse cached metadata and embedding; only rerun clustering and plots")
    parser.add_argument("--dry-run", action="store_true", help="Print scHiCluster commands without running them")
    parser.add_argument(
        "--cluster-dims",
        type=int,
        default=20,
        help="Number of SVD dimensions to cluster. Use 0 to restore the old KS-selected dimension rule.",
    )
    parser.add_argument(
        "--cluster-feature-mode",
        choices=("embedding", "embedding-l2", "embedding-depth", "depth"),
        default="embedding-depth",
        help="Features used for KMeans clustering after scHiCluster embedding.",
    )
    parser.add_argument(
        "--cluster-method",
        choices=("kmeans", "ward"),
        default="kmeans",
        help="Clustering method for the final embedding.",
    )
    parser.add_argument("--kmeans-n-init", type=int, default=100, help="KMeans random restarts for stable clustering")
    parser.add_argument("--all-merge-dim", type=int, default=50, help="Final SVD dimensions for --chrom all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chrom = normalize_chrom(args.chrom)
    resolution = args.resolution
    if resolution <= 0:
        raise ValueError("--res/--resolution must be a positive integer number of bp")
    if args.kmeans_n_init <= 0:
        raise ValueError("--kmeans-n-init must be positive")
    cluster_dims = None if args.cluster_dims == 0 else args.cluster_dims
    if cluster_dims is not None and cluster_dims <= 0:
        raise ValueError("--cluster-dims must be positive, or 0 for automatic KS selection")
    if args.all_merge_dim <= 0:
        raise ValueError("--all-merge-dim must be positive")
    ensure_local_caches()
    work_dir = run_work_dir(chrom, resolution)
    work_dir.mkdir(parents=True, exist_ok=True)
    chrom_file = work_dir / f"{chrom}.chrom.sizes"
    if chrom != "all":
        write_chrom_sizes(chrom_file, chrom)

    if chrom == "all":
        if args.limit is not None:
            raise ValueError("--limit cannot be combined with --chrom all")
        cells = read_all_chrom_metadata(resolution)
        cells.to_csv(work_dir / "cell_metadata.tsv", sep="\t", index=False)
        filelist = work_dir / "filelist" / f"embedlist_{MODE}.txt"
    elif args.analysis_only:
        if args.limit is not None:
            raise ValueError("--limit cannot be combined with --analysis-only")
        cells = read_prepared_metadata(chrom, resolution)
        filelist = run_work_dir(chrom, resolution) / "filelist" / f"imputelist_{MODE}.txt"
    else:
        cells = discover_cells(args.limit)
        cells = prepare_matrices(cells, chrom, resolution)
        filelist = write_impute_filelist(cells, chrom, resolution)
    print(f"Prepared {len(cells)} cells for {chrom} at {resolution} bp")
    print(f"Work directory: {work_dir}")
    print(f"Results directory: {run_results_dir(chrom, resolution)}")
    print(f"Metadata: {work_dir / 'cell_metadata.tsv'}")
    print(f"Impute file list: {filelist}")

    if args.prepare_only:
        return
    if not args.python.exists() and not args.dry_run:
        raise FileNotFoundError(f"Python executable not found: {args.python}")

    if chrom == "all":
        embedding_path = run_all_chrom_embedding(args.python, resolution, args.all_merge_dim, args.dry_run)
    elif args.analysis_only:
        embedding_path = run_work_dir(chrom, resolution) / "merged" / "embed" / f"{MODE}.svd20.hdf5"
        if not embedding_path.exists():
            raise FileNotFoundError(
                f"Cached embedding not found: {embedding_path}. Run without --analysis-only first."
            )
    else:
        impute_cells(cells, chrom_file, args.python, chrom, resolution, args.dry_run)
        embedding_path = run_embedding(filelist, args.python, chrom, resolution, args.dry_run)
    if not args.dry_run:
        analyze_embedding(
            embedding_path,
            cells,
            chrom,
            resolution,
            cluster_dims,
            args.cluster_feature_mode,
            args.kmeans_n_init,
            args.cluster_method,
        )


if __name__ == "__main__":
    main()
