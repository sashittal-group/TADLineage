#!/usr/bin/env python3
"""
Cluster cells using the Higashi tutorial scA/B compartment workflow.

This mirrors the A/B compartment section of the Higashi tutorials:
1. call single-cell A/B compartment scores with the same normalization/PCA
   logic as Higashi's scCompartment.py,
2. use the quantile-normalized genome-wide scA/B matrix as cell features,
3. project those features with PCA for embedding visualization,
4. visualize the PCA features by available label metadata.
"""
import json
import os
import pickle
import warnings

import h5py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import LabelEncoder, quantile_transform

from run_paths import parse_run_args, run_paths, write_run_config

try:
    from scipy.stats import zscore

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
TEMP_DIR = os.path.join(ROOT, "temp")
RAW_DIR = os.path.join(TEMP_DIR, "raw")
CACHE_DIR = os.path.join(TEMP_DIR, "python_cache")
RESULTS_DIR = ROOT
HAS_VIS = False
VIS_IMPORT_ERROR = None
CELL_TYPE_ORDER = ["1-cell", "2-cell", "4-cell", "8-cell", "64-cell"]


def configure_run_dirs(paths):
    global DATA_DIR, TEMP_DIR, RAW_DIR, CACHE_DIR, RESULTS_DIR

    DATA_DIR = paths["data_dir"]
    TEMP_DIR = paths["temp_dir"]
    RAW_DIR = os.path.join(TEMP_DIR, "raw")
    CACHE_DIR = os.path.join(TEMP_DIR, "python_cache")
    RESULTS_DIR = paths["results_dir"]

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(CACHE_DIR, "matplotlib"))
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(CACHE_DIR, "numba"))
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(CACHE_DIR, "xdg"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def configure_visualization():
    global HAS_VIS, VIS_IMPORT_ERROR, plt, UMAP

    try:
        import matplotlib.pyplot as plt
        from umap import UMAP

        HAS_VIS = True
    except Exception as exc:
        VIS_IMPORT_ERROR = exc
        HAS_VIS = False


def load_config(config_path):
    with open(config_path, "r") as handle:
        return json.load(handle)


def load_label_info():
    labels_path = os.path.join(DATA_DIR, "label_info.pickle")
    with open(labels_path, "rb") as handle:
        labels = pickle.load(handle)
    if "cell type" not in labels:
        raise KeyError('Expected "cell type" in data/label_info.pickle')
    return labels


def sqrt_norm(matrix):
    coverage = np.sqrt(np.sum(matrix, axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = matrix / coverage.reshape((-1, 1))
        matrix = matrix / coverage.reshape((1, -1))
    matrix[np.isnan(matrix)] = 0.0
    matrix[np.isinf(matrix)] = 0.0
    return matrix


def kth_diag_indices(matrix, k):
    rows, cols = np.diag_indices_from(matrix)
    if k < 0:
        return rows[-k:], cols[:k]
    if k > 0:
        return rows[:-k], cols[k:]
    return rows, cols


def oe(matrix, expected=None):
    new_matrix = np.zeros_like(matrix)
    for k in range(len(matrix)):
        rows, cols = kth_diag_indices(matrix, k)
        diag = np.diag(matrix, k)
        if expected is not None:
            expect = expected[k]
        else:
            expect = np.sum(diag) / (np.sum(diag != 0.0) + 1e-15)
        if expect == 0:
            new_matrix[rows, cols] = 0.0
        else:
            new_matrix[rows, cols] = diag / expect
    return new_matrix + new_matrix.T


def pearson(matrix):
    return np.corrcoef(matrix)


def compartment(matrix, model=None, expected=None, return_model=False):
    contact = sqrt_norm(matrix)
    contact = oe(contact, expected)
    np.fill_diagonal(contact, 1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        contact = pearson(contact)
    np.fill_diagonal(contact, 1)
    contact[np.isnan(contact)] = 0.0
    if model is not None:
        return model.transform(contact).reshape(-1)

    pca = PCA(n_components=1)
    values = pca.fit_transform(contact).reshape(-1)
    if return_model:
        return values, pca
    return values


def find_impute_file(chrom, embedding_name, neighbor_num, use_neighbor=True):
    if use_neighbor and neighbor_num > 0:
        candidates = [
            os.path.join(TEMP_DIR, f"{chrom}_{embedding_name}_nbr_{neighbor_num}_impute.hdf5"),
            os.path.join(TEMP_DIR, f"{chrom}_{embedding_name}_nbr_0_impute.hdf5"),
        ]
    else:
        candidates = [
            os.path.join(TEMP_DIR, f"{chrom}_{embedding_name}_nbr_0_impute.hdf5"),
            os.path.join(TEMP_DIR, f"{chrom}_{embedding_name}_nbr_{neighbor_num}_impute.hdf5"),
        ]

    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No imputed contact map found for {chrom}: {candidates}")


def load_raw_bulk_matrix(chrom):
    raw_path = os.path.join(RAW_DIR, f"{chrom}_sparse_adj.npy")
    if not os.path.exists(raw_path):
        return None

    origin_sparse = np.load(raw_path, allow_pickle=True)
    bulk = np.sum(origin_sparse, axis=0)
    if hasattr(bulk, "todense"):
        bulk = bulk.todense()
    return np.asarray(bulk, dtype=float)


def parse_calibration_file(path, config):
    resolution = int(config["resolution"])
    chrom_sizes = np.load(os.path.join(TEMP_DIR, "chrom_start_end.npy"))
    chrom_list = config.get("impute_list", config.get("chrom_list", []))
    rows = np.genfromtxt(path, dtype=None, encoding=None)
    if rows.ndim == 0:
        rows = np.asarray([rows])

    result = {}
    for chrom_index, chrom in enumerate(chrom_list):
        size = int(chrom_sizes[chrom_index, 1] - chrom_sizes[chrom_index, 0])
        values = np.zeros(size)
        for row in rows:
            row_chrom, bin_start, value = str(row[0]), float(row[1]), float(row[2])
            if row_chrom != chrom:
                continue
            bin_index = int(bin_start / resolution)
            if 0 <= bin_index < size:
                values[bin_index] = np.nan if value == -1 else value
        result[chrom] = values
    return result


def load_calibration(config):
    for key in ("calib_file", "cpg_file"):
        path = config.get(key)
        if path and os.path.exists(path):
            return parse_calibration_file(path, config)

    calib_path = os.path.join(TEMP_DIR, "calib.npy")
    if os.path.exists(calib_path):
        return np.load(calib_path, allow_pickle=True).item()

    return None


def maybe_reverse_by_calibration(compartment_values, calib_values):
    if calib_values is None:
        return compartment_values, False

    high = np.nanmean(calib_values[compartment_values > np.quantile(compartment_values, 0.9)])
    low = np.nanmean(calib_values[compartment_values < np.quantile(compartment_values, 0.1)])
    if high < low:
        return -1 * compartment_values, True
    return compartment_values, False


def mask_gaps(matrix):
    covered = np.sum(matrix, axis=-1) > 0
    mask = np.ones_like(matrix)
    mask[~covered, :] = 0
    mask[:, ~covered] = 0
    return mask


def reconstruct_matrix(size, coordinates, values):
    matrix = np.zeros((size, size), dtype=float)
    xs = coordinates[:, 0].astype(int)
    ys = coordinates[:, 1].astype(int)
    matrix[xs, ys] = values
    matrix = matrix + matrix.T
    return matrix


def load_chromosome_compartment_features(chrom, config, calibration=None):
    embedding_name = config.get("embedding_name", "embed")
    neighbor_num = int(config.get("neighbor_num", 0))
    use_neighbor = bool(config.get("impute_with_nbr", True))
    impute_file = find_impute_file(chrom, embedding_name, neighbor_num, use_neighbor)

    with h5py.File(impute_file, "r") as handle:
        coordinates = handle["coordinates"][:]
        size = int(np.max(coordinates)) + 1
        cell_ids = sorted(
            [key for key in handle.keys() if key.startswith("cell_")],
            key=lambda key: int(key.split("_")[1]),
        )

        imputed_bulk = np.zeros((size, size), dtype=float)
        cell_matrices = []
        for cell_id in cell_ids:
            matrix = reconstruct_matrix(size, coordinates, handle[cell_id][:])
            imputed_bulk += matrix
            cell_matrices.append(matrix)

    bulk_matrix = load_raw_bulk_matrix(chrom)
    if bulk_matrix is None:
        bulk_matrix = imputed_bulk
    if bulk_matrix.shape != imputed_bulk.shape:
        raise ValueError(f"Raw and imputed matrix sizes differ for {chrom}: {bulk_matrix.shape} vs {imputed_bulk.shape}")

    mask = mask_gaps(bulk_matrix)
    bulk_matrix = bulk_matrix * mask
    cell_matrices = [matrix * mask for matrix in cell_matrices]

    use_rows = np.where(np.sum(bulk_matrix > 0, axis=-1) > 0.01 * len(bulk_matrix))[0]
    if len(use_rows) <= 1:
        raise ValueError(f"No reliable compartment bins found for {chrom}")

    bulk_slice = bulk_matrix[np.ix_(use_rows, use_rows)]
    bulk_compartment, bulk_model = compartment(bulk_slice, return_model=True)

    calib_values = None
    if calibration is not None and chrom in calibration:
        calib_values = np.asarray(calibration[chrom]).reshape(-1)[use_rows]
    bulk_compartment, reverse_flag = maybe_reverse_by_calibration(bulk_compartment, calib_values)

    raw_features = []
    for matrix in cell_matrices:
        cell_slice = matrix[np.ix_(use_rows, use_rows)]
        cell_compartment = compartment(cell_slice, model=bulk_model)
        if reverse_flag:
            cell_compartment = -1 * cell_compartment
        raw_features.append(cell_compartment)
    raw_features = np.asarray(raw_features)

    quantile_features = quantile_transform(
        raw_features,
        output_distribution="uniform",
        n_quantiles=int(raw_features.shape[-1]),
        axis=1,
    )
    zscore_features = zscore(raw_features, axis=1) if HAS_SCIPY else None

    return {
        "chrom": chrom,
        "impute_file": impute_file,
        "features": quantile_features,
        "raw_features": raw_features,
        "zscore_features": zscore_features,
        "bulk_compartment": bulk_compartment,
        "bulk_model": bulk_model,
        "reverse_flag": reverse_flag,
        "use_rows": use_rows,
    }


def build_feature_matrix(chrom_list, config, calibration):
    blocks = []
    metadata = []
    for chrom in chrom_list:
        print(f"Calling Higashi scA/B features for {chrom}...")
        result = load_chromosome_compartment_features(chrom, config, calibration)
        print(f"  Imputed source: {result['impute_file']}")
        blocks.append(result["features"])
        metadata.append(result)

    sample_count = {block.shape[0] for block in blocks}
    if len(sample_count) != 1:
        raise ValueError(f"Chromosomes have different cell counts: {sorted(sample_count)}")

    return np.concatenate(blocks, axis=1), metadata


def reduce_scab_with_pca(genome_scab):
    n_components = min(512, genome_scab.shape[0], genome_scab.shape[1])
    if n_components < 1:
        raise ValueError(f"Cannot run PCA on empty scA/B matrix: {genome_scab.shape}")
    return PCA(n_components=n_components, random_state=0).fit_transform(genome_scab)


def save_sccompartment_hdf5(out_path, metadata, resolution):
    with h5py.File(out_path, "w") as handle:
        group = handle.create_group("compartment")
        bin_group = group.create_group("bin")
        chrom_values = []
        starts = []
        ends = []
        bulk_values = []
        cell_blocks = []

        for result in metadata:
            use_rows = result["use_rows"]
            chrom_values.extend([result["chrom"]] * len(use_rows))
            starts.append((use_rows * resolution).astype(int))
            ends.append(((use_rows + 1) * resolution).astype(int))
            bulk_values.append(result["bulk_compartment"])
            cell_blocks.append(result["features"])

        bin_group.create_dataset("chrom", data=[chrom.encode("utf8") for chrom in chrom_values])
        bin_group.create_dataset("start", data=np.concatenate(starts))
        bin_group.create_dataset("end", data=np.concatenate(ends))
        group.create_dataset("bulk", data=np.concatenate(bulk_values))

        scab = np.concatenate(cell_blocks, axis=1)
        for cell_index in range(scab.shape[0]):
            group.create_dataset(f"cell_{cell_index}", data=scab[cell_index])


def display_resolution(resolution):
    if resolution % 1_000_000 == 0:
        return f"{resolution // 1_000_000}Mb resolution"
    if resolution % 1_000 == 0:
        return f"{resolution // 1_000}kb resolution"
    return f"{resolution}bp resolution"


def cluster_compartment_features(pca_feats, cell_type):
    label_ids = LabelEncoder().fit_transform(cell_type)
    n_clusters = len(np.unique(label_ids))
    predicted = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(pca_feats)
    scores = {
        "ARI": adjusted_rand_score(label_ids, predicted),
        "NMI": normalized_mutual_info_score(label_ids, predicted),
        "Silhouette": silhouette_score(pca_feats, predicted)
        if len(np.unique(predicted)) > 1
        else float("nan"),
    }
    return predicted, scores


def save_results(out_path, genome_scab, pca_feats, label_info, predicted, scores):
    with open(out_path, "w") as handle:
        handle.write("Higashi tutorial scA/B compartment clustering results\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"scAB_features\tquantile-normalized genome-wide scA/B\t{genome_scab.shape}\n")
        handle.write(f"PCA_features\t{pca_feats.shape}\n")
        handle.write("Clustering\tKMeans on PCA(scA/B), plus Lee et al. hierarchical scA/B clustermap\n")
        handle.write(f"ARI\t{scores['ARI']:.6f}\n")
        handle.write(f"NMI\t{scores['NMI']:.6f}\n")
        handle.write(f"Silhouette\t{scores['Silhouette']:.6f}\n")
        handle.write("\ncell_index\tpredicted_cluster\tcell_type\tcoverage\n")
        cell_type = np.asarray(label_info.get("cell type", ["NA"] * len(pca_feats)))
        coverage = np.asarray(label_info.get("coverage", ["NA"] * len(pca_feats)))
        for cell_index, (pred, truth, cov) in enumerate(zip(predicted, cell_type, coverage)):
            handle.write(f"{cell_index}\t{pred}\t{truth}\t{cov}\n")


def encode_cell_types(cell_type):
    cell_type = np.asarray(cell_type)
    known = [name for name in CELL_TYPE_ORDER if name in set(cell_type)]
    extra = sorted(set(cell_type) - set(known))
    classes = known + extra
    class_to_idx = {name: index for index, name in enumerate(classes)}
    return np.asarray([class_to_idx[name] for name in cell_type]), classes


def visualize_pca_features(feats, label_info, predicted, output_path, title):
    if not HAS_VIS:
        return

    n_neighbors = min(15, max(2, feats.shape[0] - 1))
    x_vis = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.3, random_state=0).fit_transform(feats)

    cell_type = np.asarray(label_info.get("cell type", ["unknown"] * len(feats)))
    cell_type_int, cell_type_names = encode_cell_types(cell_type)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=150)
    cluster_cmap = plt.get_cmap("tab10", len(np.unique(predicted)))
    true_cmap = plt.get_cmap("tab10", len(cell_type_names))

    cluster_scatter = axes[0].scatter(
        x_vis[:, 0],
        x_vis[:, 1],
        c=predicted,
        cmap=cluster_cmap,
        s=45,
        alpha=0.8,
        linewidths=0,
    )
    axes[0].set_title("Predicted Clusters", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    plt.colorbar(cluster_scatter, ax=axes[0], label="Cluster")

    axes[1].scatter(
        x_vis[:, 0],
        x_vis[:, 1],
        c=cell_type_int,
        cmap=true_cmap,
        s=45,
        alpha=0.8,
        linewidths=0,
    )
    axes[1].set_title("True Cell Types", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=true_cmap(index),
            label=label,
            markersize=8,
        )
        for index, label in enumerate(cell_type_names)
    ]
    axes[1].legend(handles=handles, loc="best", fontsize=9)

    fig.suptitle(title, y=1.04, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_scab_clustermap(genome_scab, label_info, output_path):
    if not HAS_VIS:
        return

    import seaborn as sns

    cell_type = np.asarray(label_info.get("cell type", ["unknown"] * len(genome_scab)))
    classes = sorted(np.unique(cell_type))
    palette = dict(zip(classes, sns.color_palette("tab10", n_colors=len(classes))))
    row_colors = [palette[label] for label in cell_type]

    cluster_grid = sns.clustermap(
        genome_scab,
        row_cluster=True,
        col_cluster=True,
        row_colors=row_colors,
        cmap="viridis",
        figsize=(14, 8),
        xticklabels=False,
        yticklabels=False,
    )
    cluster_grid.fig.suptitle("Hierarchical clustering on genome-wide scA/B", y=1.02)
    cluster_grid.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(cluster_grid.fig)


def main():
    args = parse_run_args("Call and visualize Higashi scA/B compartment features.")
    config_path, paths = write_run_config(args.chromosome, args.resolution)
    configure_run_dirs(paths)
    configure_visualization()

    config = load_config(config_path)
    label_info = load_label_info()

    chrom_list = config.get("impute_list", ["chr1"])
    calibration = load_calibration(config)

    print("=" * 70)
    print("HIGASHI TUTORIAL scA/B PCA VISUALIZATION")
    print(f"Chromosomes: {', '.join(chrom_list)}")
    print(f"Cells: {len(label_info['cell type'])}")
    print(f"Calibration: {'enabled' if calibration is not None else 'not found'}")
    print("=" * 70)

    genome_scab, metadata = build_feature_matrix(chrom_list, config, calibration)
    print(f"Genome scA/B matrix shape: {genome_scab.shape}")
    pca_feats = reduce_scab_with_pca(genome_scab)
    print(f"PCA feature shape for visualization: {pca_feats.shape}")
    cell_type = np.asarray(label_info.get("cell type", ["unknown"] * len(pca_feats)))
    predicted, scores = cluster_compartment_features(pca_feats, cell_type)
    print("Following Lee et al. notebook: PCA(scA/B) visualization plus hierarchical scA/B clustermap.")
    print(f"Compartment KMeans: ARI={scores['ARI']:.4f} NMI={scores['NMI']:.4f}")

    out_path = os.path.join(RESULTS_DIR, "compartment_clustering_results.txt")
    save_results(out_path, genome_scab, pca_feats, label_info, predicted, scores)
    print(f"Results saved to: {out_path}")

    hdf5_path = os.path.join(TEMP_DIR, "scCompartment.hdf5")
    save_sccompartment_hdf5(hdf5_path, metadata, int(config["resolution"]))
    print(f"scA/B features saved to: {hdf5_path}")

    if HAS_VIS:
        fig_path = os.path.join(RESULTS_DIR, "compartment_clustering_figure.png")
        print("Generating PCA(scA/B)+UMAP visualization...")
        title = f"{paths['chromosome']} Compartment-Based Clustering ({display_resolution(paths['resolution'])})"
        visualize_pca_features(pca_feats, label_info, predicted, fig_path, title)
        print(f"Figure saved to: {fig_path}")

        clustermap_path = os.path.join(RESULTS_DIR, "compartment_clustermap.png")
        print("Generating Lee-style hierarchical scA/B clustermap...")
        visualize_scab_clustermap(genome_scab, label_info, clustermap_path)
        print(f"Clustermap saved to: {clustermap_path}")
    else:
        print(f"Skipping visualization because matplotlib or umap is unavailable: {VIS_IMPORT_ERROR}")


if __name__ == "__main__":
    main()
