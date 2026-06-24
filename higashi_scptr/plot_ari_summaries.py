from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESOLUTIONS = ["500000", "1000000"]
RESULT_SPECS = {
    "cluster": {
        "filename": "cluster_results.txt",
        "title": "Cluster ARI",
        "color": "#2F6F9F",
    },
    "compartment_cluster": {
        "filename": "compartment_clustering_results.txt",
        "title": "Compartment Cluster ARI",
        "color": "#B45F06",
    },
}


def chrom_key(chromosome):
    label = chromosome.replace("chr", "")
    if label == "X":
        return 23
    if label == "Y":
        return 24
    try:
        return int(label)
    except ValueError:
        return 99


def read_ari(path):
    with path.open() as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "ARI":
                return float(parts[1])
    raise ValueError(f"ARI not found in {path}")


def average(values):
    return sum(values) / len(values)


def collect_rows():
    rows = []
    for resolution in RESOLUTIONS:
        resolution_dir = RESULTS_DIR / resolution
        for chrom_dir in sorted(resolution_dir.glob("chr*"), key=lambda p: chrom_key(p.name)):
            if not chrom_dir.is_dir():
                continue
            for kind, spec in RESULT_SPECS.items():
                result_path = chrom_dir / spec["filename"]
                if result_path.exists():
                    rows.append(
                        {
                            "resolution": resolution,
                            "resolution_bp": int(resolution),
                            "chromosome": chrom_dir.name,
                            "kind": kind,
                            "ari": read_ari(result_path),
                        }
                    )
    return rows


def write_summary(rows, fig_dir):
    summary_path = fig_dir / "ari_summary.tsv"
    with summary_path.open("w") as out:
        out.write("kind\tresolution\tchromosome\tari\n")
        for row in sorted(
            rows,
            key=lambda r: (r["kind"], r["resolution_bp"], chrom_key(r["chromosome"])),
        ):
            out.write(
                f"{row['kind']}\t{row['resolution']}\t{row['chromosome']}\t{row['ari']:.6f}\n"
            )
    return summary_path


def plot_by_resolution(rows, kind, spec, fig_dir):
    by_resolution = []
    for resolution in RESOLUTIONS:
        values = [
            row["ari"]
            for row in rows
            if row["kind"] == kind and row["resolution"] == resolution
        ]
        if values:
            by_resolution.append((resolution, average(values)))

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    x_values = [item[0] for item in by_resolution]
    y_values = [item[1] for item in by_resolution]
    ax.scatter(x_values, y_values, s=90, color=spec["color"], zorder=3)
    ax.set_title(f"{spec['title']} by Resolution")
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Average ARI")
    ax.set_ylim(0, max(0.01, max(y_values) * 1.25))
    for x_value, y_value in by_resolution:
        ax.annotate(
            f"{y_value:.3f}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )
    fig.savefig(fig_dir / f"{kind}_ari_by_resolution.png", dpi=300)
    plt.close(fig)


def plot_by_chromosome(rows, kind, spec, fig_dir):
    chromosomes = sorted(
        {row["chromosome"] for row in rows if row["kind"] == kind}, key=chrom_key
    )
    by_chromosome = []
    for chromosome in chromosomes:
        values = [
            row["ari"]
            for row in rows
            if row["kind"] == kind and row["chromosome"] == chromosome
        ]
        if values:
            by_chromosome.append((chromosome, average(values)))

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    x_values = [item[0] for item in by_chromosome]
    y_values = [item[1] for item in by_chromosome]
    ax.scatter(x_values, y_values, s=55, color=spec["color"], zorder=3)
    ax.set_title(f"{spec['title']} by Chromosome")
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Average ARI")
    ax.set_ylim(0, max(0.01, max(y_values) * 1.20))
    ax.tick_params(axis="x", rotation=45)
    fig.savefig(fig_dir / f"{kind}_ari_by_chromosome.png", dpi=300)
    plt.close(fig)


def main():
    rows = collect_rows()
    if not rows:
        raise SystemExit("No ARI rows found")

    fig_dir = RESULTS_DIR / "figure"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    summary_path = write_summary(rows, fig_dir)
    for kind, spec in RESULT_SPECS.items():
        plot_by_resolution(rows, kind, spec, fig_dir)
        plot_by_chromosome(rows, kind, spec, fig_dir)

    print(f"Wrote {summary_path}")
    for path in sorted(fig_dir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
