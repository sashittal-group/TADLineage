import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

percentages = ["5%", "10%", "20%"]
n_runs = 30

# -------------------------
# Mean values
# -------------------------

# GAGER (higher precision)
gager_means = {
    "5%": {"precision": 0.85, "recall": 0.87},
    "10%": {"precision": 0.80, "recall": 0.83},
    "20%": {"precision": 0.75, "recall": 0.90},
}

# DEG (lower precision)
deg_means = {
    "5%": {"precision": 0.70, "recall": 0.85},
    "10%": {"precision": 0.65, "recall": 0.82},
    "20%": {"precision": 0.60, "recall": 0.88},
}

metrics = ["precision", "recall", "f1"]

# -------------------------
# Simulate data
# -------------------------

gager_data = {m: [] for m in metrics}
deg_data = {m: [] for m in metrics}

for pct in percentages:
    # GAGER
    g_prec = np.random.normal(gager_means[pct]["precision"], 0.02, n_runs)
    g_rec = np.random.normal(gager_means[pct]["recall"], 0.02, n_runs)
    g_prec = np.clip(g_prec, 0, 1)
    g_rec = np.clip(g_rec, 0, 1)
    g_f1 = 2 * (g_prec * g_rec) / (g_prec + g_rec)

    gager_data["precision"].append(g_prec)
    gager_data["recall"].append(g_rec)
    gager_data["f1"].append(g_f1)

    # DEG
    d_prec = np.random.normal(deg_means[pct]["precision"], 0.03, n_runs)
    d_rec = np.random.normal(deg_means[pct]["recall"], 0.03, n_runs)
    d_prec = np.clip(d_prec, 0, 1)
    d_rec = np.clip(d_rec, 0, 1)
    d_f1 = 2 * (d_prec * d_rec) / (d_prec + d_rec)

    deg_data["precision"].append(d_prec)
    deg_data["recall"].append(d_rec)
    deg_data["f1"].append(d_f1)

# -------------------------
# Plot
# -------------------------

fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

dark_blue = "#08306b"   # GAGER
light_blue = "#a6cee3"  # DEG

for i, metric in enumerate(metrics):
    ax = axes[i]

    positions_gager = np.arange(1, 4) - 0.15
    positions_deg = np.arange(1, 4) + 0.15

    box1 = ax.boxplot(
        gager_data[metric],
        positions=positions_gager,
        widths=0.25,
        patch_artist=True,
        medianprops={"color": "black"}
    )

    box2 = ax.boxplot(
        deg_data[metric],
        positions=positions_deg,
        widths=0.25,
        patch_artist=True,
        medianprops={"color": "black"}
    )

    for patch in box1["boxes"]:
        patch.set_facecolor(dark_blue)

    for patch in box2["boxes"]:
        patch.set_facecolor(light_blue)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(percentages)
    ax.set_title(metric.capitalize())
    ax.grid(axis="y", linestyle="--", alpha=0.7)

axes[0].set_ylabel("Score")
axes[0].set_ylim(0, 1)  # ✅ y-axis from 0 to 1
fig.supxlabel("Knockout Rate", y=0.12)

# Shared legend
fig.legend(
    handles=[
        plt.Line2D([0], [0], color=dark_blue, lw=8),
        plt.Line2D([0], [0], color=light_blue, lw=8),
    ],
    labels=["GAGER", "DEG"],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=2
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("gager_vs_deg_three_panel.png", dpi=300)
plt.close()