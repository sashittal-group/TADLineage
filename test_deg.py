import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Reproducibility
np.random.seed(42)

percentages = ["5%", "10%", "20%"]

# DEG scenario: lower precision than previous TF knockout setting
means = {
    "5%": {"precision": 0.70, "recall": 0.85},
    "10%": {"precision": 0.65, "recall": 0.82},
    "20%": {"precision": 0.60, "recall": 0.88},
}

n_runs = 30

data = []
positions = []
group_centers = []
pos = 1

for pct in percentages:
    precision = np.random.normal(means[pct]["precision"], 0.03, n_runs)
    recall = np.random.normal(means[pct]["recall"], 0.03, n_runs)
    f1 = 2 * (precision * recall) / (precision + recall)

    data.extend([precision, recall, f1])
    positions.extend([pos, pos + 1, pos + 2])
    group_centers.append(pos + 1)
    pos += 4

# Create plot
plt.figure()

box = plt.boxplot(
    data,
    positions=positions,
    widths=0.6,
    patch_artist=True,
    medianprops=dict(color="black")
)

# Blue gradient colors
colors = ["#a6cee3", "#4f81bd", "#08306b"]  # Precision, Recall, F1

# Apply colors
for i, patch in enumerate(box["boxes"]):
    patch.set_facecolor(colors[i % 3])

# X-axis labels
plt.xticks(group_centers, percentages)

plt.ylabel("Score")
plt.title("DEG Analysis Performance Metrics")
plt.ylim(0.4, 1.0)

# Y-axis grid
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Legend
legend_elements = [
    Patch(facecolor=colors[0], label="Precision"),
    Patch(facecolor=colors[1], label="Recall"),
    Patch(facecolor=colors[2], label="F1"),
]
plt.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig("deg_performance_metrics_boxplot.png", dpi=300)
plt.close()