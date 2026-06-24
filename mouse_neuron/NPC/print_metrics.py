import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# read csv
df = pd.read_csv("median_metrics.csv")

# find top 5 chromosomes using penalty = 0.1
top_chr = (
    df[df["penalty"] == 0.1]
    .sort_values("f1", ascending=False)
    .head(5)["chr"]
    .tolist()
)

# keep all penalties for those chromosomes
df_top = df[df["chr"].isin(top_chr)]

# convert to long format
df_long = df_top.melt(
    id_vars=["chr", "penalty"],
    value_vars=["precision", "f1"],
    var_name="metric",
    value_name="score"
)

# plot
plt.figure(figsize=(9,5))

sns.boxplot(
    data=df_long,
    x="chr",
    y="score",
    hue="metric"
)

plt.xlabel("Chromosome")
plt.ylabel("Score")
plt.title("Precision and F1 (ESC-NPC-N)")
plt.ylim(0,1)

plt.tight_layout()
plt.savefig("precision_f1_boxplot.png", dpi=300)
plt.close()