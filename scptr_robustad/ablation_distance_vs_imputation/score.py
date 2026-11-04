#!/usr/bin/env python3
"""Post-hoc ARI/NMI of every run; the only script in this folder that reads stage labels.

For each (bin size, scope, hierarchy, method) three start-selection rules, each picking the
start with the lowest objective of that method (never by ARI):

  tad_only        k-means++ starts 1..20 only: contacts enter only through the hierarchies
  paper_starts    spectral + k-means++ 1..4, the paper's start set
  spectral_start  the spectral-started run alone

The spectral start itself (contacts only, no TADs) is scored as hierarchy "none".

Writes summary.tsv (one row per selection) and per_start.tsv (every start).

    python score.py
"""

import csv
import json

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from common import HERE, PAPER_RANDOM_STARTS, REPO_ROOT, UM

RULES = {"tad_only": lambda names: [i for i, n in enumerate(names) if n != "spectral"],
         "paper_starts": lambda names: list(range(PAPER_RANDOM_STARTS + 1)),
         "spectral_start": lambda names: [0]}


def truth():
    ids = json.loads((UM / "data/cells.json").read_text())
    with open(REPO_ROOT / "scptr/GSE129029_metadata.csv") as handle:
        stage = {row["GSM"]: row["CELL_STAGE"] for row in csv.DictReader(handle)}
    return np.array([stage[cell] for cell in ids])


def scores(labels, y):
    return {"ari": adjusted_rand_score(labels, y), "nmi": normalized_mutual_info_score(labels, y)}


def main():
    labels = truth()
    summary, per_start = [], []
    for path in sorted(HERE.glob("runs/*/*/*/result.npz")):
        hierarchy, scope, resolution = path.parent.name, path.parent.parent.name, path.parents[2].name
        key = {"resolution": int(resolution), "scope": scope}
        r = np.load(path)
        names = list(r["starts"])
        if hierarchy == "imputed":   # the spectral start is the same for raw and imputed
            summary.append({**key, "hierarchy": "none", "method": "spectral (contacts only)",
                            "rule": "spectral_start", "start": "spectral", "objective": "",
                            **scores(labels, r["init"][0])})
        for method in ("median", "kmeans"):
            objective = r[f"{method}_objective"]
            for i, name in enumerate(names):
                per_start.append({**key, "hierarchy": hierarchy, "method": method, "start": name,
                                  "objective": objective[i],
                                  "init_ari": adjusted_rand_score(labels, r["init"][i]),
                                  **scores(labels, r[f"{method}_labels"][i])})
            for rule, pick in RULES.items():
                i = min(pick(names), key=lambda j: objective[j])
                summary.append({**key, "hierarchy": hierarchy, "method": method, "rule": rule,
                                "start": names[i], "objective": objective[i],
                                **scores(labels, r[f"{method}_labels"][i])})
    refs = HERE / "runs/contact_references/labels.npz"
    if refs.exists():
        for name, y in np.load(refs).items():
            summary.append({"resolution": 1_000_000, "scope": "all", "hierarchy": "none",
                            "method": name, "rule": "", "start": "", "objective": "",
                            **scores(labels, y)})
    for rows, file in ((summary, "summary.tsv"), (per_start, "per_start.tsv")):
        with open(HERE / file, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    for row in summary:
        if row["scope"] == "all":
            print(f"{row['resolution']:>8} {row['hierarchy']:>8} {row['method']:>26} "
                  f"{row['rule']:>15}  ARI {row['ari']:.3f}  NMI {row['nmi']:.3f}")


if __name__ == "__main__":
    main()
