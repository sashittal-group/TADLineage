#!/usr/bin/env python3
"""Post-hoc scoring of run_nussinov_median.py; the only script here that reads stage labels."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from evaluate import truth  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="run")
    args = parser.parse_args()
    out = HERE / args.output
    fit = np.load(out / "fit.npz")
    run = json.loads((out / "run.json").read_text())
    labels = truth()
    majority = np.load(HERE.parent / "final/fit.npz")["labels"]
    result = {
        "ARI": float(adjusted_rand_score(labels, fit["labels"])),
        "NMI": float(normalized_mutual_info_score(labels, fit["labels"])),
        "selected_start": run["selected_start"],
        "ARI_spectral_start": float(adjusted_rand_score(labels, fit["spectral_start"])),
        "ARI_vs_majority_rule_final": float(adjusted_rand_score(majority, fit["labels"])),
        "ARI_majority_rule_final": float(adjusted_rand_score(labels, majority)),
        "per_start": [{"start": s["start"], "objective": s["objective"],
                       "ARI": float(adjusted_rand_score(labels, y))}
                      for s, y in zip(run["starts"], fit["restart_labels"])],
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
