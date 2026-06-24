#!/usr/bin/env bash
set -euo pipefail

out_dir="clustering_figures"
mkdir -p "$out_dir"

for seed in $(seq 50 150); do
    for max_iter in 15 30 50; do

        echo "Running seed $seed with max_iter $max_iter..."
        python clustering.py --random-state "$seed" --max_iter "$max_iter"
        fig="clustering_result_${seed}_${max_iter}.png"
        if [[ -f "$fig" ]]; then
            mv "$fig" "$out_dir/"
        else
            echo "Warning: expected $fig not found"
        fi
    done
done

echo "Done. Figures saved to $out_dir/"