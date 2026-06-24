#!/usr/bin/env python3

import argparse
import csv
import glob
import json
from pathlib import Path


ALLOWED_RESOLUTIONS = {"500kb", "1000kb"}


def load_records(input_glob):
    records = []
    for path_str in sorted(glob.glob(input_glob, recursive=True)):
        path = Path(path_str)
        if not path.is_file():
            continue
        if not any(part in ALLOWED_RESOLUTIONS for part in path.parts):
            continue
        with open(path) as handle:
            record = json.load(handle)
        record["json_path"] = str(path)
        records.append(record)
    return records


def flatten_record(record):
    flattened = dict(record)
    flattened["cluster_majorities"] = json.dumps(record.get("cluster_majorities", {}), sort_keys=True)
    flattened["cluster_summary"] = json.dumps(record.get("cluster_summary", {}), sort_keys=True)
    flattened["filtered_paths"] = json.dumps(record.get("filtered_paths", []))
    flattened["imputed_paths"] = json.dumps(record.get("imputed_paths", []))
    return flattened


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-run JSON outputs into one CSV.")
    parser.add_argument("--input-glob", default="results/scptr_runs/**/*.json", help="Glob for per-run JSON files")
    parser.add_argument("--output-csv", default="results/scptr_tree_distance/all_tree_distances.csv", help="Output CSV path")
    args = parser.parse_args()

    records = [flatten_record(record) for record in load_records(args.input_glob)]
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        raise SystemExit(f"No JSON files found for pattern: {args.input_glob}")

    fieldnames = sorted({key for record in records for key in record.keys()})
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})

    print(f"Saved aggregated CSV: {output_path}")


if __name__ == "__main__":
    main()