#!/usr/bin/env python3
"""Shared CLI arguments and run-specific paths for Higashi scptr jobs."""

import argparse
import json
import os


ROOT = os.path.abspath(os.path.dirname(__file__))
BASE_CONFIG_PATH = os.path.join(ROOT, "config.json")
DEFAULT_CHROMOSOME = "chr1"
DEFAULT_RESOLUTION = 1000000
ALL_CHROMOSOMES = [f"chr{i}" for i in range(1, 20)] + ["chrX"]
ALL_CHROMOSOME_RUN = "all_higashi"


def normalize_chromosome(chromosome):
    chromosome = str(chromosome)
    if chromosome.lower() in {"all", "genome"}:
        return "all"
    if chromosome.lower() in {ALL_CHROMOSOME_RUN, "all-chr", "all_chr"}:
        return ALL_CHROMOSOME_RUN
    return chromosome if chromosome.startswith("chr") else f"chr{chromosome}"


def add_run_arguments(parser):
    parser.add_argument(
        "--chromosome",
        "--chrom",
        default=DEFAULT_CHROMOSOME,
        help="Chromosome to process, for example chr1 or 1.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Bin resolution in base pairs.",
    )
    return parser


def parse_run_args(description):
    parser = argparse.ArgumentParser(description=description)
    add_run_arguments(parser)
    return parser.parse_args()


def run_paths(chromosome, resolution):
    """Return all filesystem locations for one chromosome/resolution run."""
    chrom = normalize_chromosome(chromosome)
    resolution_label = str(int(resolution))
    data_dir = os.path.join(ROOT, "data", resolution_label, chrom)
    temp_dir = os.path.join(ROOT, "temp", resolution_label, chrom)
    results_dir = os.path.join(ROOT, "results", resolution_label, chrom)
    return {
        "chromosome": chrom,
        "resolution": int(resolution),
        "resolution_label": resolution_label,
        "data_dir": data_dir,
        "processed_dir": os.path.join(data_dir, "processed"),
        "temp_dir": temp_dir,
        "results_dir": results_dir,
        "config_path": os.path.join(results_dir, "config.json"),
    }


def load_base_config():
    with open(BASE_CONFIG_PATH, "r") as handle:
        return json.load(handle)


def write_run_config(chromosome, resolution):
    """Write a Higashi config copy pointed at the run-specific directories."""
    paths = run_paths(chromosome, resolution)
    os.makedirs(paths["results_dir"], exist_ok=True)
    os.makedirs(paths["temp_dir"], exist_ok=True)

    config = load_base_config()
    config.update(
        {
            "data_dir": paths["data_dir"],
            "temp_dir": paths["temp_dir"],
            "chrom_list": [paths["chromosome"]],
            "impute_list": [paths["chromosome"]],
            "resolution": paths["resolution"],
            "resolution_cell": paths["resolution"],
            "minimum_distance": paths["resolution"],
            "config_name": f"scptr_{paths['chromosome']}_{paths['resolution_label']}",
        }
    )

    with open(paths["config_path"], "w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    return paths["config_path"], paths


def write_all_chromosome_run_config(resolution):
    """Write a Higashi config for one model trained on all chromosomes."""
    paths = run_paths(ALL_CHROMOSOME_RUN, resolution)
    os.makedirs(paths["results_dir"], exist_ok=True)
    os.makedirs(paths["temp_dir"], exist_ok=True)

    config = load_base_config()
    config.update(
        {
            "data_dir": paths["data_dir"],
            "temp_dir": paths["temp_dir"],
            "chrom_list": ALL_CHROMOSOMES,
            "impute_list": ALL_CHROMOSOMES,
            "resolution": paths["resolution"],
            "resolution_cell": paths["resolution"],
            "minimum_distance": paths["resolution"],
            "config_name": f"scptr_{ALL_CHROMOSOME_RUN}_{paths['resolution_label']}",
        }
    )

    with open(paths["config_path"], "w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    return paths["config_path"], paths
