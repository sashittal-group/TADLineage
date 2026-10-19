#!/usr/bin/env python3
"""Run one Higashi embedding model using all configured chromosomes."""

import os
import pickle

from run_paths import ALL_CHROMOSOMES, parse_run_args, write_all_chromosome_run_config


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def run_embedding_training(config_path):
    # Higashi imports torch/OpenMP libraries, so runtime env vars must be set first.
    from higashi.Higashi_wrapper import Higashi

    model = Higashi(config_path)
    model.process_data()
    model.prep_model()
    model.train_for_embeddings()


def load_label_info(data_dir):
    label_path = os.path.join(data_dir, "label_info.pickle")
    with open(label_path, "rb") as handle:
        return pickle.load(handle)


def print_summary(config_path, paths):
    labels = load_label_info(paths["data_dir"])
    print("Loaded labels keys:", sorted(labels.keys()))
    print("Cells:", len(labels["cell type"]))
    print("Chromosomes:", ", ".join(ALL_CHROMOSOMES))
    print("Config:", config_path)
    print("Data:", paths["data_dir"])
    print("Temp:", paths["temp_dir"])
    print("All-chromosome embedding training finished.")


def main():
    args = parse_run_args("Train one Higashi embedding model with all chromosomes.")
    config_path, paths = write_all_chromosome_run_config(args.resolution)
    run_embedding_training(config_path)
    print_summary(config_path, paths)


if __name__ == "__main__":
    main()
