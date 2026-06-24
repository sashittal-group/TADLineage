#!/usr/bin/env python3
"""Run Higashi training and imputation for one chromosome/resolution."""

import os
import pickle

from run_paths import parse_run_args, write_run_config


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def run_training(config_path):
    # Higashi imports torch/OpenMP libraries, so runtime env vars must be set first.
    from higashi.Higashi_wrapper import Higashi

    model = Higashi(config_path)

    # Data processing and model setup must happen before any training stage.
    model.process_data()
    model.prep_model()

    model.train_for_embeddings()

    model.train_for_imputation_nbr_0()
    model.impute_no_nbr()

    model.train_for_imputation_with_nbr()
    model.impute_with_nbr()


def load_label_info(data_dir):
    label_path = os.path.join(data_dir, "label_info.pickle")
    with open(label_path, "rb") as handle:
        return pickle.load(handle)


def print_summary(config_path, paths):
    labels = load_label_info(paths["data_dir"])
    print("Loaded labels keys:", sorted(labels.keys()))
    print("Cells:", len(labels["cell type"]))
    print("Config:", config_path)
    print("Data:", paths["data_dir"])
    print("Temp:", paths["temp_dir"])
    print("Training and imputation finished.")


def main():
    args = parse_run_args("Train Higashi and run imputation for one chromosome and resolution.")
    config_path, paths = write_run_config(args.chromosome, args.resolution)

    run_training(config_path)
    print_summary(config_path, paths)


if __name__ == "__main__":
    main()
