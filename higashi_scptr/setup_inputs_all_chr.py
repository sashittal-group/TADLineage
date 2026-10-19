#!/usr/bin/env python3
"""Prepare per-cell Higashi input files for one all-chromosome model."""

import os
import pickle

from run_paths import ALL_CHROMOSOME_RUN, ALL_CHROMOSOMES, load_base_config, parse_run_args, run_paths
from setup_inputs import clamp_position, list_cell_type_dirs, source_data_root


def genome_size_path():
    return load_base_config()["genome_reference_path"]


def load_chrom_sizes(path):
    chrom_sizes = {}
    with open(path, "r") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 2:
                chrom_sizes[parts[0]] = int(parts[1])
    return chrom_sizes


def convert_contact_file(source_path, target_path, chrom_sizes, allowed_chromosomes):
    """Keep intra-chromosomal contacts for all requested chromosomes."""
    kept = 0
    clamped = 0
    coverage_by_chrom = {chrom: 0 for chrom in allowed_chromosomes}

    with open(source_path, "r") as source_file, open(target_path, "w") as target_file:
        for line in source_file:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            chrom1, pos1, chrom2, pos2 = parts[1], parts[2], parts[4], parts[5]
            if chrom1 != chrom2 or chrom1 not in allowed_chromosomes:
                continue

            original_pos1, original_pos2 = int(pos1), int(pos2)
            chrom_size = chrom_sizes[chrom1]
            pos1 = clamp_position(original_pos1, chrom_size)
            pos2 = clamp_position(original_pos2, chrom_size)
            if pos1 != original_pos1 or pos2 != original_pos2:
                clamped += 1

            target_file.write(f"{chrom1}\t{pos1}\t{chrom2}\t{pos2}\n")
            kept += 1
            coverage_by_chrom[chrom1] += 1

    return kept, clamped, coverage_by_chrom


def process_inputs(data_root, processed_dir, chrom_sizes, allowed_chromosomes):
    filelist = []
    cell_types = []
    cell_names = []
    coverage = []
    coverage_by_chrom = []
    clamped_counts = []

    for cell_type in list_cell_type_dirs(data_root):
        cell_type_dir = os.path.join(data_root, cell_type)
        for filename in sorted(os.listdir(cell_type_dir)):
            if not filename.endswith(".txt"):
                continue

            source_path = os.path.join(cell_type_dir, filename)
            target_path = os.path.join(processed_dir, f"{cell_type}__{filename}.tsv")
            kept, clamped, by_chrom = convert_contact_file(
                source_path,
                target_path,
                chrom_sizes,
                allowed_chromosomes,
            )

            filelist.append(os.path.abspath(target_path))
            cell_types.append(cell_type)
            cell_names.append(os.path.splitext(filename)[0])
            coverage.append(kept)
            coverage_by_chrom.append(by_chrom)
            clamped_counts.append(clamped)

    return filelist, cell_types, cell_names, coverage, coverage_by_chrom, clamped_counts


def write_filelist(out_dir, filelist):
    filelist_path = os.path.join(out_dir, "filelist.txt")
    with open(filelist_path, "w") as filelist_file:
        for path in filelist:
            filelist_file.write(path + "\n")
    return filelist_path


def write_label_info(out_dir, cell_types, cell_names, coverage, coverage_by_chrom, clamped_counts):
    label_info = {
        "cell type": cell_types,
        "cell_name_higashi": cell_names,
        "coverage": coverage,
        "coverage_by_chrom": coverage_by_chrom,
        "clamped_contacts": clamped_counts,
    }
    label_path = os.path.join(out_dir, "label_info.pickle")
    with open(label_path, "wb") as label_file:
        pickle.dump(label_info, label_file)
    return label_path


def main():
    args = parse_run_args("Prepare Higashi input files for one all-chromosome model.")
    paths = run_paths(ALL_CHROMOSOME_RUN, args.resolution)

    out_dir = paths["data_dir"]
    processed_dir = paths["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    chrom_sizes = load_chrom_sizes(genome_size_path())
    missing = [chrom for chrom in ALL_CHROMOSOMES if chrom not in chrom_sizes]
    if missing:
        raise KeyError(f"Missing chromosomes from {genome_size_path()}: {', '.join(missing)}")

    filelist, cell_types, cell_names, coverage, coverage_by_chrom, clamped_counts = process_inputs(
        source_data_root(),
        processed_dir,
        chrom_sizes,
        ALL_CHROMOSOMES,
    )
    filelist_path = write_filelist(out_dir, filelist)
    label_path = write_label_info(
        out_dir,
        cell_types,
        cell_names,
        coverage,
        coverage_by_chrom,
        clamped_counts,
    )

    print(f"Wrote {filelist_path}")
    print(f"Wrote {label_path}")
    print(f"Chromosomes: {', '.join(ALL_CHROMOSOMES)}")
    print(f"Resolution: {paths['resolution']}")
    print(f"Clamped contacts: {sum(clamped_counts)}")
    print(f"Processed {len(filelist)} cells into {processed_dir}")


if __name__ == "__main__":
    main()
