#!/usr/bin/env python3
"""Prepare per-cell Higashi input files for one chromosome."""

import os
import pickle

from run_paths import load_base_config, parse_run_args, run_paths


def source_data_root():
    workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(workspace, "scptr", "cell_differentiation")


def list_cell_type_dirs(data_root):
    return sorted(
        name
        for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name))
    )


def genome_size_path():
    return load_base_config()["genome_reference_path"]


def load_chrom_sizes(path):
    chrom_sizes = {}
    with open(path, "r") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            chrom_sizes[parts[0]] = int(parts[1])
    return chrom_sizes


def clamp_position(position, chrom_size):
    position = int(position)
    if position < 0:
        return 0
    if position >= chrom_size:
        return chrom_size - 1
    return position


def convert_contact_file(source_path, target_path, chromosome, chrom_size):
    """Keep only intra-chromosomal contacts for the requested chromosome."""
    kept = 0
    clamped = 0
    with open(source_path, "r") as source_file, open(target_path, "w") as target_file:
        for line in source_file:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            chrom1, pos1, chrom2, pos2 = parts[1], parts[2], parts[4], parts[5]
            if chrom1 != chromosome or chrom2 != chromosome:
                continue

            original_pos1, original_pos2 = int(pos1), int(pos2)
            pos1 = clamp_position(original_pos1, chrom_size)
            pos2 = clamp_position(original_pos2, chrom_size)
            if pos1 != original_pos1 or pos2 != original_pos2:
                clamped += 1

            target_file.write(f"{chrom1}\t{pos1}\t{chrom2}\t{pos2}\n")
            kept += 1
    return kept, clamped


def process_inputs(data_root, processed_dir, chromosome, chrom_size):
    filelist = []
    cell_types = []
    cell_names = []
    coverage = []
    clamped_counts = []

    for cell_type in list_cell_type_dirs(data_root):
        cell_type_dir = os.path.join(data_root, cell_type)
        for filename in sorted(os.listdir(cell_type_dir)):
            if not filename.endswith(".txt"):
                continue

            source_path = os.path.join(cell_type_dir, filename)
            target_path = os.path.join(processed_dir, f"{cell_type}__{filename}.tsv")
            kept, clamped = convert_contact_file(source_path, target_path, chromosome, chrom_size)

            filelist.append(os.path.abspath(target_path))
            cell_types.append(cell_type)
            cell_names.append(os.path.splitext(filename)[0])
            coverage.append(kept)
            clamped_counts.append(clamped)

    return filelist, cell_types, cell_names, coverage, clamped_counts


def write_filelist(out_dir, filelist):
    filelist_path = os.path.join(out_dir, "filelist.txt")
    with open(filelist_path, "w") as filelist_file:
        for path in filelist:
            filelist_file.write(path + "\n")
    return filelist_path


def write_label_info(out_dir, cell_types, cell_names, coverage, clamped_counts):
    label_info = {
        "cell type": cell_types,
        "cell_name_higashi": cell_names,
        "coverage": coverage,
        "clamped_contacts": clamped_counts,
    }
    label_path = os.path.join(out_dir, "label_info.pickle")
    with open(label_path, "wb") as label_file:
        pickle.dump(label_info, label_file)
    return label_path


def main():
    args = parse_run_args("Prepare Higashi input files for one chromosome and resolution.")
    paths = run_paths(args.chromosome, args.resolution)

    out_dir = paths["data_dir"]
    processed_dir = paths["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    chrom_sizes = load_chrom_sizes(genome_size_path())
    if paths["chromosome"] not in chrom_sizes:
        raise KeyError(f"{paths['chromosome']} is missing from {genome_size_path()}")

    filelist, cell_types, cell_names, coverage, clamped_counts = process_inputs(
        source_data_root(),
        processed_dir,
        paths["chromosome"],
        chrom_sizes[paths["chromosome"]],
    )
    filelist_path = write_filelist(out_dir, filelist)
    label_path = write_label_info(out_dir, cell_types, cell_names, coverage, clamped_counts)

    print(f"Wrote {filelist_path}")
    print(f"Wrote {label_path}")
    print(f"Chromosome: {paths['chromosome']}")
    print(f"Resolution: {paths['resolution']}")
    print(f"Clamped contacts: {sum(clamped_counts)}")
    print(f"Processed {len(filelist)} cells into {processed_dir}")


if __name__ == "__main__":
    main()
