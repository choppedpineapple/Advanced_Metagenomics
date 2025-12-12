#!/usr/bin/env python3

import os
import sys

LINKER_TSV = "linker.tsv"


def load_linkers(path):
    linkers = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            k, v = line.split("\t", 1)
            linkers[k] = v
    return linkers


def extract_core_id(seq_id):
    # sequence_1627_before → 1627
    parts = seq_id.split("_")
    if len(parts) < 3:
        return None
    return parts[1]


def find_linker(core_id, linkers):
    # prefer fwd, fallback to any containing the id
    fwd = f"linker_{core_id}_fwd"
    if fwd in linkers:
        return linkers[fwd]

    for k, v in linkers.items():
        if f"_{core_id}_" in k:
            return v

    return None


def process_txt(txt_path, linkers, out_dir):
    basename = os.path.basename(txt_path)
    out_fasta = os.path.join(
        out_dir, basename.rsplit(".", 1)[0] + ".fasta"
    )

    with open(txt_path) as fin, open(out_fasta, "w") as fout:
        header = fin.readline()  # skip header

        for line in fin:
            line = line.strip()
            if not line:
                continue

            cols = line.split()
            if len(cols) < 4:
                continue

            id1, seq1, id2, seq2 = cols[:4]

            core_id = extract_core_id(id1)
            if core_id is None:
                continue

            linker = find_linker(core_id, linkers)
            if linker is None:
                continue

            if id1.endswith("_before") and id2.endswith("_after"):
                full_seq = seq1 + linker + seq2
            elif id1.endswith("_after") and id2.endswith("_before"):
                full_seq = seq2 + linker + seq1
            else:
                continue

            fout.write(f">sequence_{core_id}\n")
            fout.write(full_seq + "\n")


def main(txt_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    linkers = load_linkers(LINKER_TSV)

    for fname in os.listdir(txt_dir):
        if not fname.endswith(".txt"):
            continue
        process_txt(os.path.join(txt_dir, fname), linkers, out_dir)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python join_scFv.py <txt_dir> <out_dir>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
