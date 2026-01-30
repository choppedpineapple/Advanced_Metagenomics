#!/usr/bin/env python3
import sys
import gzip
from collections import Counter
import edlib


def open_maybe_gz(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def fasta_reader(path):
    with open_maybe_gz(path) as fh:
        header = None
        seq_chunks = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks).upper()
                header = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            yield header, "".join(seq_chunks).upper()


def best_variant(seq, linker, linker_rc, max_ed):
    a = edlib.align(linker, seq, mode="HW", task="locations", k=max_ed)
    b = edlib.align(linker_rc, seq, mode="HW", task="locations", k=max_ed)

    candidates = []
    if a["editDistance"] != -1 and a.get("locations"):
        s, e = min(a["locations"], key=lambda x: x[0])
        candidates.append((a["editDistance"], s, e))
    if b["editDistance"] != -1 and b.get("locations"):
        s, e = min(b["locations"], key=lambda x: x[0])
        candidates.append((b["editDistance"], s, e))

    if not candidates:
        return None

    ed, s, e = min(candidates, key=lambda x: (x[0], x[1]))
    return seq[s:e + 1]


def main():
    fasta_path = sys.argv[1]
    linker = sys.argv[2].upper()
    linker_rc = sys.argv[3].upper()
    out_tsv = sys.argv[4]
    max_ed = int(sys.argv[5]) if len(sys.argv) > 5 else 4

    counts = Counter()
    for _, seq in fasta_reader(fasta_path):
        v = best_variant(seq, linker, linker_rc, max_ed)
        if v is not None:
            counts[v] += 1

    with open(out_tsv, "w") as out:
        out.write("variant_seq\tcount\n")
        for var, c in counts.most_common():
            out.write(f"{var}\t{c}\n")


if __name__ == "__main__":
    main()
