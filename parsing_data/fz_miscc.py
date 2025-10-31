#!/usr/bin/env python3
import csv

cdr3_list_file = "cdr3_list.txt"          # one CDR3 AA per line
igblast_csv    = "igblast_output.csv"     # IgBlast CSV with header
output_prefix  = "cdr3_"                  # output prefix for generated FASTAs
MAX_MISMATCH   = 1                        # allow up to 1 mismatch in match window

def load_cdr3_list(path):
    cdr3s = []
    with open(path) as f:
        for line in f:
            seq = line.strip().upper().replace("*", "")
            if seq:
                cdr3s.append(seq)
    return cdr3s

def mismatch_count(a, b):
    # assumes len(a) == len(b)
    return sum(x != y for x, y in zip(a, b))

def cdr3_matches_full_aa(cdr3, full_aa, max_mm=1):
    """
    Slide a window of len(cdr3) across full_aa.
    Return True if any window has <= max_mm mismatches.
    """
    cdr3 = cdr3.upper()
    full_aa = full_aa.upper()
    L = len(cdr3)
    if L == 0 or len(full_aa) < L:
        return False
    for i in range(len(full_aa) - L + 1):
        window = full_aa[i:i+L]
        if mismatch_count(cdr3, window) <= max_mm:
            return True
    return False

def read_igblast_rows(path):
    """
    Returns list of dicts containing:
        nt: full nucleotide seq from column 2
        aa: full amino acid seq from column 3
    We assume:
      col[0] = something else (id, etc)
      col[1] = nt sequence
      col[2] = aa sequence
    """
    rows = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header row
        for r in reader:
            if len(r) < 3:
                continue
            nt_seq = r[1].strip()
            aa_seq = r[2].strip()
            if nt_seq and aa_seq:
                rows.append({"nt": nt_seq, "aa": aa_seq})
    return rows

def write_fasta(filename, seqs, header_prefix):
    """
    seqs: list of sequences (strings)
    header_prefix: string for fasta IDs
    Writes >header_prefix_idx
    """
    if not seqs:
        return
    with open(filename, "w") as fh:
        for i, s in enumerate(seqs, start=1):
            fh.write(f">{header_prefix}_{i}\n")
            # wrap lines at 60/80 chars for readability (optional but nicer)
            width = 80
            for start in range(0, len(s), width):
                fh.write(s[start:start+width] + "\n")

def main():
    # 1. load inputs
    cdr3_list = load_cdr3_list(cdr3_list_file)
    ig_rows   = read_igblast_rows(igblast_csv)

    # 2. for each cdr3, collect matched parent sequences
    for idx, cdr3 in enumerate(cdr3_list, start=1):
        aa_hits = []
        nt_hits = []

        for row in ig_rows:
            if cdr3_matches_full_aa(cdr3, row["aa"], MAX_MISMATCH):
                aa_hits.append(row["aa"])
                nt_hits.append(row["nt"])

        # 3. write two FASTAs for this CDR3
        aa_out = f"{output_prefix}{idx:03d}_aa.fasta"
        nt_out = f"{output_prefix}{idx:03d}_nt.fasta"

        write_fasta(aa_out, aa_hits, header_prefix=f"cdr3_{idx:03d}_aa")
        write_fasta(nt_out, nt_hits, header_prefix=f"cdr3_{idx:03d}_nt")

        print(f"CDR3 {idx:03d} ({cdr3}): {len(aa_hits)} hits -> {aa_out}, {nt_out}")

if __name__ == "__main__":
    main()
