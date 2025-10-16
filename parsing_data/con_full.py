import csv
from collections import Counter, defaultdict

def consensus(seqs):
    max_len = max(len(s) for s in seqs)
    cons = ""
    for i in range(max_len):
        column = [s[i] for s in seqs if i < len(s)]
        cons += Counter(column).most_common(1)[0][0]
    return cons

# File names
input_file = "input.csv"
igblast_file = "igblast_output.csv"

# --- Read CDR3 amino acids from the first file ---
cdr3_list = set()
with open(input_file) as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if len(row) >= 2 and row[1].strip():
            cdr3_list.add(row[1].strip())

# --- Parse IgBlast output ---
with open(igblast_file) as f:
    reader = csv.DictReader(f)
    cdr3_to_nucs = defaultdict(list)
    for row in reader:
        cdr3_aa = row.get("cdr3_aa", "").strip()
        seq_nt = row.get("sequence", "").strip()
        if cdr3_aa in cdr3_list and seq_nt:
            cdr3_to_nucs[cdr3_aa].append(seq_nt)

# --- Generate consensus for each matching CDR3 ---
for cdr3, seqs in cdr3_to_nucs.items():
    if len(seqs) > 1:
        cons = consensus(seqs)
        print(f">{cdr3}_consensus")
        print(cons)
    else:
        print(f">{cdr3}_single")
        print(seqs[0])
