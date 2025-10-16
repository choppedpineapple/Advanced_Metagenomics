import csv
from collections import Counter, defaultdict

def consensus(seqs):
    """Return a simple consensus sequence from a list of nucleotide sequences."""
    max_len = max(len(s) for s in seqs)
    cons = ""
    for i in range(max_len):
        column = [s[i] for s in seqs if i < len(s)]
        # Remove gaps and ambiguous bases before counting
        column = [b for b in column if b not in ['-', 'N', 'n']]
        if not column:
            cons += 'N'
        else:
            cons += Counter(column).most_common(1)[0][0]
    return cons

# === Input files ===
input_file = "input.csv"             # Your file with CDR3 AAs in 2nd column
igblast_file = "igblast_output.csv"  # IgBlast output file
output_fasta = "cdr3_consensus.fasta"

# === Step 1: Read the CDR3 amino acid sequences from input.csv ===
cdr3_list = set()
with open(input_file) as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if len(row) >= 2 and row[1].strip():
            aa = row[1].strip().replace('*', '')  # remove stop codon markers
            cdr3_list.add(aa)

# === Step 2: Read the IgBlast CSV and map (V, J, CDR3) → nucleotide sequences ===
cdr3_to_nucs = defaultdict(list)
with open(igblast_file) as f:
    reader = csv.DictReader(f)
    print("Detected columns:", reader.fieldnames)  # helps you confirm column names

    for row in reader:
        cdr3_aa = row.get("cdr3_aa", "").strip().replace('*', '')
        v_call = row.get("v_call", "").split(',')[0].strip()  # take first if multiple
        j_call = row.get("j_call", "").split(',')[0].strip()
        seq_nt = row.get("sequence", "").strip()

        # If no matches, skip
        if not (cdr3_aa and v_call and j_call and seq_nt):
            continue

        # Match only if this CDR3 is in our input list
        if cdr3_aa in cdr3_list:
            key = (v_call, j_call, cdr3_aa)
            cdr3_to_nucs[key].append(seq_nt)

# === Step 3: Generate consensus per (V, J, CDR3) and write to FASTA ===
with open(output_fasta, "w") as out:
    for (v, j, cdr3), seqs in cdr3_to_nucs.items():
        if len(seqs) == 1:
            cons = seqs[0]
            label = f"{cdr3}_{v}_{j}_single"
        else:
            cons = consensus(seqs)
            label = f"{cdr3}_{v}_{j}_consensus"

        cons = cons.replace('-', '').replace('N', '')  # final cleanup
        out.write(f">{label}\n{cons}\n")

print(f"\n✅ Done! Consensus sequences written to {output_fasta}")
