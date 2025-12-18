import sys
from collections import Counter
from Bio import SeqIO

def get_clean_consensus(alignment_file, threshold=0.5):
    align = list(SeqIO.parse(alignment_file, "fasta"))
    if not align: return
    
    length = len(align[0])
    num_seqs = len(align)
    consensus = []

    for i in range(length):
        col = [s.seq[i] for s in align]
        # Count non-gap characters
        non_gaps = [aa for aa in col if aa != "-"]
        
        # If fewer than 50% of sequences have data here, skip it (junk filter)
        if len(non_gaps) / num_seqs < threshold:
            continue
            
        # Majority rule on the remaining AAs
        if non_gaps:
            most_common = Counter(non_gaps).most_common(1)[0][0]
            consensus.append(most_common)

    print("".join(consensus))

if __name__ == "__main__":
    get_clean_consensus(sys.argv[1])
  
