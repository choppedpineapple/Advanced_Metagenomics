import sys
from collections import Counter
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def get_smart_consensus(alignment_file, threshold=0.5, subclone_limit=0.4):
    align = list(SeqIO.parse(alignment_file, "fasta"))
    if not align: return
    
    length = len(align[0])
    num_seqs = len(align)
    
    main_cons = []
    alt_cons = []
    has_subclone = False

    for i in range(length):
        col = [s.seq[i] for s in align]
        non_gaps = [aa for aa in col if aa != "-"]
        
        # 1. Junk Filter
        if len(non_gaps) / num_seqs < threshold:
            continue
            
        counts = Counter(non_gaps).most_common(2)
        
        # Winner
        winner = counts[0][0]
        main_cons.append(winner)
        
        # 2. Sub-clone Detection
        # If the runner-up is very close to the winner (e.g. 45% vs 50%)
        if len(counts) > 1:
            runner_up, count = counts[1]
            if (count / len(non_gaps)) >= subclone_limit:
                alt_cons.append(runner_up)
                has_subclone = True
            else:
                alt_cons.append(winner)
        else:
            alt_cons.append(winner)

    # Prepare outputs
    results = [("Main", "".join(main_cons))]
    if has_subclone:
        results.append(("Subclone", "".join(alt_cons)))
        
    return results

if __name__ == "__main__":
    # Simplified for your Bash loop
    file_path = sys.argv[1]
    variants = get_smart_consensus(file_path)
    
    for suffix, seq in variants:
        print(f">{suffix}\n{seq}")
