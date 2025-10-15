from collections import Counter

def read_sequences(filename):
    with open(filename) as f:
        return [line.strip().upper() for line in f if line.strip()]

def consensus_sequence(seqs):
    max_len = max(len(s) for s in seqs)
    consensus = ""
    for i in range(max_len):
        column = [s[i] for s in seqs if i < len(s)]
        consensus += Counter(column).most_common(1)[0][0]
    return consensus

if __name__ == "__main__":
    seqs = read_sequences("sequences.txt")  # change filename
    print(consensus_sequence(seqs))
