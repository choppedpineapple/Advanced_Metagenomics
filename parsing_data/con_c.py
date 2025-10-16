from collections import Counter

def read_sequences(filename):
    with open(filename) as f:
        return [line.strip().upper() for line in f if line.strip()]

def identity(s1, s2):
    matches = sum(a == b for a, b in zip(s1, s2))
    return matches / max(len(s1), len(s2))

def consensus(seqs):
    max_len = max(len(s) for s in seqs)
    cons = ""
    for i in range(max_len):
        column = [s[i] for s in seqs if i < len(s)]
        cons += Counter(column).most_common(1)[0][0]
    return cons

def cluster_sequences(seqs, threshold=0.9):
    clusters = []
    for s in seqs:
        placed = False
        for cluster in clusters:
            if identity(s, cluster[0]) >= threshold:
                cluster.append(s)
                placed = True
                break
        if not placed:
            clusters.append([s])
    return clusters

if __name__ == "__main__":
    seqs = read_sequences("sequences.txt")
    clusters = cluster_sequences(seqs, threshold=0.9)
    
    print(f"Total clusters: {len(clusters)}")
    for i, cluster in enumerate(clusters, 1):
        print(f">Consensus_{i}")
        print(consensus(cluster))
