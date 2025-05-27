Below are some of the most “native” data-structure idioms you can use in pure bash when manipulating FASTQ files (or any other line-based bioinformatics data).  I’ll introduce each one, give a real-world analogy, and then show you a tiny working example.

---

## 1. Streams/Pipelines  
**Analogy:** A factory conveyor belt.  Data items (reads) flow by and get processed on the fly, one after the other.

### Why use it?  
- FASTQ files are huge—don’t try to load the whole thing into memory.
- You can chain small tools (`zcat`, `grep`, `awk`, your own `while` loop) like assembly‐line stations.

### Example: compute average read length

```bash
zcat reads.fastq.gz \
| awk 'NR%4==2 { total+=length($0); count++ } 
       END { printf "Average length: %.2f\n", total/count }'
```

Here `NR%4==2` picks out every 2nd line of each 4‐line FASTQ record (the sequence), and `awk` holds just two scalars (`total`, `count`) in memory.

---

## 2. Indexed Arrays  
**Analogy:** A row of numbered mailboxes on a wall—box 0, box 1, box 2, …  

Bash arrays let you store small to moderate amounts of data in RAM, indexed by integer.

```bash
# declare an array
declare -a read_lengths

# walk the FASTQ, collect each read length into the array
i=0
zcat reads.fastq.gz | while read -r h; read -r seq; read -r p; read -r qual; do
  read_lengths[i]=${#seq}
  (( i++ ))
done

# later: compute stats on "${read_lengths[@]}"
sum=0
for L in "${read_lengths[@]}"; do
  (( sum += L ))
done
echo "Got $i reads; avg length = $(( sum / i ))"
```

**Caveat:** Arrays live in memory—if you have tens of millions of reads, don’t try this.

---

## 3. Associative Arrays (Bash 4+)  
**Analogy:** A dictionary or phonebook: you look up a “key” (e.g. the k-mer “ATGC”) and get back a “value” (how many times you’ve seen it).

```bash
# declare an associative array
declare -A kmer_counts

zcat reads.fastq.gz \
| while read -r h; read -r seq; read -r p; read -r qual; do
    # slide a 4-mer window over the sequence
    for (( i=0; i<=${#seq}-4; i++ )); do
      k=${seq:i:4}
      (( kmer_counts[$k]++ ))
    done
  done

# print top 5 most frequent 4-mers
for k in "${!kmer_counts[@]}"; do
  echo -e "${k}\t${kmer_counts[$k]}"
done \
| sort -k2,2nr \
| head -5
```

This is the classic k-mer counting trick in pure bash.  Keys are stored sparsely—only observed k-mers live in the map.

---

## 4. Stacks & Queues (via Arrays)  
**Analogy:** A stack of plates (LIFO) or a waiting line/queue (FIFO).  

Useful if you need to synchronize paired-end FASTQ files, or hold a small buffer of reads to compare.

### FIFO Queue Example (synchronizing two gzipped FASTQs)

```bash
# declare two FIFOs
mkfifo q1 q2

# feed each compressed file into its pipe
zcat pe1.fastq.gz > q1 &
zcat pe2.fastq.gz > q2 &

# now read from both in parallel
while true; do
  read -r h1 < q1 || break
  read -r s1 < q1
  read -r p1 < q1
  read -r ql1 < q1

  read -r h2 < q2 || break
  read -r s2 < q2
  read -r p2 < q2
  read -r ql2 < q2

  # process the pair (h1, s1, …) & (h2, s2, …)
  echo "R1: $h1  R2: $h2"
done
```

When you’re done, `rm q1 q2`.  The FIFOs let you treat two endless streams as “buffers” you can pull from in lock‐step.

---

## 5. Temporary Files as Persistent Storage  
**Analogy:** Using sticky notes on a whiteboard—scratch data you’ll revisit later.  

Sometimes the simplest “data structure” is a small temp file:

```bash
# extract headers to a temp file
zcat reads.fastq.gz | sed -n '1~4p' > headers.txt

# later: grep or join against headers.txt
```

Combine with `sort`/`uniq` for counting or with `join` for lookups.

---

## Putting It All Together: QC Summary Script

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1) Count reads + average length (stream + awk)
echo -n "Reads & avg length: "
zcat reads.fastq.gz \
| awk 'NR%4==2 { total+=length; count++ }
       END { printf "%d reads, avg_len=%.1f\n", count, total/count }'

# 2) Per-base quality score distribution (assoc array)
declare -A qd
pos=0
zcat reads.fastq.gz \
| while read -r h; read -r s; read -r p; read -r qual; do
    for ((i=0;i<${#qual};i++)); do
      q=${qual:i:1}
      (( qd["$i,$q"]++ ))
    done
  done

echo "Position\tPhred\tCount"
for key in "${!qd[@]}"; do
  IFS=, read pos q <<<"$key"
  echo -e "${pos}\t${q}\t${qd[$key]}"
done | sort -k1,1n -k2,2

# 3) k-mer summary (assoc array)
declare -A kc
zcat reads.fastq.gz \
| awk 'NR%4==2' \
| while read -r seq; do
    for ((i=0;i<=${#seq}-5;i++)); do
      k=${seq:i:5}
      (( kc[$k]++ ))
    done
  done

echo "Top 5 5-mers:"
for k in "${!kc[@]}"; do
  echo -e "${k}\t${kc[$k]}"
done | sort -k2,2nr | head -5
```

---

### Key Takeaways

- **Pipelines/streams** are your bread & butter—minimal memory, maximal speed.
- **Arrays** for small, indexed collections.
- **Associative arrays** for maps (k-mer counts, quality tallies, GC buckets).
- **FIFOs/queues/stacks** when you really need to buffer or synchronize streams.
- **Temp files** for on-disk scratch space you can sort/join/grep later.

With these “data structures” in your bash toolkit, you can tackle most day-to-day FASTQ munging tasks without ever leaving the shell.
