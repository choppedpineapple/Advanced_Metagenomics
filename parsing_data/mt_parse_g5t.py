#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-threaded FASTQ parser (no Biopython).
- Python 3.14+ free-threaded (no-GIL) friendly.
- Accepts gzipped or plain FASTQ via sys.argv[1].
- Uses threads to parallelize CPU work (GC%, quality stats, length histogram).
- Streaming + batching to keep memory bounded.

Run (true multithreading):
    python -X gil=0 fastq_mt_parser.py reads.fastq.gz

If you run a regular GIL build, it still works—just less speed-up.
"""

from __future__ import annotations
import sys, os, io, gzip, math
from collections import Counter
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Tuple

# --- Config ---
N_THREADS = 4               # you said you have 4 cores/threads
BATCH_RECORDS = 4096        # records per batch; tune for memory/throughput
PRINT_PER_CYCLE = 50        # print first N cycles' mean Q for a quick glance

# --------- Utilities ---------
def open_maybe_gzip(path: str):
    """
    Open file as binary; auto-detect gzip via magic bytes.
    Returns a buffered binary file-like object (readline -> bytes).
    """
    # If it's stdin-like, fall back
    if path == "-" or path == "/dev/stdin":
        return sys.stdin.buffer

    # Quick peek to detect gzip magic
    f = open(path, "rb")
    try:
        head = f.peek(2) if hasattr(f, "peek") else f.read(2)
    except Exception:
        f.close()
        # If peek fails for some reason, try gzip by extension
        if path.endswith(".gz"):
            return gzip.open(path, "rb")
        return open(path, "rb")
    if head[:2] == b"\x1f\x8b":
        f.close()
        return gzip.open(path, "rb")
    # else reuse the already-opened file
    return f

def fastq_batches(path: str, batch_size: int = BATCH_RECORDS) -> Iterable[List[Tuple[bytes, bytes, bytes]]]:
    """
    Stream FASTQ as batches of (header, seq, qual) bytes tuples (newline-stripped).
    Minimal validation (expects 4-line records).
    """
    with open_maybe_gzip(path) as fh:
        readline = fh.readline
        batch: List[Tuple[bytes, bytes, bytes]] = []
        # Reading as bytes is faster; strip \r?\n
        while True:
            h = readline()
            if not h:
                if batch:
                    yield batch
                break
            s = readline(); p = readline(); q = readline()
            if not (s and p and q):
                sys.stderr.write("WARN: Incomplete FASTQ record at EOF — ignoring trailing lines.\n")
                if batch:
                    yield batch
                break

            # Light validation (optional; costs a tiny bit)
            # if not h.startswith(b"@") or not p.startswith(b"+"):
            #     sys.stderr.write("WARN: FASTQ formatting anomaly detected.\n")

            h = h.rstrip(b"\r\n")
            s = s.rstrip(b"\r\n")
            q = q.rstrip(b"\r\n")
            # We don't need the '+' line contents for stats, so we don't store it.
            batch.append((h, s, q))

            if len(batch) >= batch_size:
                yield batch
                batch = []

# --------- Statistics ---------
@dataclass
class Stats:
    reads: int = 0
    bases: int = 0
    gc: int = 0
    ns: int = 0
    total_q: int = 0  # sum of all (phred) qualities across all bases
    length_hist: Counter = field(default_factory=Counter)
    # Per-cycle quality aggregation
    qual_sums: List[int] = field(default_factory=list)
    qual_counts: List[int] = field(default_factory=list)

    def merge(self, other: "Stats") -> None:
        self.reads += other.reads
        self.bases += other.bases
        self.gc    += other.gc
        self.ns    += other.ns
        self.total_q += other.total_q
        self.length_hist.update(other.length_hist)
        if len(self.qual_sums) < len(other.qual_sums):
            self.qual_sums.extend([0] * (len(other.qual_sums) - len(self.qual_sums)))
            self.qual_counts.extend([0] * (len(other.qual_counts) - len(self.qual_counts)))
        # Merge per-cycle arrays
        for i in range(len(other.qual_sums)):
            self.qual_sums[i]  += other.qual_sums[i]
            self.qual_counts[i] += other.qual_counts[i]

def process_batch(batch: List[Tuple[bytes, bytes, bytes]]) -> Stats:
    """
    CPU-bound processing per batch:
      - counts reads/bases
      - GC and N counts
      - length histogram
      - per-base quality (Sanger Phred+33)
    """
    st = Stats()
    # Local vars for speed
    qual_sums = st.qual_sums
    qual_counts = st.qual_counts
    for _h, seq_b, qual_b in batch:
        L = len(seq_b)
        st.reads += 1
        st.bases += L
        # GC + Ns (case-insensitive)
        # Counting on bytes is pretty fast
        st.gc += seq_b.count(b"G") + seq_b.count(b"C") + seq_b.count(b"g") + seq_b.count(b"c")
        st.ns += seq_b.count(b"N") + seq_b.count(b"n")
        st.length_hist[L] += 1

        # Ensure per-cycle arrays are long enough
        if len(qual_sums) < L:
            qual_sums.extend([0] * (L - len(qual_sums)))
            qual_counts.extend([0] * (L - len(qual_counts)))

        # Quality bytes: each is ASCII; Phred = byte - 33
        mv = memoryview(qual_b)
        # Safety: FASTQ requires len(qual)==len(seq); if not, skip per-position update
        if len(mv) == L:
            # Tight loop over ints
            # Accumulate both per-cycle and total quality
            tq = 0
            for i in range(L):
                qv = mv[i] - 33
                qual_sums[i]  += qv
                qual_counts[i] += 1
                tq += qv
            st.total_q += tq
        else:
            # Fallback: still try to accumulate total_q if lengths mismatch
            st.total_q += sum(x - 33 for x in mv)

    return st

# --------- Main ---------
def main():
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: fastq_mt_parser.py <reads.fastq[.gz]|->\n")
        sys.exit(2)
    path = sys.argv[1]

    # GIL status hint (Python 3.13+ provides sys._is_gil_enabled)
    try:
        import sys as _sys
        if hasattr(_sys, "_is_gil_enabled"):
            if _sys._is_gil_enabled():
                sys.stderr.write(
                    "NOTE: GIL appears ENABLED. For true parallel threads on CPython 3.13+/3.14, run with a free-threaded build and disable the GIL:\n"
                    "      python -X gil=0 fastq_mt_parser.py <file>\n"
                )
    except Exception:
        pass

    total = Stats()
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        # executor.map lazily pulls batches; bounded memory
        for part in ex.map(process_batch, fastq_batches(path, BATCH_RECORDS)):
            total.merge(part)

    # --- Output summary ---
    if total.reads == 0:
        print("No reads parsed.")
        return

    avg_len = total.bases / total.reads if total.reads else 0.0
    gc_pct  = (100.0 * total.gc / total.bases) if total.bases else 0.0
    mean_q  = (total.total_q / total.bases) if total.bases else 0.0
    # Convert mean_q (Phred) to approx error rate if you want: p = 10^(-Q/10)

    print(f"# File: {path}")
    print(f"reads\t{total.reads}")
    print(f"bases\t{total.bases}")
    print(f"avg_read_len\t{avg_len:.2f}")
    print(f"GC%\t{gc_pct:.3f}")
    print(f"mean_phred\t{mean_q:.2f}")

    # Quick per-cycle peek (first N cycles)
    upto = min(PRINT_PER_CYCLE, len(total.qual_sums))
    if upto:
        print("\n# cycle\tmean_Q")
        for i in range(upto):
            cnt = total.qual_counts[i]
            mq = (total.qual_sums[i] / cnt) if cnt else float("nan")
            print(f"{i+1}\t{mq:.2f}")

    # Optional: dump length histogram compactly (top few)
    # Commented by default to keep output tidy; uncomment if desired.
    # from itertools import islice
    # print("\n# length\tcount (top 20)")
    # for L, c in islice(total.length_hist.most_common(), 20):
    #     print(f"{L}\t{c}")

if __name__ == "__main__":
    main()
