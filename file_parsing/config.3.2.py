#!/usr/bin/env python3
"""
Extract upstream-60 segments from linker-positive reads.

Assumptions:
- Input file is already enriched for linker-containing reads (e.g. via ugrep
  using 15..41bp linker substrings with <=2 mismatches).
- We no longer do global fuzzy search over the whole dataset; only refine
  hit position & orientation and enforce upstream >= 60 nt.

What it does per read:
  1) Find a linker-like region using a small set of tiles:
       full 41-mer, 31-mer, 21-mer, 15-mer
     for both linker and its reverse complement, with <= MAX_MISMATCH edits.
  2) Determine orientation (+ for linker, - for RC).
  3) Compute upstream length toward CDR3:
       - if orient '+':  upstream_len = start
       - if orient '-':  upstream_len = seqlen - end
  4) Keep read only if upstream_len >= MIN_UPSTREAM.
  5) Extract exactly 60 nt upstream in VH orientation:
       - '+' : last 60 bases before match
       - '-' : first 60 bases after match, reverse-complemented
  6) Output:
       OUTPREFIX.filtered_reads.fasta    (full reads)
       OUTPREFIX.upstream_60.fasta       (upstream 60-mers, VH orientation)

Dependencies:
  pip install edlib

Edit the CONFIG section below before running.
"""

# =============================
# ========= CONFIG ============
# =============================

INPUT_FILE   = "ugrep_linker/linker_hits.fasta"  # linker-enriched FASTA from ugrep
OUTPREFIX    = "anchors_from_ugrep"

LINKER       = "GGTGGTGGTGGTAGCGGTGGTGGTGGTAGCGGTGGTGGTGGTAGC"  # your 41bp linker
MIN_LINKER   = 15         # minimum tile length (we use 41,31,21,15)
MAX_MISMATCH = 2          # max edit distance for tile <> read (substitutions/indels)
MIN_UPSTREAM = 60         # require at least this much upstream toward CDR3

# =============================
# ========= SCRIPT ============
# =============================

import sys
import edlib

COMP = str.maketrans("ACGTUNacgtun", "TGCANNtgcann")

def rc(s: str) -> str:
    return s.translate(COMP)[::-1]

def is_fastq(path: str) -> bool:
    with open(path, "rb") as fh:
        b = fh.read(1)
    return b == b'@'

def read_fasta(path):
    with open(path, "rt", encoding="utf-8", newline=None) as fh:
        header = None
        seq_chunks = []
        for line in fh:
            if not line:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            if line[0] == ">":
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            yield header, "".join(seq_chunks)

def read_fastq(path):
    with open(path, "rt", encoding="utf-8", newline=None) as fh:
        while True:
            h = fh.readline()
            if not h:
                return
            s = fh.readline()
            plus = fh.readline()
            q = fh.readline()
            if not q:
                return
            yield h[1:].strip(), s.strip()

def build_tiles(linker: str):
    """
    Build a small set of tiles: all contiguous substrings of lengths
    [41, 31, 21, MIN_LINKER] from linker and its RC.

    Returns a list of (tile_seq, orient) with orient in {'+','-'},
    sorted roughly by length descending.
    """
    linker = linker.upper().replace("U", "T")
    L = len(linker)
    rc_linker = rc(linker)

    lengths = []
    for k in (L, 31, 21, MIN_LINKER):
        if k <= L:
            lengths.append(k)
    lengths = sorted(set(lengths), reverse=True)

    tiles = []
    for seq, orient in ((linker, '+'), (rc_linker, '-')):
        for k in lengths:
            for i in range(0, L - k + 1):
                tiles.append((seq[i:i+k], orient))
    # sort tiles by length desc so we prefer longer matches
    tiles.sort(key=lambda x: len(x[0]), reverse=True)
    return tiles

def find_linker_region(seq: str, tiles, max_mismatch: int):
    """
    Given a sequence and a list of (tile_seq, orient), try to find
    a linker-like region using edlib with <= max_mismatch edits.

    Returns: (orient, start, end) or (None, None, None)
    where start/end are 0-based coordinates in seq (end is exclusive).
    """
    for tile, orient in tiles:
        # HW mode: find best match of tile to any substring of seq
        r = edlib.align(tile, seq, mode="HW", k=max_mismatch, task="locations")
        locs = r.get("locations", [])
        if not locs:
            continue
        # choose the shortest span, then earliest start
        start, end = min(locs, key=lambda x: (x[1]-x[0], x[0]))
        return orient, start, end
    return None, None, None

def main():
    linker = LINKER.upper().replace("U", "T")
    if len(linker) < MIN_LINKER:
        print("ERROR: LINKER length < MIN_LINKER", file=sys.stderr)
        sys.exit(1)

    tiles = build_tiles(linker)

    # Decide input format
    fmt_fastq = is_fastq(INPUT_FILE)
    reader = read_fastq(INPUT_FILE) if fmt_fastq else read_fasta(INPUT_FILE)

    out_full = open(f"{OUTPREFIX}.filtered_reads.fasta", "w", buffering=1<<20)
    out_up60 = open(f"{OUTPREFIX}.upstream_60.fasta", "w", buffering=1<<20)

    total = 0
    matched = 0
    kept = 0

    for h, raw in reader:
        total += 1
        s = raw.upper().replace("U", "T")
        n = len(s)
        if n < MIN_UPSTREAM + MIN_LINKER:
            continue

        orient, start, end = find_linker_region(s, tiles, MAX_MISMATCH)
        if orient is None:
            # should be rare, since ugrep pre-filtered; skip if can't refine
            continue
        matched += 1

        if orient == '+':
            upstream_len = start
            if upstream_len < MIN_UPSTREAM:
                continue
            up60 = s[start-MIN_UPSTREAM:start]
        else:
            upstream_len = n - end
            if upstream_len < MIN_UPSTREAM:
                continue
            up60 = rc(s[end:end+MIN_UPSTREAM])

        kept += 1

        meta = f"{h}|orient={orient}|match={start+1}-{end}|upstream={upstream_len}"
        # wrap full seq at 80 nt
        full_wrapped = "\n".join(s[i:i+80] for i in range(0, n, 80))
        out_full.write(f">{meta}\n{full_wrapped}\n")
        out_up60.write(f">{h}|orient={orient}|up60\n{up60}\n")

    out_full.close()
    out_up60.close()

    print(f"Input records (linker_hits): {total}")
    print(f"Reads where a linker-like region was located: {matched}")
    print(f"Kept with upstream >= {MIN_UPSTREAM}: {kept}")
    print(f"Outputs:")
    print(f"  {OUTPREFIX}.filtered_reads.fasta")
    print(f"  {OUTPREFIX}.upstream_60.fasta")

if __name__ == "__main__":
    main()
