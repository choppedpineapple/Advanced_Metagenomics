#!/usr/bin/env python3
"""
Extract reads that contain a 41 bp linker (or any >=15 bp contiguous piece of it)
in either orientation with up to 2 mismatches, AND have at least 60 nt upstream
toward CDR3 (VH side). Output full reads and the 60-nt upstream segments.

Edit the variables in the CONFIG section below before running.

Requires: biopython
Optional: regex  (for fuzzy matching)
"""

# =============================
# ======== CONFIG =============
# =============================

INPUT_FILE = "merged.fasta"         # FASTA or FASTQ file
LINKER = "GGTGGTGGTGGTAGCGGTGGTGGTGGTAGCGGTGGTGGTGGTAGC"  # your 41bp linker
MIN_LINKER = 15                     # minimum contiguous linker length
MAX_MISMATCH = 2                    # max mismatches allowed in linker
MIN_UPSTREAM = 60                   # minimum upstream length required
OUTPREFIX = "anchors"               # output file prefix

# =============================
# ======== SCRIPT =============
# =============================

import sys
from statistics import median
from Bio import SeqIO
from Bio.Seq import Seq

def revcomp(s: str) -> str:
    return str(Seq(s).reverse_complement())

def make_alt_substrings(linker: str, min_len: int):
    L = len(linker)
    alts = []
    for k in range(min_len, L + 1):
        for i in range(0, L - k + 1):
            alts.append(linker[i:i + k])
    return alts

def find_partial_fuzzy_regex(seqU: str, fwd_pat, rc_pat):
    import regex
    m = fwd_pat.search(seqU)
    n = rc_pat.search(seqU)
    if m and n:
        len_m = m.end() - m.start()
        len_n = n.end() - n.start()
        if len_m > len_n: return ('+', m.start(), m.end())
        if len_n > len_m: return ('-', n.start(), n.end())
        em = sum(m.fuzzy_counts) if hasattr(m, 'fuzzy_counts') else 0
        en = sum(n.fuzzy_counts) if hasattr(n, 'fuzzy_counts') else 0
        if em < en: return ('+', m.start(), m.end())
        if en < em: return ('-', n.start(), n.end())
        return ('+', m.start(), m.end()) if m.start() <= n.start() else ('-', n.start(), n.end())
    elif m:
        return ('+', m.start(), m.end())
    elif n:
        return ('-', n.start(), n.end())
    else:
        return (None, None, None)

def find_partial_exact(seqU: str, fwd_subs, rc_subs):
    for sub in fwd_subs:
        i = seqU.find(sub)
        if i != -1:
            return ('+', i, i + len(sub))
    for sub in rc_subs:
        i = seqU.find(sub)
        if i != -1:
            return ('-', i, i + len(sub))
    return (None, None, None)

def main():
    linker = LINKER.upper().replace("U", "T")
    linker_rc = revcomp(linker)
    minL = max(15, MIN_LINKER)
    max_mis = MAX_MISMATCH
    min_up = MIN_UPSTREAM

    HAVE_REGEX = True
    try:
        import regex as re
    except Exception:
        HAVE_REGEX = False

    if HAVE_REGEX:
        fwd_alts = make_alt_substrings(linker, minL)
        rc_alts = make_alt_substrings(linker_rc, minL)
        fwd_alts.sort(key=len, reverse=True)
        rc_alts.sort(key=len, reverse=True)
        fwd_pat = re.compile(r'(?b)(?:' + '|'.join(map(re.escape, fwd_alts)) + r'){s<=' + str(max_mis) + r'}', re.IGNORECASE)
        rc_pat = re.compile(r'(?b)(?:' + '|'.join(map(re.escape, rc_alts)) + r'){s<=' + str(max_mis) + r'}', re.IGNORECASE)
    else:
        fwd_exact = make_alt_substrings(linker, minL); fwd_exact.sort(key=len, reverse=True)
        rc_exact = make_alt_substrings(linker_rc, minL); rc_exact.sort(key=len, reverse=True)

    fmt = None
    with open(INPUT_FILE, "rb") as fh:
        first = fh.read(1)
        fmt = "fastq" if first == b"@" else "fasta"

    out_full = open(f"{OUTPREFIX}.filtered_reads.fasta", "w")
    out_up60 = open(f"{OUTPREFIX}.upstream_60.fasta", "w")

    total = 0
    matched = 0
    kept = 0
    up_lengths = []

    for rec in SeqIO.parse(INPUT_FILE, fmt):
        total += 1
        seqU = str(rec.seq).upper().replace("U", "T")
        n = len(seqU)

        if HAVE_REGEX:
            orient, s, e = find_partial_fuzzy_regex(seqU, fwd_pat, rc_pat)
        else:
            orient, s, e = find_partial_exact(seqU, fwd_exact, rc_exact)

        if orient is None:
            continue
        matched += 1

        if orient == "+":
            upstream_len = s
            if upstream_len < min_up:
                continue
            upstream60 = seqU[s - min_up:s]
        else:
            upstream_len = n - e
            if upstream_len < min_up:
                continue
            upstream60 = revcomp(seqU[e:e + min_up])

        kept += 1
        up_lengths.append(upstream_len)

        header = f"{rec.id}|orient={orient}|match={s+1}-{e}|upstream={upstream_len}"
        out_full.write(f">{header}\n")
        for i in range(0, n, 80):
            out_full.write(seqU[i:i + 80] + "\n")

        out_up60.write(f">{rec.id}|orient={orient}|up60\n{upstream60}\n")

    out_full.close()
    out_up60.close()

    print(f"Input records: {total}")
    print(f"Reads with linker (≥{minL} nt, mismatches ≤{max_mis}): {matched}")
    print(f"Kept (upstream ≥{min_up}): {kept}")
    if up_lengths:
        print(f"Upstream length: min={min(up_lengths)} median={int(median(up_lengths))} max={max(up_lengths)}")
    print(f"Outputs:\n  {OUTPREFIX}.filtered_reads.fasta\n  {OUTPREFIX}.upstream_60.fasta")

if __name__ == "__main__":
    main()
