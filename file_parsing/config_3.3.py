#!/usr/bin/env python3
# Extract upstream-60 segments from linker-positive reads (FASTA) using edlib.

INPUT_FILE   = "linker_hits.fasta"   # from ugrep
OUTPREFIX    = "anchors_from_ugrep"
LINKER       = "GGTGGTGGTGGTAGCGGTGGTGGTGGTAGCGGTGGTGGTGGTAGC"
MIN_LINKER   = 15
MAX_MISMATCH = 2
MIN_UPSTREAM = 60

import edlib

COMP = str.maketrans("ACGTUNacgtun", "TGCANNtgcann")
def rc(s): return s.translate(COMP)[::-1]

def fasta_reader(path):
    h, seq = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line: 
                continue
            if line.startswith(">"):
                if h: 
                    yield h, "".join(seq)
                h, seq = line[1:], []
            else:
                seq.append(line)
        if h:
            yield h, "".join(seq)

def build_tiles(linker):
    linker = linker.upper().replace("U","T")
    L = len(linker)
    r = rc(linker)
    ks = [k for k in (L, 31, 21, MIN_LINKER) if k <= L]
    tiles = []
    for seq, o in ((linker, '+'), (r, '-')):
        for k in ks:
            for i in range(0, L-k+1):
                tiles.append((seq[i:i+k], o))
    tiles.sort(key=lambda x: len(x[0]), reverse=True)
    return tiles

def find_linker_region(seq, tiles):
    for tile, o in tiles:
        r = edlib.align(tile, seq, mode="HW", k=MAX_MISMATCH, task="locations")
        locs = r.get("locations", [])
        if not locs: 
            continue
        s, e = min(locs, key=lambda x: (x[1]-x[0], x[0]))
        return o, s, e
    return None, None, None

def wrap80(s): 
    return "\n".join(s[i:i+80] for i in range(0, len(s), 80))

def main():
    tiles = build_tiles(LINKER)
    f_full = open(f"{OUTPREFIX}.filtered_reads.fasta", "w")
    f_up   = open(f"{OUTPREFIX}.upstream_60.fasta", "w")
    total = matched = kept = 0

    for h, raw in fasta_reader(INPUT_FILE):
        total += 1
        s = raw.upper().replace("U","T")
        if len(s) < MIN_UPSTREAM + MIN_LINKER:
            continue
        o, start, end = find_linker_region(s, tiles)
        if o is None:
            continue
        matched += 1
        if o == '+':
            up_len = start
            if up_len < MIN_UPSTREAM: 
                continue
            up60 = s[start-MIN_UPSTREAM:start]
        else:
            up_len = len(s) - end
            if up_len < MIN_UPSTREAM: 
                continue
            up60 = rc(s[end:end+MIN_UPSTREAM])
        kept += 1
        meta = f"{h}|orient={o}|match={start+1}-{end}|upstream={up_len}"
        f_full.write(f">{meta}\n{wrap80(s)}\n")
        f_up.write(f">{h}|orient={o}|up60\n{up60}\n")

    f_full.close(); f_up.close()
    print(f"Input: {total}, linker-located: {matched}, kept (up>={MIN_UPSTREAM}): {kept}")
    print(f"Outputs:\n  {OUTPREFIX}.filtered_reads.fasta\n  {OUTPREFIX}.upstream_60.fasta")

if __name__ == "__main__":
    main()
