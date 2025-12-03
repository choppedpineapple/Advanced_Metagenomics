#!/usr/bin/env python3
import edlib

input_file = "linker_hits.fasta"
outprefix = "upstream"
linker = "GGTGGTGGTGGTAGCGGTGGTGGTGGTAGCGGTGGTGGTGGTAGC"
min_linker = 15
max_mismatch = 2
anchor_len = 120   # captures full CDR3

def rc(s):
    return s.translate(str.maketrans("ACGTUNacgtun","TGCANNtgcann"))[::-1]

def fasta_reader(p):
    h, seq = None, []
    with open(p) as f:
        for l in f:
            l=l.strip()
            if l.startswith(">"):
                if h: yield h, "".join(seq)
                h, seq = l[1:], []
            else:
                seq.append(l)
        if h: yield h, "".join(seq)

def build_tiles(linker):
    L = len(linker)
    r = rc(linker)
    ks = [k for k in (L,31,21,min_linker) if k<=L]
    tiles=[]
    for s, o in ((linker,'+'),(r,'-')):
        for k in ks:
            for i in range(0, L-k+1):
                tiles.append((s[i:i+k],o))
    tiles.sort(key=lambda x: len(x[0]), reverse=True)
    return tiles

def find_linker(seq, tiles):
    for tile,o in tiles:
        r = edlib.align(tile, seq, mode="HW", k=max_mismatch, task="locations")
        loc = r.get("locations",[])
        if loc:
            s,e = loc[0]
            return o,s,e
    return None,None,None

tiles = build_tiles(linker)
f_full = open(f"{outprefix}.reads.fasta","w")
f_up = open(f"{outprefix}.anchor.fasta","w")

for h,s in fasta_reader(input_file):
    s=s.upper()
    o,start,end = find_linker(s,tiles)
    if o is None: continue
    if o == '+':
        if start < anchor_len: continue
        a = s[start-anchor_len:start]
    else:
        if len(s)-end < anchor_len: continue
        a = rc(s[end:end+anchor_len])
    f_full.write(f">{h}\n{s}\n")
    f_up.write(f">{h}\n{a}\n")

f_full.close()
f_up.close()
