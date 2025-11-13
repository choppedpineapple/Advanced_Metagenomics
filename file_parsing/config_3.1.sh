LINKER="ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"

awk '
  /^>/ {
    if (seq != "") {
      print header "\t" seq
    }
    header = $0
    seq = ""
    next
  }
  {
    seq = seq $0
  }
  END {
    if (seq != "") {
      print header "\t" seq
    }
  }
' merged.fasta > merged.oneline.tsv

python - << 'EOF'
linker = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"  # <-- put real linker here
k = 15

comp = str.maketrans("ACGTacgt", "TGCAtgca")
linker_rc = linker.translate(comp)[::-1]

with open("linker_15mers.txt", "w") as out:
    for seq in (linker, linker_rc):
        for i in range(len(seq) - k + 1):
            out.write(seq[i:i+k] + "\n")
EOF

ugrep -Z2 -f linker_15mers.txt \
      --label=merged.oneline.tsv \
      merged.oneline.tsv > linker_hits.tsv

awk -F'\t' '{print $1; print $2}' linker_hits.tsv > linker_hits.fasta




  
