#!/usr/bin/env bash
set -euo pipefail

indir="${1:-.}"
outdir="${2:-$indir}"
threshold=500000
threads="${threads:-32}"
cons_py="${cons_py:-./consensus.py}"

mkdir -p "$outdir"

for f in "$indir"/*.fasta; do
  [ -e "$f" ] || continue

  base="$(basename "$f")"
  stem="${base%.fasta}"

  align_out="$outdir/${stem}_align.fasta"
  cons_out="$outdir/${stem}_cons.fasta"

  nseq="$(grep -c '^>' "$f")"
  hdr="$(awk '/^>/{sub(/^>/,""); print; exit}' "$f")"
  cons_hdr="${hdr}_cons"

  if (( nseq > threshold )); then
    seqkit rmdup -s "$f" | mafft --thread "$threads" - > "$align_out"
  else
    mafft --thread "$threads" "$f" > "$align_out"
  fi

  cons_seq="$(python3 "$cons_py" "$align_out")"

  {
    printf '>%s\n' "$cons_hdr"
    printf '%s\n' "$cons_seq"
  } > "$cons_out"
done
