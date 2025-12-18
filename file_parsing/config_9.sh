# Ensure you have a 'results' directory
mkdir -p consensus_outputs

for f in *.fasta; do
    echo "Processing $f..."
    
    # 1. Faster-than-light alignment
    # We use the first sequence as a seed to keep the alignment structural
    mafft --6merpair --thread 8 --addfragments "$f" "$f" > temp_aligned.aln 2>/dev/null
    
    # 2. Run our custom consensus tool to strip the 'lowercase' junk
    # Redirecting the output to a new consensus file
    echo ">Consensus_${f%.*}" > "consensus_outputs/cons_${f%.*}.fasta"
    python3 clean_cons.py temp_aligned.aln >> "consensus_outputs/cons_${f%.*}.fasta"
    
    rm temp_aligned.aln
done

echo "Done. Tell the lab guy he's welcome."
