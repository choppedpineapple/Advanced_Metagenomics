clustalo -i input_sequences.fasta -o aligned_sequences.fasta \
    --threads=8 \
    --iter=1 \
    --max-guidetree-iterations=1 \
    --outfmt=fasta \
    --verbose

mafft --parttree --retree 1 --thread 8 input_sequences.fasta > aligned_sequences.fasta

cons -sequence aligned_sequences.fasta -outseq consensus.fasta \
    -plurality 0.1 \
    -setcase 0.5 \
    -identity 0 \
    -name "Consensus_VH_VL"

