seqkit fq2fa your_trimmed.fastq -o your_sequences.fasta

uniq_id.py your_basename your_sequences.fasta

# Combine V genes
cat sheep_VH.fasta sheep_VL.fasta > sheep_V_combined.fasta
makeblastdb -parse_seqids -dbtype nucl -in sheep_V_combined.fasta

# Combine J genes  
cat sheep_JH.fasta sheep_JL.fasta > sheep_J_combined.fasta
makeblastdb -parse_seqids -dbtype nucl -in sheep_J_combined.fasta

# D genes (heavy chain only, no light chain D)
makeblastdb -parse_seqids -dbtype nucl -in sheep_DH.fasta

vscan_functions.py \
  --query your_basename.nt_seguid.fa \
  --germline_V /path/to/sheep_gl_V \
  --germline_D /path/to/sheep_gl_D \
  --germline_J /path/to/sheep_gl_J \
  --auxiliary_data /path/to/sheep_gl.aux \
  --organism custom \
  --ig_seqtype Ig \
  --domain_system imgt \
  --num_threads 8 \
  --minlen 150 \
  --escore 0.01

  scFv_splitter.py \
  --igBLAST-hits igBLAST.tsv \
  --domain-type imgt

  # Create aa_reference_linker.fasta with your known linker
echo ">reference_linker" > aa_reference_linker.fasta
echo "GGGGSGGGGSGGGGS" >> aa_reference_linker.fasta  # Replace with your linker

linker_scorer.py \
  --scFv in_frame_igBLAST_paired_delim.tsv \
  --linker aa_reference_linker.fasta

  scfv_flags.py \
  --scFv in_frame_igBLAST_paired_delim_linker_scored.tsv \
  --linker-min-id 90 \
  --mismatches 2 \
  --chain-length 80
