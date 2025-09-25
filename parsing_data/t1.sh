#!/bin/bash
samples=("Transferrin-light_S1" "Transferrin-heavy_S2" "ES-RAGE-light_S3" "ES-RAGE-heavy_S4")
chains=("light" "heavy" "light" "heavy")

for i in "${!samples[@]}"; do
  sample=${samples[$i]}
  chain=${chains[$i]}
  echo "Processing $sample ($chain chain)..."

  # Assemble
  presto AssemblePairs \
    --1 ${sample}_L001_R1_001.fastq.gz \
    --2 ${sample}_L001_R2_001.fastq.gz \
    --coord sanger --rc tail --minlen 300 --nproc 8 \
    --outname $sample

  # Convert to FASTA
  presto SeqConv --fastq ${sample}-assembly-pass.fastq --fasta ${sample}.fasta

  # Collapse
  presto CollapseSeq --fasta ${sample}.fasta --id 1.0 --outname ${sample}-collapsed

  # IgBLAST
  if [[ "$chain" == "heavy" ]]; then
    ig_seqtype="IGH"
    VDB="imgt_sheep_ighv.fasta"
    DDB="imgt_sheep_ighd.fasta"
    JDB="imgt_sheep_ighj.fasta"
  else
    ig_seqtype="IGK"  # or IGL
    VDB="imgt_sheep_igkv.fasta"
    DDB=""
    JDB="imgt_sheep_igkj.fasta"
  fi

  igblastn \
    -germline_db_V $VDB \
    -germline_db_D $DDB \
    -germline_db_J $JDB \
    -organism sheep \
    -ig_seqtype $ig_seqtype \
    -domain_system imgt \
    -outfmt 19 \
    -query ${sample}-collapsed.fasta \
    -out ${sample}_igblast.tsv
done
