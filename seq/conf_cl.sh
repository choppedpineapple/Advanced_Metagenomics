vsearch --derep_fulllength merged.fasta \
        --output derep_scfv.fasta \
        --sizeout \
        --minseqlength 650 \
        --threads 32


vsearch --cluster_size derep_scfv.fasta \
        --id 0.98 \
        --sizein \
        --sizeout \
        --centroids clustered_scfv.fasta \
        --threads 32


vsearch --uchime_denovo clustered_scfv.fasta \
        --sizein \
        --nonchimeras final_clean_scfv.fasta \
        --chimeras bad_chimeras.fasta
