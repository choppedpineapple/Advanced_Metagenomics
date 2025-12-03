###############################################
# STEP 1 — Convert merged FASTA → one-line TSV
###############################################

awk '
  /^>/ {
    if(seq!=""){print header"\t"seq}
    header=$0
    seq=""
    next
  }
  { seq=seq$0 }
  END{
    if(seq!="") print header"\t"seq
  }
' merged.fasta > merged.oneline.tsv



#########################################################
# STEP 2 — ugrep fuzzy search (find linker-containing reads)
#########################################################

# linker_15mers.txt must already exist (generated earlier)

ugrep -Z2 -f linker_15mers.txt merged.oneline.tsv > linker_hits.tsv



###########################################
# STEP 3 — Convert TSV back into FASTA
###########################################

awk -F'\t' '{print $1; print $2}' linker_hits.tsv > linker_hits.fasta



###########################################################
# STEP 4 — Extract VH upstream anchors (120 bp upstream)
###########################################################

python3 extract_upstream_anchor.py
# Produces:
#   upstream.reads.fasta
#   upstream.anchor.fasta



###########################################
# STEP 5 — Cluster VH upstream anchors
###########################################

vsearch --cluster_fast upstream.anchor.fasta \
        --id 0.97 \
        --centroids vh_centroids.fasta \
        --uc vh_clusters.uc



#############################################################
# STEP 6 — Extract VL downstream anchors (120 bp downstream)
#############################################################

python3 extract_downstream_anchor.py
# Produces:
#   downstream.reads.fasta
#   downstream.anchor.fasta



#############################################################
# STEP 7 — Build VH cluster mapping (clusterID → readID)
#############################################################

awk '$1=="S" || $1=="H" || $1=="C" { print $9, $10 }' vh_clusters.uc > cluster_map.txt



###############################################
# STEP 8 — Make cluster directories
###############################################

mkdir -p vh_clusters



###########################################################
# STEP 9 — Split read names into per-cluster ID files
###########################################################

while read cid rid; do
  echo ">$rid" >> vh_clusters/$cid.ids
done < cluster_map.txt



##############################################################
# STEP 10 — Extract upstream reads for each VH cluster
##############################################################

for cid in vh_clusters/*.ids; do
  base=${cid%.ids}
  seqtk subseq upstream.reads.fasta $cid > ${base}.up.fasta
done



##############################################################
# STEP 11 — Prepare downstream anchor table
##############################################################

awk 'NR%2==1{h=$0} NR%2==0{print h"\t"$0}' downstream.anchor.fasta > downstream.tab



###################################################################
# STEP 12 — For each VH cluster: extract downstream anchor matches
###################################################################

for cid in vh_clusters/*.ids; do
  base=${cid%.ids}

  # Upstream IDs → extract only the IDs without ">"
  awk '{print $2}' $cid > ${base}.up_ids.txt

  # Downstream anchors belonging to those upstream IDs
  grep -F -f ${base}.up_ids.txt downstream.tab | cut -f2 > ${base}.ds_keys.txt

  # All downstream reads that match those anchors
  grep -F -f ${base}.ds_keys.txt downstream.tab | cut -f1 > ${base}.ds_ids.txt
done



#########################################################
# STEP 13 — Combine upstream + downstream read IDs
#########################################################

for cid in vh_clusters/*.ids; do
  base=${cid%.ids}
  cat ${base}.up_ids.txt ${base}.ds_ids.txt | sort -u > ${base}.all_ids.txt
done



#########################################################
# STEP 14 — Extract full reads for each cluster
#########################################################

for cid in vh_clusters/*.ids; do
  base=${cid%.ids}
  seqtk subseq linker_hits.fasta ${base}.all_ids.txt > ${base}.reads.fasta
done



#########################################################
# STEP 15 — Assemble each cluster with SPAdes
#########################################################

for cid in vh_clusters/*.ids; do
  base=${cid%.ids}
  spades.py -s ${base}.reads.fasta \
            -o ${base}_spades \
            --only-assembler \
            -k 21,33,55,77,99,127
done
