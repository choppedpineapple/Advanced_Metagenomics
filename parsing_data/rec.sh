#!/usr/bin/env bash
set -euo pipefail
# approachB_chain_targeted.sh
# Usage: ./approachB_chain_targeted.sh SAMPLE R1.fastq.gz R2.fastq.gz
# Example: ./approachB_chain_targeted.sh ES-RAGE SRR_R1.fastq.gz SRR_R2.fastq.gz

SAMPLE="$1"
R1="$2"
R2="$3"
THREADS=8
MINLEN=50   # slightly more permissive here to capture more reads
Q=20
OUTDIR="${SAMPLE}_approachB"
mkdir -p "$OUTDIR"
cd "$OUTDIR"

### 1) QC trim
echo "1) fastp trimming..."
fastp -i ../"$R1" -I ../"$R2" -o clean_R1.fastq.gz -O clean_R2.fastq.gz \
      --qualified_quality_phred $Q --length_required $MINLEN --n_base_limit 5 \
      --cut_mean_quality $Q --thread $THREADS --html fastp_report.html --json fastp_report.json

### 2) Merge reads with BBMerge (or pandaseq)
echo "2) merging with BBMerge..."
bbmerge.sh in=clean_R1.fastq.gz in2=clean_R2.fastq.gz out=merged.fastq outu1=unmerged_R1.fastq outu2=unmerged_R2.fastq strict=t ihist=bb_ihist.txt threads=$THREADS
# pandaseq alternative:
# pandaseq -f clean_R1.fastq.gz -r clean_R2.fastq.gz -o merged.fastq -F

### 3) Convert merged to FASTA and dereplicate
seqtk seq -a merged.fastq > merged.fasta
vsearch --derep_fulllength merged.fasta --output merged.derep.fasta --sizeout --threads $THREADS

### 4) Run IgBlast on merged.derep.fasta (tabular output) and parse chain calls
echo "4) running igblastn on merged sequences..."
igblastn -query merged.derep.fasta \
         -germline_db_V ../imgt/sheep_V.fasta \
         -germline_db_D ../imgt/sheep_D.fasta \
         -germline_db_J ../imgt/sheep_J.fasta \
         -organism sheep \
         -outfmt "6 qseqid qseq sseqid pident qlen length qstart qend sstart send evalue bitscore sseq" \
         -out igblast_merged.tsv

### 5) Parse IgBlast TSV to separate IGH and IGL/IGK reads (python helper)
cat > parse_igblast_tsv.py <<'PY'
#!/usr/bin/env python3
# parse_igblast_tsv.py
# Reads igblast_merged.tsv (tabular) and writes IGH.fasta and IGL.fasta with headers as qseqid|orig_header
import sys,os
infile='igblast_merged.tsv'
f_igh=open('IGH_reads.fasta','w')
f_igl=open('IGL_reads.fasta','w')
with open(infile) as fh:
    for line in fh:
        parts=line.rstrip('\n').split('\t')
        if len(parts) < 3:
            continue
        qseqid, qseq, sseqid = parts[0], parts[1], parts[2]
        # Try to detect chain tag in sseqid or sseqid string (depends on your germline headers)
        s = sseqid.upper()
        if ('IGH' in s) or ('IGHE' in s) or ('IGHV' in s) or ('IG_H' in s) or ('IGHV' in s):
            f_igh.write(f">{qseqid}\n{qseq}\n")
        elif ('IGL' in s) or ('IGK' in s) or ('IGL' in s):
            f_igl.write(f">{qseqid}\n{qseq}\n")
# Close
f_igh.close(); f_igl.close()
print('Wrote IGH_reads.fasta and IGL_reads.fasta')
PY
chmod +x parse_igblast_tsv.py
./parse_igblast_tsv.py

### 6) If parse produced nothing, try a looser parse using a fallback awk (chain column might be different)
if [ ! -s IGH_reads.fasta ] || [ ! -s IGL_reads.fasta ]; then
  echo "IGH/IGL fasta empty — trying fallback chain detection via grep on igblast output"
  awk -F'\t' '{ if ($3 ~ /IGH/ || $3 ~ /IGHV/ ) print ">"$1"\n"$2 > "IGH_reads.fasta"; else if ($3 ~ /IGL/ || $3 ~ /IGK/) print ">"$1"\n"$2 > "IGL_reads.fasta"; }' igblast_merged.tsv || true
fi

### 7) Assemble IGH and IGL pools separately (SPAdes or rnaSPAdes)
echo "7) assembling IGH and IGL pools (if present)..."
if [ -s IGH_reads.fasta ]; then
  # create a synthetic fastq for assembler (SPAdes can accept fasta with -s)
  spades.py --only-assembler -s IGH_reads.fasta -o spades_IGH -t $THREADS -m 32
  cp spades_IGH/contigs.fasta IGH_contigs.fasta || true
fi
if [ -s IGL_reads.fasta ]; then
  spades.py --only-assembler -s IGL_reads.fasta -o spades_IGL -t $THREADS -m 32
  cp spades_IGL/contigs.fasta IGL_contigs.fasta || true
fi

### 8) Cluster contigs or dereplicated reads to get consensus clones
# We'll cluster derep reads per chain at 97% and create cluster representatives; use vsearch to cluster and output centroids.
echo "8) clustering to get candidate clones..."
if [ -s merged.derep.fasta ]; then
  # make chain-specific fasta files by mapping merged.derep.fasta ids to IGH/IGL read headers
  # Create header list for IGH ids
  grep '^>' IGH_reads.fasta | sed 's/^>//' > IGH_ids.txt || true
  grep '^>' IGL_reads.fasta | sed 's/^>//' > IGL_ids.txt || true
  if [ -s IGH_ids.txt ]; then
    seqtk subseq merged.derep.fasta IGH_ids.txt > merged_IGH.subset.fasta || true
    vsearch --cluster_fast merged_IGH.subset.fasta --id 0.97 --centroids IGH_centroids.fasta --uc IGH_clusters.uc --threads $THREADS
  fi
  if [ -s IGL_ids.txt ]; then
    seqtk subseq merged.derep.fasta IGL_ids.txt > merged_IGL.subset.fasta || true
    vsearch --cluster_fast merged_IGL.subset.fasta --id 0.97 --centroids IGL_centroids.fasta --uc IGL_clusters.uc --threads $THREADS
  fi
fi

### 9) Build consensus per centroid cluster via MAFFT + majority rule (python helper)
cat > build_consensus.py <<'PY'
#!/usr/bin/env python3
# build_consensus.py centroids.fasta clusters.uc out.cons.fasta
import sys
from Bio import SeqIO
from Bio.Align.Applications import MafftCommandline
from Bio import AlignIO
import subprocess,os
def consensus_from_seqs(in_fasta, out_fasta, tmp_prefix):
    # Align with mafft
    mafft_cline = MafftCommandline(input=in_fasta)
    stdout,stderr = mafft_cline()
    aln_file = tmp_prefix + ".aln"
    with open(aln_file,"w") as fh:
        fh.write(stdout)
    aln = AlignIO.read(aln_file,"fasta")
    # consensus: majority base (N for tie)
    cons=[]
    L=len(aln[0].seq)
    for i in range(L):
        col = [rec.seq[i] for rec in aln]
        from collections import Counter
        c=Counter(col)
        # ignore gaps when possible
        if '-' in c:
            c.pop('-',None)
        if c:
            base, cnt = c.most_common(1)[0]
            cons.append(base)
        else:
            cons.append('N')
    with open(out_fasta,"w") as fh:
        fh.write(">consensus\\n")
        fh.write("".join(cons)+"\\n")
if __name__=='__main__':
    # For simplicity: if centroids.fasta exists, generate simple consensus as centroid (centroid ~= consensus)
    # This script can be extended to group members and align them. For now, copy centroids to consensus
    if len(sys.argv)<3:
        print("Usage: build_consensus.py centroids.fasta out.cons.fasta")
        sys.exit(1)
    centroids=sys.argv[1]
    out=sys.argv[2]
    # simple pass-through: just copy centroids
    import shutil
    shutil.copyfile(centroids,out)
    print("Wrote consensus (copied centroids) to", out)
PY
chmod +x build_consensus.py
# Run build_consensus to copy centroids to consensus as simple approach
if [ -s IGH_centroids.fasta ]; then
  ./build_consensus.py IGH_centroids.fasta IGH_consensus.fasta
fi
if [ -s IGL_centroids.fasta ]; then
  ./build_consensus.py IGL_centroids.fasta IGL_consensus.fasta
fi

### 10) Translate and annotate consensus sequences with IgBlast (to get CDR positions)
echo "9) annotating consensus sequences with igblast..."
if [ -s IGH_consensus.fasta ]; then
  igblastn -query IGH_consensus.fasta \
           -germline_db_V ../imgt/sheep_V.fasta \
           -germline_db_D ../imgt/sheep_D.fasta \
           -germline_db_J ../imgt/sheep_J.fasta \
           -organism sheep -outfmt "6 qseqid qseq sseqid pident qlen length qstart qend sstart send evalue bitscore sseq" \
           -out IGH_consensus_igblast.tsv || true
fi
if [ -s IGL_consensus.fasta ]; then
  igblastn -query IGL_consensus.fasta \
           -germline_db_V ../imgt/sheep_V.fasta \
           -germline_db_D ../imgt/sheep_D.fasta \
           -germline_db_J ../imgt/sheep_J.fasta \
           -organism sheep -outfmt "6 qseqid qseq sseqid pident qlen length qstart qend sstart send evalue bitscore sseq" \
           -out IGL_consensus_igblast.tsv || true
fi

### 11) Create candidate scFv by pairing top N VH x top M VL with canonical linker
echo "10) pairing top clones to make candidate scFv sequences..."
TOPN=10
# simple selection: pick first TOPN from consensus file sequences
if [ -s IGH_consensus.fasta ] && [ -s IGL_consensus.fasta ]; then
  mkdir -p candidates
  # extract top N sequences (centroids order may reflect abundance > if not, user adjust)
  seqtk seq -l0 IGH_consensus.fasta | awk '/^>/{i++} {if(i<=2*'$TOPN') print $0}' > topIGH.tmp.fasta || true
  seqtk seq -l0 IGL_consensus.fasta | awk '/^>/{i++} {if(i<=2*'$TOPN') print $0}' > topIGL.tmp.fasta || true
  # canonical linker (DNA) - adjust codons if necessary
  LINKER="GGTGGTGGTGGTAGCGGTGGTGGTGGTAGCGGTGGTGGTGGTAGC"
  # Now brute-force pair
  python3 - <<'PY'
from Bio import SeqIO
linker="GGTGGTGGTGGTAGCGGTGGTGGTGGTAGCGGTGGTGGTGGTAGC"
igh=list(SeqIO.parse("topIGH.tmp.fasta","fasta"))
igl=list(SeqIO.parse("topIGL.tmp.fasta","fasta"))
outf=open("candidates/candidate_scFv.fasta","w")
count=0
for i, vh in enumerate(igh):
    for j, vl in enumerate(igl):
        count+=1
        header=f">candidate_{count}|vh={vh.id}|vl={vl.id}"
        seq=str(vh.seq)+linker+str(vl.seq)
        outf.write(header+"\n")
        # wrap at 80
        for k in range(0,len(seq),80):
            outf.write(seq[k:k+80]+"\n")
outf.close()
print("Wrote",count,"candidate scFv sequences to candidates/candidate_scFv.fasta")
PY
fi

### 12) Validate candidates: map reads and get support
echo "11) mapping reads to candidate scFv..."
if [ -f candidates/candidate_scFv.fasta ]; then
  bwa index candidates/candidate_scFv.fasta
  bwa mem -t $THREADS candidates/candidate_scFv.fasta clean_R1.fastq.gz clean_R2.fastq.gz | samtools view -b - | samtools sort -o mapped_candidates.bam
  samtools index mapped_candidates.bam
  samtools idxstats mapped_candidates.bam > candidates/mapping_stats.tsv
  echo "Candidate mapping stats: candidates/mapping_stats.tsv"
  # get per-candidate depth
  samtools depth mapped_candidates.bam | awk '{cov[$1]+=$3;cnt[$1]++} END{for(k in cov) print k, cov[k], cnt[k], cov[k]/cnt[k]}' > candidates/coverage_summary.tsv
fi

echo "Approach B complete. Inspect candidates/candidate_scFv.fasta and mapping stats in candidates/."
