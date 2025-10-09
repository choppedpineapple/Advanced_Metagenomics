#!/usr/bin/env python3
import argparse
import re
import edlib
from Bio import SeqIO
import pandas as pd

def find_linker_approx(seq, linker, max_mismatch):
    """Find linker with up to max_mismatch edits (mismatches/indels)"""
    result = edlib.align(linker, seq, mode="HW", task="locations")
    if result["editDistance"] <= max_mismatch and result["locations"]:
        start, end = result["locations"][0]
        return start, end
    return None, None

def parse_igblast_tsv(tsv_file):
    """Load IgBLAST TSV into dict: {seq_id: row}"""
    df = pd.read_csv(tsv_file, sep='\t', low_memory=False)
    return {row['sequence_id']: row for _, row in df.iterrows()}

def is_productive(v_call, j_call, junction):
    """Basic check for productive rearrangement"""
    if not isinstance(junction, str) or len(junction) % 3 != 0:
        return False
    if '*' in junction or 'X' in junction:
        return False
    return v_call and j_call

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', required=True)
    parser.add_argument('--igblast_tsv', required=True)
    parser.add_argument('--linker', required=True)
    parser.add_argument('--max_mismatch', type=int, default=2)
    parser.add_argument('--output_fasta', required=True)
    parser.add_argument('--output_tsv', required=True)
    args = parser.parse_args()

    igblast_dict = parse_igblast_tsv(args.igblast_tsv)
    linker = args.linker
    max_mm = args.max_mismatch

    scfv_records = []
    metadata = []

    for record in SeqIO.parse(args.fasta, "fastq"):
        seq_id = record.id
        seq = str(record.seq)

        # Skip if not in IgBLAST results
        if seq_id not in igblast_dict:
            continue

        start, end = find_linker_approx(seq, linker, max_mm)
        if start is None:
            continue  # no linker found

        left_part = seq[:start]
        right_part = seq[end+1:]

        # Get IgBLAST annotations for full read
        row = igblast_dict[seq_id]
        v_gene = row.get('v_call', '')
        j_gene = row.get('j_call', '')
        junction = row.get('junction', '')

        # Determine orientation
        left_is_vh = 'IGHV' in str(v_gene)
        right_is_vl = any(x in str(v_gene) for x in ['IGKV', 'IGLV'])  # caution: v_gene is for full read

        # Better: run mini-IgBLAST on parts? For now, infer from full annotation
        # We'll assume: if full read has IGHV → left = VH; if IGKV/IGLV → right = VL
        # But safer: require that left aligns to IGH and right to IGK/L

        # For simplicity, we accept if:
        # - Full read is annotated as IGH (so left = VH)
        # - AND right part is long enough to be VL
        if 'IGHV' in str(v_gene) and len(right_part) > 80:
            orientation = "VH-linker-VL"
            vh_seq = left_part
            vl_seq = right_part
        elif any(x in str(v_gene) for x in ['IGKV', 'IGLV']) and len(left_part) > 80:
            orientation = "VL-linker-VH"
            vl_seq = left_part
            vh_seq = right_part
        else:
            continue

        # Basic productivity check
        if not is_productive(v_gene, j_gene, junction):
            continue

        full_scfv = vh_seq + linker + vl_seq
        new_id = f"{seq_id}_scFv"

        scfv_records.append(SeqIO.SeqRecord(
            SeqIO.Seq(full_scfv),
            id=new_id,
            description=""
        ))

        metadata.append({
            'scfv_id': new_id,
            'original_read': seq_id,
            'orientation': orientation,
            'vh_length': len(vh_seq),
            'vl_length': len(vl_seq),
            'v_call': v_gene,
            'j_call': j_gene,
            'junction': junction,
            'linker_start': start,
            'linker_end': end
        })

    # Write outputs
    SeqIO.write(scfv_records, args.output_fasta, "fasta")
    pd.DataFrame(metadata).to_csv(args.output_tsv, sep='\t', index=False)

    print(f"Reconstructed {len(scfv_records)} scFv sequences.")

if __name__ == "__main__":
    main()
