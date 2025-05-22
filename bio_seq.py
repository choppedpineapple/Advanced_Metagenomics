#!/usr/bin/env python3

import gzip
import os
import glob
import argparse
from pathlib import Path
import random
from collections import deque # For efficient sliding window

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import GC

# --- 1. Dummy Data Generation ---
def create_dummy_fastq_gz(filename, num_reads, read_length, paired_id=None, make_bad_quality=True):
    """
    Creates a dummy gzipped FASTQ file.
    If paired_id is '1' or '2', it will format the header for paired-end reads.
    """
    print(f"Creating dummy file: {filename}")
    with gzip.open(filename, "wt") as f_out:
        for i in range(num_reads):
            seq_bases = "".join(random.choice("ATGC") for _ in range(read_length))
            
            # Simulate varying quality
            qual_scores = []
            for j in range(read_length):
                if make_bad_quality and i % 5 == 0 and j > read_length * 0.7: # Every 5th read has bad tail
                    qual_scores.append(random.randint(2, 15)) 
                elif make_bad_quality and i % 10 == 0 and j > read_length * 0.5 : # Every 10th read has poor quality earlier
                    qual_scores.append(random.randint(5, 25))
                else:
                    qual_scores.append(random.randint(30, 40))
            
            qual_string = "".join(chr(q + 33) for q in qual_scores) # Phred+33

            read_id_suffix = ""
            if paired_id:
                read_id_suffix = f"/{paired_id}" # Common for older Illumina
            
            # More standard Illumina naming uses space in description
            # e.g. @INSTRUMENT:RUN:FLOWCELL:LANE:TILE:X:Y 1:N:0:INDEX
            # For simplicity, we'll use a simpler ID
            header_id = f"@SIMULATED_READ_{i+1}{read_id_suffix}"
            description = f" 1:N:0:AGATCTCG" # Example description part

            f_out.write(f"{header_id}{description}\n")
            f_out.write(f"{seq_bases}\n")
            f_out.write("+\n")
            f_out.write(f"{qual_string}\n")
    print(f"Finished creating {filename}")

# --- 2. Core Biopython I/O ---
def parse_fastq_gz(filepath):
    """
    Parses a gzipped FASTQ file and yields SeqRecord objects.
    """
    try:
        with gzip.open(filepath, "rt") as handle: # "rt" for reading in text mode
            for record in SeqIO.parse(handle, "fastq"):
                yield record
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        raise

def write_fastq_gz(records, filepath):
    """
    Writes a list/generator of SeqRecord objects to a gzipped FASTQ file.
    Returns the number of records written.
    """
    count = 0
    with gzip.open(filepath, "wt") as handle: # "wt" for writing in text mode
        for record in records:
            if record: # Ensure record is not None
                SeqIO.write(record, handle, "fastq")
                count += 1
    return count

# --- 3. Sequence Manipulation Examples ---
def demonstrate_sequence_manipulation(record):
    """
    Shows various Biopython Seq object manipulations.
    """
    if not record:
        print("  No record provided for manipulation demo.")
        return

    print(f"\n--- Demonstrating Manipulations for Record: {record.id} ---")
    print(f"  Original Sequence ({len(record.seq)}bp): {record.seq}")
    print(f"  Original Quality Scores: {record.letter_annotations['phred_quality']}")

    # GC Content
    gc_content = GC(record.seq)
    print(f"  GC Content: {gc_content:.2f}%")

    # Reverse Complement
    rc_seq = record.seq.reverse_complement()
    print(f"  Reverse Complement: {rc_seq}")
    
    # Note: Quality scores should also be reversed for a true RC record,
    # but not reverse complemented.
    rc_record = SeqRecord(rc_seq, id=record.id + "_rc", 
                          letter_annotations={"phred_quality": record.letter_annotations['phred_quality'][::-1]},
                          description="reverse complemented")
    print(f"  RC Record Qualities (reversed): {rc_record.letter_annotations['phred_quality']}")


    # Slicing
    if len(record.seq) > 10:
        sliced_seq = record.seq[5:15] # Bases from index 5 up to (not including) 15
        print(f"  Sliced Sequence (5-14): {sliced_seq}")
        # Slicing a SeqRecord (creates a new SeqRecord)
        sliced_record = record[5:15]
        print(f"  Sliced Record ID: {sliced_record.id}, Seq: {sliced_record.seq}")
        print(f"  Sliced Record Qual: {sliced_record.letter_annotations['phred_quality']}")

    # Transcription (DNA -> RNA)
    if "N" not in record.seq.upper(): # Transcription is problematic with Ns
        rna_seq = record.seq.transcribe()
        print(f"  Transcribed to RNA: {rna_seq}")

        # Translation (RNA -> Protein)
        # This is usually done on CDS, not raw reads, but for demo:
        if len(rna_seq) >= 3:
            protein_seq = rna_seq.translate(to_stop=True) # Translate until first stop codon
            print(f"  Translated to Protein (first ORF, to_stop=True): {protein_seq}")
            
            protein_seq_full = rna_seq.translate() # Translate full, including stops as '*'
            print(f"  Translated to Protein (full, stops as '*'): {protein_seq_full}")
    else:
        print("  Skipping transcription/translation due to 'N's or short length.")
    
    print("--- End of Manipulations Demo ---")

# --- 4. Trimming Logic ---
def trim_by_quality_threshold(record, quality_threshold=20, min_length=30):
    """
    Trims a SeqRecord from the 3' end based on per-base quality.
    If a base has quality < quality_threshold, the read is trimmed AT that base.
    Returns the trimmed SeqRecord, or None if it becomes shorter than min_length.
    """
    if not record: return None
    
    qualities = record.letter_annotations["phred_quality"]
    trim_index = len(qualities) # Start assuming no trim needed

    # Iterate from 3' end to find the first base to keep (or where to cut)
    for i in range(len(qualities) -1, -1, -1):
        if qualities[i] < quality_threshold:
            trim_index = i # This base is bad, trim *before* it (i.e. keep up to i-1)
        else:
            # This base is good. If all subsequent bases (towards 3' end) were bad,
            # they've already set trim_index. Now we've found a good one.
            # We want to trim *after* the last good base.
            # So, if qualities[i] is good, the cut should be at i+1
            # Wait, the logic is: find first base from 5' end whose quality is < threshold
            # No, it's trim from 3' end. So, find last good base from 5' end.
            break # Found the first good base from the 3' end, stop.

    # Let's re-think: find the longest stretch from 5' where all bases are good.
    # Or, more simply: find the rightmost base to KEEP.
    # Iterate from 5' to 3'. If quality[i] < threshold, then trim_pos = i.
    # No, that's 5' trimming.
    # For 3' trimming: find the position 'k' such that all bases from 0 to k-1 are good,
    # OR, scan from 3' end.
    
    cut_at_idx = len(qualities) # Default: keep everything
    for i in range(len(qualities)):
        if qualities[i] < quality_threshold:
            cut_at_idx = i
            break # Trim at the first low-quality base from 5' end

    # The above is 5' trimming. For 3' trimming:
    # Find the rightmost position `j` such that `quality[j] >= threshold`.
    # Then, the new length is `j+1`.
    
    keep_to_idx = -1 # Index of the last base to keep
    for i in range(len(qualities) -1, -1, -1): # Iterate from 3' end
        if qualities[i] >= quality_threshold:
            keep_to_idx = i
            break 
            
    if keep_to_idx == -1: # All bases are bad
        # print(f"DEBUG: Record {record.id} all bases below threshold {quality_threshold}")
        return None

    trimmed_record = record[:keep_to_idx+1]

    if len(trimmed_record) < min_length:
        # print(f"DEBUG: Record {record.id} too short ({len(trimmed_record)}bp) after quality trim (min_length {min_length}).")
        return None
    return trimmed_record


def trim_by_sliding_window_quality(record, window_size=10, min_avg_quality=20, min_length=30):
    """
    Trims a SeqRecord from the 3' end if average quality in a sliding window drops.
    The read is trimmed at the START of the first window that fails the quality test.
    Returns the trimmed SeqRecord, or None if it becomes shorter than min_length.
    """
    if not record: return None

    qualities = record.letter_annotations["phred_quality"]
    if len(qualities) < window_size: # Too short to apply window trimming meaningfully
        # Fallback: check average quality of the whole short read
        if sum(qualities) / len(qualities) < min_avg_quality:
             return None # Discard if even the short read has bad average
        if len(qualities) < min_length:
            return None
        return record # Otherwise keep it as is

    q_sum = sum(qualities[0:window_size])
    
    # Find the first position from 5' end where window average drops
    cut_site = len(qualities) # Default: no cut needed

    # Check initial window
    if (q_sum / window_size) < min_avg_quality:
        cut_site = 0 # Cut at the beginning if the first window is bad
    else:
        # Slide the window
        for i in range(1, len(qualities) - window_size + 1):
            q_sum = q_sum - qualities[i-1] + qualities[i + window_size - 1]
            if (q_sum / window_size) < min_avg_quality:
                cut_site = i # Trim at the start of this window
                break
    
    if cut_site == 0: # Entire read (from first window) is bad
        # print(f"DEBUG: Record {record.id} first window bad, discarding.")
        return None

    trimmed_record = record[:cut_site]

    if len(trimmed_record) < min_length:
        # print(f"DEBUG: Record {record.id} too short ({len(trimmed_record)}bp) after window trim (min_length {min_length}). Cut at {cut_site}.")
        return None
    return trimmed_record


# --- 5. Processing Functions ---
def process_single_end_file(input_file, output_file, trim_function, min_final_length=30, **trim_kwargs):
    """
    Processes a single-end FASTQ file: reads, trims, and writes.
    `trim_function` is one of the trimming functions defined above.
    `trim_kwargs` are passed to the `trim_function`.
    """
    print(f"\nProcessing Single-End File: {input_file}")
    print(f"Using trim function: {trim_function.__name__} with args: {trim_kwargs}")
    print(f"Minimum final read length: {min_final_length}")
    
    input_records = parse_fastq_gz(input_file)
    trimmed_records = []
    
    count_initial = 0
    count_trimmed_out = 0
    count_discarded_toolongtrim = 0
    count_discarded_tooshort = 0

    for record in input_records:
        count_initial += 1
        original_len = len(record)

        # Add min_length to trim_kwargs for the trimming functions
        current_trim_kwargs = {**trim_kwargs, 'min_length': min_final_length}
        trimmed_rec = trim_function(record, **current_trim_kwargs)
        
        if trimmed_rec:
            if len(trimmed_rec) >= min_final_length:
                trimmed_records.append(trimmed_rec)
                count_trimmed_out +=1
            else: # Should be caught by min_length in trim_function, but as a safeguard
                count_discarded_tooshort += 1
        else: # trim_function returned None (e.g. trimmed to nothing or below its internal min_length)
             count_discarded_toolongtrim +=1


    num_written = write_fastq_gz(trimmed_records, output_file)
    print(f"  Initial reads: {count_initial}")
    print(f"  Reads written to {output_file}: {num_written}")
    print(f"  Reads discarded by trim func (became too short or all bad): {count_discarded_toolongtrim}")
    print(f"  Reads discarded post-trim (safeguard, below min_final_length): {count_discarded_tooshort}")
    print(f"Finished processing {input_file}")


def process_paired_end_files(input_r1, input_r2, output_r1, output_r2,
                             trim_function_r1, trim_function_r2,
                             min_final_length=30,
                             trim_kwargs_r1=None, trim_kwargs_r2=None):
    """
    Processes paired-end FASTQ files. Reads are processed in pairs.
    If one read of a pair is discarded, the other is also discarded.
    `trim_kwargs_r1` and `trim_kwargs_r2` are passed to respective trim functions.
    """
    print(f"\nProcessing Paired-End Files: {input_r1} & {input_r2}")
    print(f"Using R1 trim func: {trim_function_r1.__name__} with args: {trim_kwargs_r1 or {}}")
    print(f"Using R2 trim func: {trim_function_r2.__name__} with args: {trim_kwargs_r2 or {}}")
    print(f"Minimum final read length for BOTH reads: {min_final_length}")

    # Ensure kwargs are dictionaries
    trim_kwargs_r1 = trim_kwargs_r1 or {}
    trim_kwargs_r2 = trim_kwargs_r2 or {}
    
    records_r1 = parse_fastq_gz(input_r1)
    records_r2 = parse_fastq_gz(input_r2)
    
    kept_r1_records = []
    kept_r2_records = []
    
    count_initial_pairs = 0
    count_kept_pairs = 0
    count_discarded_r1_trim = 0
    count_discarded_r2_trim = 0
    count_discarded_r1_len = 0
    count_discarded_r2_len = 0

    for rec1, rec2 in zip(records_r1, records_r2):
        count_initial_pairs += 1

        current_trim_kwargs_r1 = {**trim_kwargs_r1, 'min_length': min_final_length}
        current_trim_kwargs_r2 = {**trim_kwargs_r2, 'min_length': min_final_length}

        trimmed_rec1 = trim_function_r1(rec1, **current_trim_kwargs_r1)
        trimmed_rec2 = trim_function_r2(rec2, **current_trim_kwargs_r2)
        
        # Check if reads were successfully trimmed and meet length criteria
        r1_ok = False
        if trimmed_rec1 and len(trimmed_rec1) >= min_final_length:
            r1_ok = True
        elif not trimmed_rec1: # Discarded by trim function
            count_discarded_r1_trim += 1
        else: # Too short after trimming
            count_discarded_r1_len += 1
            
        r2_ok = False
        if trimmed_rec2 and len(trimmed_rec2) >= min_final_length:
            r2_ok = True
        elif not trimmed_rec2:
            count_discarded_r2_trim += 1
        else:
            count_discarded_r2_len += 1

        if r1_ok and r2_ok:
            kept_r1_records.append(trimmed_rec1)
            kept_r2_records.append(trimmed_rec2)
            count_kept_pairs += 1
        # Else: if one or both failed, the pair is discarded.
        # The individual discard counters above will reflect why.

    num_written_r1 = write_fastq_gz(kept_r1_records, output_r1)
    num_written_r2 = write_fastq_gz(kept_r2_records, output_r2)
    
    print(f"  Initial read pairs: {count_initial_pairs}")
    print(f"  Pairs written to {output_r1} & {output_r2}: {count_kept_pairs} (R1: {num_written_r1}, R2: {num_written_r2})")
    print(f"  Pairs where R1 discarded by trim_func: {count_discarded_r1_trim}")
    print(f"  Pairs where R2 discarded by trim_func: {count_discarded_r2_trim}")
    print(f"  Pairs where R1 became too short: {count_discarded_r1_len}")
    print(f"  Pairs where R2 became too short: {count_discarded_r2_len}")
    print(f"Finished processing {input_r1} & {input_r2}")

# --- 6. File Discovery ---
def find_paired_end_files(directory, r1_pattern="*_R1_*.fastq.gz", r2_pattern_template="*_R2_*.fastq.gz"):
    """
    Finds R1 files in a directory and attempts to find their R2 counterparts.
    Assumes Illumina naming like 'Sample_Name_S1_L001_R1_001.fastq.gz'.
    The r2_pattern_template is used to construct the R2 pattern from R1.
    Returns a list of (r1_filepath, r2_filepath) tuples.
    """
    paired_files = []
    r1_files = glob.glob(os.path.join(directory, r1_pattern))
    
    for r1_file in r1_files:
        # Attempt to construct R2 filename from R1
        # This is a common pattern, but might need adjustment for other naming schemes
        r2_file_candidate = r1_file.replace("_R1_", "_R2_") 
        if os.path.exists(r2_file_candidate):
            paired_files.append((Path(r1_file), Path(r2_file_candidate)))
        else:
            # Try a more generic replacement if the first failed (e.g. R1.fastq.gz -> R2.fastq.gz)
            if "_R1.fastq.gz" in r1_file:
                 r2_file_candidate = r1_file.replace("_R1.fastq.gz", "_R2.fastq.gz")
                 if os.path.exists(r2_file_candidate):
                     paired_files.append((Path(r1_file), Path(r2_file_candidate)))
                 else:
                     print(f"Warning: Could not find R2 pair for {r1_file} (tried {r2_file_candidate})")
            else:
                 print(f"Warning: Could not find R2 pair for {r1_file} (tried {r2_file_candidate})")
    
    if not r1_files:
        print(f"No R1 files found in {directory} with pattern {r1_pattern}")
    if not paired_files and r1_files:
        print(f"Found R1 files, but no corresponding R2 files based on naming convention.")
        
    return paired_files

# --- 7. Main Orchestrator ---
def main():
    parser = argparse.ArgumentParser(
        description="Biopython FASTQ processing learning script.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "action",
        choices=[
            "seq_manip_demo",
            "se_quality_trim",
            "se_window_trim",
            "pe_quality_trim",
            "pe_window_trim",
            "find_pairs_demo",
            "all_demos"
        ],
        help="""Action to perform:
    seq_manip_demo: Demonstrate sequence manipulations on a single read.
    se_quality_trim: Trim single-end reads by per-base quality.
    se_window_trim: Trim single-end reads by sliding window average quality.
    pe_quality_trim: Trim paired-end reads by per-base quality.
    pe_window_trim: Trim paired-end reads by sliding window average quality.
    find_pairs_demo: Demonstrate finding paired-end files.
    all_demos: Run all demonstrations (except find_pairs_demo which needs existing files).
"""
    )
    parser.add_argument(
        "--num_reads", type=int, default=100, help="Number of dummy reads to generate."
    )
    parser.add_argument(
        "--read_length", type=int, default=100, help="Length of dummy reads."
    )
    parser.add_argument(
        "--min_qual", type=int, default=20, help="Minimum Phred quality score for per-base trimming."
    )
    parser.add_argument(
        "--window_size", type=int, default=10, help="Window size for sliding window trimming."
    )
    parser.add_argument(
        "--min_avg_qual", type=int, default=20, help="Minimum average Phred quality for sliding window."
    )
    parser.add_argument(
        "--min_final_len", type=int, default=50, help="Minimum read length after trimming to keep a read."
    )
    parser.add_argument(
        "--data_dir", type=str, default="fastq_data", help="Directory for dummy data and outputs."
    )

    args = parser.parse_args()

    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Shared Dummy File Names ---
    se_input_file = data_dir / "dummy_SE_reads.fastq.gz"
    pe_r1_input_file = data_dir / "dummy_PE_reads_R1.fastq.gz"
    pe_r2_input_file = data_dir / "dummy_PE_reads_R2.fastq.gz"

    # --- Action Dispatch ---
    
    def run_seq_manip_demo():
        print("\n===== Running Sequence Manipulation Demo =====")
        create_dummy_fastq_gz(se_input_file, num_reads=5, read_length=args.read_length, make_bad_quality=False)
        # Get the first record for demo
        try:
            first_record = next(parse_fastq_gz(se_input_file))
            demonstrate_sequence_manipulation(first_record)
        except StopIteration:
            print("Could not get a record from the dummy file for manipulation demo.")
       
