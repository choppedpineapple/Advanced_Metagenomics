#!/usr/bin/env python3

import os

def import_files():
        while True:
            try:
                work_dir=input("Enter the path to the working directory: ").strip()
                if not work_dir:
                    raise ValueError("The working directory cannot be empty.")
                
                if not os.path.exists(work_dir):
                    raise FileNotFoundError(f"Error: Working directory '{work_dir}' does not exist")

                prj = input("Enter the name of the project folder: ").strip()
                if not prj:
                    raise ValueError("The project directory cannot be empty.")

                input_dir = os.path.join(work_dir, prj)
                if not os.path.exists(input_dir):
                    raise FileNotFoundError(f"Error: Project directory '{input_dir}' does not exist")
                
                output_dir = os.path.join(work_dir, prj + "_output")
                os.makedirs(output_dir, exist_ok=True)
                print(f"Output directory created: {output_dir}")
                
                print("All directories verified.")
                return input_dir, output_dir
                
            except ValueError as ve:
                print(ve)
                print("Please provide a valid input.")
            except FileNotFoundError as fe:
                print(fe)
                print("Please provide a valid working directory or project directory.")

def iterate_folders(input_dir, output_dir):
    print(f"The output files will be written to {output_dir}")
    
    valid_samples = 0
    for sample_dir in os.listdir(input_dir):
        sample_path = os.path.join(input_dir, sample_dir)
        if os.path.isdir(sample_path):
            fastq_files = [f for f in os.listdir(sample_path) if f.endswith(".fastq")]
            if len(fastq_files) == 2:
                print(f"Sample '{sample_dir}' has forward and reverse reads: {fastq_files}")
                valid_samples += 1
            else:
                print(f"Sample '{sample_dir}' does not have exactly two FASTQ files or they are missing.")
    
    print(f"Found {valid_samples} valid samples with paired FASTQ files.")
            
def main():
    input_dir, output_dir = import_files()
    iterate_folders(input_dir, output_dir)

if __name__ == "__main__":
    main()
