#include <iostream>              // For input/output with the user
#include <filesystem>            // For directory iteration and file manipulation (C++17)
#include <fstream>               // For writing output files
#include <sstream>               // For string stream processing
#include <vector>                // For dynamic arrays
#include <string>                // For strings
#include <algorithm>             // For string comparison
#include <zlib.h>                // For reading gzipped files

namespace fs = std::filesystem;  // Shorten the namespace for easier use

// Function to check if a filename matches Illumina paired-end pattern (simplified)
bool isFastqGz(const std::string& fname) {
    // Looks for files ending with ".fastq.gz" or ".fq.gz"
    return (fname.size() > 8 &&
            (fname.compare(fname.size()-8, 8, ".fastq.gz") == 0 ||
             fname.compare(fname.size()-6, 6, ".fq.gz") == 0));
}

// Function to process a gzipped FASTQ file
void processFastqGz(const fs::path& input_file,
                    std::ofstream& short_reads_out,
                    int min_len,
                    const std::string& sample_name,
                    const std::string& read_pair) {
    // Open gzipped file for reading using zlib's gzopen
    gzFile gzfp = gzopen(input_file.string().c_str(), "rb");
    if (!gzfp) {
        std::cerr << "Failed to open " << input_file << " for reading.\n";
        return;
    }

    // Buffers for reading lines from gzipped file
    const int buf_size = 8192;
    char buffer[buf_size];

    // Each FASTQ record is 4 lines: header, sequence, plus, quality
    std::vector<std::string> record(4);

    // Keep reading until end of file
    while (true) {
        bool eof = false;
        for (int i = 0; i < 4; ++i) {
            if (gzgets(gzfp, buffer, buf_size) == nullptr) {
                eof = true;
                break;
            }
            record[i] = buffer;
            // Remove possible trailing newline character
            if (!record[i].empty() && (record[i].back() == '\n' || record[i].back() == '\r'))
                record[i].erase(record[i].find_last_not_of("\r\n") + 1);
        }
        if (eof)
            break;

        // Check sequence length (record[1])
        if ((int)record[1].length() < min_len) {
            // Write short read to short_reads.txt with sample and read_pair annotation
            short_reads_out << "Sample: " << sample_name
                            << " | File: " << input_file.filename().string()
                            << " | ReadPair: " << read_pair << "\n";
            for (const auto& line : record) {
                short_reads_out << line << "\n";
            }
            short_reads_out << "\n";
        }
        // Otherwise, do nothing (in real use, you might want to write the kept reads to another file)
    }
    gzclose(gzfp); // Always close files!
}

int main() {
    // ----------------------- User Input Section -------------------------
    std::string input_dir;
    int min_length;

    std::cout << "Enter the path to the input directory containing sample folders: ";
    std::getline(std::cin, input_dir);

    std::cout << "Enter the minimum read length to keep: ";
    std::cin >> min_length;

    // Sanitize input directory path
    fs::path input_path = fs::path(input_dir);

    // ----------------------- Output Directory Setup ---------------------
    // Append "_output" to the input directory name to create output directory
    std::string output_dir = input_path.filename().string() + "_output";
    fs::path output_path = input_path.parent_path() / output_dir;

    // Create the output directory if it doesn't exist
    if (!fs::exists(output_path)) {
        fs::create_directory(output_path);
        std::cout << "Created output directory: " << output_path << "\n";
    } else {
        std::cout << "Output directory already exists: " << output_path << "\n";
    }

    // Open the short_reads.txt output file for writing all short reads
    std::ofstream short_reads_out(output_path / "short_reads.txt");
    if (!short_reads_out) {
        std::cerr << "Failed to create short_reads.txt in " << output_path << "\n";
        return 1;
    }

    // ---------------------- Directory Traversal --------------------------
    // Iterate through all subdirectories (i.e., sample folders) in the input directory
    for (const auto& entry : fs::directory_iterator(input_path)) {
        if (entry.is_directory()) {
            fs::path sample_dir = entry.path();
            std::string sample_name = sample_dir.filename().string();

            std::cout << "Processing sample directory: " << sample_name << "\n";

            // Find paired-end fastq.gz files in this sample directory
            // We'll look for files containing "_R1" and "_R2" in their names
            std::vector<fs::path> r1_files, r2_files;

            for (const auto& file_entry : fs::directory_iterator(sample_dir)) {
                if (!file_entry.is_regular_file())
                    continue;
                std::string fname = file_entry.path().filename().string();
                if (isFastqGz(fname)) {
                    if (fname.find("_R1") != std::string::npos)
                        r1_files.push_back(file_entry.path());
                    else if (fname.find("_R2") != std::string::npos)
                        r2_files.push_back(file_entry.path());
                }
            }

            // Process R1 files
            for (const auto& r1 : r1_files) {
                std::cout << "  Processing file: " << r1.filename() << " (R1)\n";
                processFastqGz(r1, short_reads_out, min_length, sample_name, "R1");
            }
            // Process R2 files
            for (const auto& r2 : r2_files) {
                std::cout << "  Processing file: " << r2.filename() << " (R2)\n";
                processFastqGz(r2, short_reads_out, min_length, sample_name, "R2");
            }
        }
    }

    // Close the output file
    short_reads_out.close();

    std::cout << "Processing complete! Short reads have been written to "
              << (output_path / "short_reads.txt") << "\n";
    return 0;
}

/*
====================
DETAILED EXPLANATIONS
====================

1. <filesystem>
   - Used for traversing directories and handling file paths in a platform-independent way (since C++17).
   - fs::directory_iterator allows you to loop through all items in a directory.

2. <zlib.h>
   - zlib is a popular compression library.
   - gzopen opens a .gz file for reading ("rb" = read binary).
   - gzgets reads a line from the gzipped file (like fgets for gzipped files).
   - gzclose closes the gzipped file.

3. Output Directory Creation
   - By appending "_output" to the input directory name, we create a new sibling directory for outputs.
   - fs::create_directory creates the directory if it doesn't already exist.

4. Sample Directory Processing
   - Each sample directory is assumed to contain paired-end files.
   - Paired-end files are matched based on "_R1"/"_R2" in their filenames.
   - Only files ending in ".fastq.gz" or ".fq.gz" are considered.

5. FASTQ File Processing
   - Reads are processed in blocks of 4 lines (standard FASTQ record).
   - Reads with sequence length below the user threshold are written to "short_reads.txt".
   - Each entry in "short_reads.txt" is annotated with sample, filename, and read pair info.

6. Error Handling
   - If a file or directory can't be opened/created, the program outputs an error and skips it.

7. Flexibility
   - The code can easily be modified to process single-end files or to write kept reads to new FASTQ files.
   - For real bioinformatics usage, you might also keep track of statistics or write filtered reads to new gzipped files.

8. Compilation
   - Compile with: g++ -std=c++17 -lz yourfile.cpp -o pipeline
   - "-lz" links zlib.

====================
END OF EXPLANATIONS
====================
*/
