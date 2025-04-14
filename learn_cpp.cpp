/**
 * Bioinformatics Learning Pipeline
 * ================================
 * 
 * This program demonstrates core C++ concepts through a simplified bioinformatics pipeline.
 * It processes DNA sequences using various operations common in bioinformatics,
 * while teaching fundamental C++ programming concepts.
 * 
 * Topics covered:
 * - Basic syntax and structure
 * - Data types and variables
 * - Control structures (loops, conditionals)
 * - Functions and modular design
 * - File I/O
 * - String manipulation
 * - STL containers (vectors, maps)
 * - Classes and objects
 * - Error handling
 * - Algorithm implementation
 */

// Header files inclusion - These provide access to pre-defined functionality
#include <iostream>     // For input/output operations
#include <fstream>      // For file operations
#include <string>       // For string manipulation
#include <vector>       // For dynamic arrays
#include <map>          // For key-value pairs
#include <algorithm>    // For various algorithms (find, sort, etc.)
#include <stdexcept>    // For exception handling
#include <memory>       // For smart pointers

// Namespace declaration - This prevents naming conflicts
// std:: is the standard library namespace, using it means we don't have to prefix std:: everywhere
using namespace std; 
// Note: In larger projects, it's often better practice to use specific using declarations
// e.g., using std::cout; using std::string; rather than the entire namespace

/**
 * Class for representing a DNA sequence.
 * Demonstrates class definition, encapsulation, member functions, and constructors.
 */
class DNASequence {
private:
    // Member variables (attributes)
    string sequence;
    string id;
    
public:
    // Constructor with default arguments
    // This is called when creating a new object of this class
    DNASequence(string seq = "", string identifier = "") : sequence(seq), id(identifier) {
        // Validate that the sequence only contains valid nucleotides
        for (char c : sequence) {
            if (c != 'A' && c != 'T' && c != 'G' && c != 'C' && c != 'N') {
                throw invalid_argument("Invalid nucleotide in sequence: " + string(1, c));
            }
        }
    }
    
    // Getter methods - Provide access to private members
    string getSequence() const {
        return sequence;
    }
    
    string getId() const {
        return id;
    }
    
    // Member function to calculate GC content (percentage of G and C in DNA)
    double calculateGCContent() const {
        if (sequence.empty()) {
            return 0.0;
        }
        
        int gcCount = 0;
        for (char nucleotide : sequence) {
            if (nucleotide == 'G' || nucleotide == 'C') {
                gcCount++;
            }
        }
        
        // Typecasting to double to ensure floating-point division
        return (static_cast<double>(gcCount) / sequence.length()) * 100.0;
    }
    
    // Member function to get the reverse complement of the DNA sequence
    string getReverseComplement() const {
        string complement = sequence;
        
        // First, replace each nucleotide with its complement
        for (char& nucleotide : complement) {
            switch (nucleotide) {
                case 'A': nucleotide = 'T'; break;
                case 'T': nucleotide = 'A'; break;
                case 'G': nucleotide = 'C'; break;
                case 'C': nucleotide = 'G'; break;
                case 'N': nucleotide = 'N'; break; // N remains N
            }
        }
        
        // Then reverse the entire string
        reverse(complement.begin(), complement.end());
        return complement;
    }
    
    // Function to find subsequences (k-mers) of specified length
    map<string, int> findKmers(int k) const {
        map<string, int> kmerCounts;
        
        // Error checking
        if (k <= 0) {
            throw invalid_argument("k must be a positive integer");
        }
        
        if (static_cast<size_t>(k) > sequence.length()) {
            throw invalid_argument("k cannot be larger than sequence length");
        }
        
        // Loop through the sequence and extract all k-mers
        for (size_t i = 0; i <= sequence.length() - k; i++) {
            string kmer = sequence.substr(i, k);
            kmerCounts[kmer]++; // Increment count for this k-mer
        }
        
        return kmerCounts;
    }
    
    // Length of the sequence
    size_t length() const {
        return sequence.length();
    }
};

/**
 * Class for handling FASTA file operations.
 * Demonstrates file I/O in C++.
 */
class FastaHandler {
public:
    // Read sequences from a FASTA file
    // Returns a vector of DNASequence objects
    vector<DNASequence> readFastaFile(const string& filename) {
        vector<DNASequence> sequences;
        ifstream file(filename);
        string line, currentId, currentSeq;
        
        // Check if file opened successfully
        if (!file.is_open()) {
            throw runtime_error("Could not open file: " + filename);
        }
        
        // Process the file line by line
        while (getline(file, line)) {
            // Check if this is a header line (starts with '>')
            if (line.empty()) {
                continue; // Skip empty lines
            } else if (line[0] == '>') {
                // If we already have a sequence, save it before starting a new one
                if (!currentId.empty() && !currentSeq.empty()) {
                    sequences.push_back(DNASequence(currentSeq, currentId));
                }
                
                // Extract ID from the header (removing the '>' character)
                currentId = line.substr(1);
                currentSeq = ""; // Reset the sequence
            } else {
                // Add this line to the current sequence
                currentSeq += line;
            }
        }
        
        // Don't forget to add the last sequence in the file
        if (!currentId.empty() && !currentSeq.empty()) {
            sequences.push_back(DNASequence(currentSeq, currentId));
        }
        
        file.close(); // Close the file
        return sequences;
    }
    
    // Write sequences to a FASTA file
    void writeFastaFile(const string& filename, const vector<DNASequence>& sequences) {
        ofstream file(filename);
        
        // Check if file opened successfully
        if (!file.is_open()) {
            throw runtime_error("Could not open file for writing: " + filename);
        }
        
        // Write each sequence in FASTA format
        for (const auto& seq : sequences) {
            file << ">" << seq.getId() << endl;
            
            // Write sequence in chunks of 60 characters per line (standard FASTA format)
            string sequence = seq.getSequence();
            for (size_t i = 0; i < sequence.length(); i += 60) {
                file << sequence.substr(i, 60) << endl;
            }
        }
        
        file.close(); // Close the file
    }
};

/**
 * Class for performing bioinformatics analyses on DNA sequences.
 * Demonstrates more complex algorithms and data processing.
 */
class BioinformaticsAnalyzer {
private:
    // Vector to store the sequences we're analyzing
    vector<DNASequence> sequences;
    
public:
    // Constructor that takes a vector of sequences
    BioinformaticsAnalyzer(const vector<DNASequence>& seqs) : sequences(seqs) {}
    
    // Calculate GC content for all sequences
    map<string, double> calculateAllGCContents() const {
        map<string, double> results;
        
        // Loop through each sequence and calculate its GC content
        for (const auto& seq : sequences) {
            results[seq.getId()] = seq.calculateGCContent();
        }
        
        return results;
    }
    
    // Find motifs (specific patterns) in sequences
    map<string, vector<size_t>> findMotif(const string& motif) const {
        map<string, vector<size_t>> results;
        
        // Loop through each sequence
        for (const auto& seq : sequences) {
            string sequence = seq.getSequence();
            vector<size_t> positions;
            
            // Search for the motif starting at each position in the sequence
            size_t pos = sequence.find(motif, 0);
            while (pos != string::npos) {
                positions.push_back(pos);
                pos = sequence.find(motif, pos + 1);
            }
            
            // Store the positions for this sequence
            if (!positions.empty()) {
                results[seq.getId()] = positions;
            }
        }
        
        return results;
    }
    
    // Calculate k-mer frequencies across all sequences
    map<string, int> calculateKmerFrequencies(int k) const {
        map<string, int> totalKmerCounts;
        
        // Get k-mers from each sequence and combine the counts
        for (const auto& seq : sequences) {
            try {
                map<string, int> seqKmers = seq.findKmers(k);
                
                // Add these k-mer counts to our total
                for (const auto& kmerPair : seqKmers) {
                    totalKmerCounts[kmerPair.first] += kmerPair.second;
                }
            } catch (const invalid_argument& e) {
                // Skip sequences that are too short for this k value
                cerr << "Warning: " << e.what() << " for sequence " << seq.getId() << endl;
            }
        }
        
        return totalKmerCounts;
    }
    
    // Find shared k-mers between two sequences (useful for alignment seeding)
    vector<string> findSharedKmers(size_t seqIndex1, size_t seqIndex2, int k) const {
        // Check if indices are valid
        if (seqIndex1 >= sequences.size() || seqIndex2 >= sequences.size()) {
            throw out_of_range("Sequence index out of range");
        }
        
        // Get k-mers for each sequence
        map<string, int> kmers1 = sequences[seqIndex1].findKmers(k);
        map<string, int> kmers2 = sequences[seqIndex2].findKmers(k);
        
        // Find shared k-mers
        vector<string> shared;
        for (const auto& kmerPair : kmers1) {
            if (kmers2.find(kmerPair.first) != kmers2.end()) {
                shared.push_back(kmerPair.first);
            }
        }
        
        return shared;
    }
    
    // Basic pairwise global alignment using simplified Needleman-Wunsch algorithm
    // This is a teaching example - real bioinformatics would use optimized libraries
    string alignSequences(size_t seqIndex1, size_t seqIndex2) const {
        // Check if indices are valid
        if (seqIndex1 >= sequences.size() || seqIndex2 >= sequences.size()) {
            throw out_of_range("Sequence index out of range");
        }
        
        // For simplicity, we'll just implement a very basic alignment
        // Real bioinformatics software uses more sophisticated algorithms
        
        string seq1 = sequences[seqIndex1].getSequence();
        string seq2 = sequences[seqIndex2].getSequence();
        
        // For teaching purposes, we'll limit sequence length to avoid excessive computation
        const size_t MAX_ALIGN_LENGTH = 500;
        if (seq1.length() > MAX_ALIGN_LENGTH || seq2.length() > MAX_ALIGN_LENGTH) {
            return "Sequences too long for basic alignment. Using first " + 
                   to_string(MAX_ALIGN_LENGTH) + " bases.\n";
        }
        
        // Trim sequences if needed
        if (seq1.length() > MAX_ALIGN_LENGTH) seq1 = seq1.substr(0, MAX_ALIGN_LENGTH);
        if (seq2.length() > MAX_ALIGN_LENGTH) seq2 = seq2.substr(0, MAX_ALIGN_LENGTH);
        
        // Create alignment result
        string result = "Basic alignment between " + sequences[seqIndex1].getId() + 
                       " and " + sequences[seqIndex2].getId() + ":\n";
        
        // Compare sequences character by character (very simplified)
        size_t minLength = min(seq1.length(), seq2.length());
        int matches = 0;
        
        for (size_t i = 0; i < minLength; i++) {
            if (seq1[i] == seq2[i]) {
                matches++;
            }
        }
        
        // Calculate percent identity
        double percentIdentity = (static_cast<double>(matches) / minLength) * 100.0;
        
        result += "Sequence 1: " + seq1 + "\n";
        result += "Sequence 2: " + seq2 + "\n";
        result += "Matches: " + to_string(matches) + " out of " + to_string(minLength) + "\n";
        result += "Percent identity: " + to_string(percentIdentity) + "%\n";
        
        return result;
    }
};

/**
 * Function to generate a simple random DNA sequence.
 * Demonstrates random generation and string building.
 * 
 * @param length The desired length of the sequence
 * @return A randomly generated DNA sequence
 */
string generateRandomDNA(size_t length) {
    string nucleotides = "ACGT";
    string result;
    result.reserve(length); // Optimize by pre-allocating memory
    
    // Initialize random seed based on current time
    srand(static_cast<unsigned int>(time(nullptr)));
    
    // Generate random nucleotides
    for (size_t i = 0; i < length; i++) {
        size_t randomIndex = rand() % 4; // Generate random index between 0-3
        result += nucleotides[randomIndex];
    }
    
    return result;
}

/**
 * Function to find all open reading frames (ORFs) in a sequence.
 * Demonstrates string manipulation and bioinformatics algorithms.
 * 
 * ORFs are regions between a start codon (ATG) and a stop codon (TAG, TAA, TGA).
 */
vector<string> findOpenReadingFrames(const DNASequence& sequence) {
    vector<string> orfs;
    string seq = sequence.getSequence();
    
    // Define start and stop codons
    const string START_CODON = "ATG";
    const vector<string> STOP_CODONS = {"TAG", "TAA", "TGA"};
    
    // Search in all three reading frames
    for (int frame = 0; frame < 3; frame++) {
        size_t pos = frame;
        
        while (pos + 3 <= seq.length()) {
            // Look for start codon
            if (seq.substr(pos, 3) == START_CODON) {
                size_t orfStart = pos;
                bool foundStop = false;
                
                // Look for the next stop codon in this frame
                for (size_t i = pos + 3; i + 3 <= seq.length(); i += 3) {
                    string codon = seq.substr(i, 3);
                    
                    // Check if this is a stop codon
                    if (find(STOP_CODONS.begin(), STOP_CODONS.end(), codon) != STOP_CODONS.end()) {
                        // We found a complete ORF
                        string orf = seq.substr(orfStart, i + 3 - orfStart);
                        orfs.push_back(orf);
                        pos = i + 3; // Move past this ORF
                        foundStop = true;
                        break;
                    }
                }
                
                if (!foundStop) {
                    // If we didn't find a stop codon, move to the next position
                    pos += 3;
                }
            } else {
                // Move to the next position
                pos += 3;
            }
        }
    }
    
    return orfs;
}

/**
 * Function to translate a DNA sequence to protein (amino acid sequence).
 * Demonstrates map usage and genetic code implementation.
 */
string translateDNA(const string& dnaSequence) {
    // Define the genetic code (DNA codon to amino acid mapping)
    map<string, char> geneticCode = {
        {"TTT", 'F'}, {"TTC", 'F'}, {"TTA", 'L'}, {"TTG", 'L'},
        {"CTT", 'L'}, {"CTC", 'L'}, {"CTA", 'L'}, {"CTG", 'L'},
        {"ATT", 'I'}, {"ATC", 'I'}, {"ATA", 'I'}, {"ATG", 'M'},
        {"GTT", 'V'}, {"GTC", 'V'}, {"GTA", 'V'}, {"GTG", 'V'},
        {"TCT", 'S'}, {"TCC", 'S'}, {"TCA", 'S'}, {"TCG", 'S'},
        {"CCT", 'P'}, {"CCC", 'P'}, {"CCA", 'P'}, {"CCG", 'P'},
        {"ACT", 'T'}, {"ACC", 'T'}, {"ACA", 'T'}, {"ACG", 'T'},
        {"GCT", 'A'}, {"GCC", 'A'}, {"GCA", 'A'}, {"GCG", 'A'},
        {"TAT", 'Y'}, {"TAC", 'Y'}, {"TAA", '*'}, {"TAG", '*'},
        {"CAT", 'H'}, {"CAC", 'H'}, {"CAA", 'Q'}, {"CAG", 'Q'},
        {"AAT", 'N'}, {"AAC", 'N'}, {"AAA", 'K'}, {"AAG", 'K'},
        {"GAT", 'D'}, {"GAC", 'D'}, {"GAA", 'E'}, {"GAG", 'E'},
        {"TGT", 'C'}, {"TGC", 'C'}, {"TGA", '*'}, {"TGG", 'W'},
        {"CGT", 'R'}, {"CGC", 'R'}, {"CGA", 'R'}, {"CGG", 'R'},
        {"AGT", 'S'}, {"AGC", 'S'}, {"AGA", 'R'}, {"AGG", 'R'},
        {"GGT", 'G'}, {"GGC", 'G'}, {"GGA", 'G'}, {"GGG", 'G'}
    };
    
    string protein;
    
    // Process the DNA sequence in codons (3 nucleotides at a time)
    for (size_t i = 0; i + 2 < dnaSequence.length(); i += 3) {
        string codon = dnaSequence.substr(i, 3);
        
        // If the codon is in our genetic code map, add the corresponding amino acid
        if (geneticCode.find(codon) != geneticCode.end()) {
            protein += geneticCode[codon];
        } else {
            // For partial codons or invalid nucleotides
            protein += 'X';
        }
    }
    
    return protein;
}

/**
 * Class to demonstrate working with protein sequences
 */
class ProteinAnalyzer {
private:
    map<char, double> aminoAcidMasses = {
        {'A', 71.07}, {'R', 156.19}, {'N', 114.08}, {'D', 115.08},
        {'C', 103.10}, {'E', 129.11}, {'Q', 128.13}, {'G', 57.05},
        {'H', 137.14}, {'I', 113.16}, {'L', 113.16}, {'K', 128.17},
        {'M', 131.19}, {'F', 147.17}, {'P', 97.11}, {'S', 87.07},
        {'T', 101.10}, {'W', 186.21}, {'Y', 163.17}, {'V', 99.13},
        {'*', 0.0}  // Stop codon
    };
    
    string proteinSequence;
    
public:
    ProteinAnalyzer(const string& sequence) : proteinSequence(sequence) {}
    
    // Calculate molecular weight of the protein
    double calculateMolecularWeight() const {
        double weight = 0.0;
        
        for (char aa : proteinSequence) {
            if (aminoAcidMasses.find(aa) != aminoAcidMasses.end()) {
                weight += aminoAcidMasses.at(aa);
            }
        }
        
        // Add weight of water molecule (subtract H2O for each peptide bond)
        weight += 18.01 - ((proteinSequence.length() - 1) * 18.01);
        
        return weight;
    }
    
    // Calculate amino acid composition
    map<char, int> calculateAminoAcidComposition() const {
        map<char, int> composition;
        
        for (char aa : proteinSequence) {
            composition[aa]++;
        }
        
        return composition;
    }
};

/**
 * Function to demonstrate basic data structures and algorithms with a sequence alignment example
 */
void demonstrateSequenceAlignment() {
    cout << "\n--- Demonstrating Sequence Alignment ---\n" << endl;
    
    // Create two simple DNA sequences
    string seq1 = "ACGTACGTACGT";
    string seq2 = "ACGTTCGTACGT";
    
    cout << "Sequence 1: " << seq1 << endl;
    cout << "Sequence 2: " << seq2 << endl;
    
    // A simple alignment visualization
    cout << "Alignment:" << endl;
    cout << seq1 << endl;
    
    // Create a match/mismatch string
    string matchString;
    for (size_t i = 0; i < min(seq1.length(), seq2.length()); i++) {
        matchString += (seq1[i] == seq2[i]) ? '|' : ' ';
    }
    cout << matchString << endl;
    cout << seq2 << endl;
    
    // Calculate percent identity
    int matches = 0;
    for (size_t i = 0; i < min(seq1.length(), seq2.length()); i++) {
        if (seq1[i] == seq2[i]) {
            matches++;
        }
    }
    
    double percentIdentity = (static_cast<double>(matches) / min(seq1.length(), seq2.length())) * 100.0;
    cout << "\nPercent identity: " << percentIdentity << "%" << endl;
}

/**
 * Main function - Entry point of the program
 * Demonstrates program flow and orchestration
 */
int main() {
    cout << "=== Bioinformatics C++ Learning Pipeline ===" << endl;
    cout << "This program demonstrates C++ concepts through bioi
