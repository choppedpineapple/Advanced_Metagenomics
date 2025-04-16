/*
 * BioC++ Learning Pipeline
 * 
 * This program teaches C++ fundamentals through bioinformatics examples.
 * It progresses from basic syntax to file I/O, then to bioinformatics-specific
 * applications like sequence analysis.
 * 
 * Each section has:
 * 1. Concept explanation
 * 2. Example code
 * 3. Interactive exercise
 * 4. Application to bioinformatics
 */

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <map>
#include <unordered_map>

using namespace std;

// ----------------- SECTION 1: C++ FUNDAMENTALS -----------------

void basics_introduction() {
    /*
     * BASIC SYNTAX AND STRUCTURE
     * 
     * Every C++ program has:
     * - #include directives (import libraries)
     * - main() function (program entry point)
     * - Statements end with semicolons ;
     * - Code blocks enclosed in braces {}
     * - Comments with // (single-line) or /* */ (multi-line)
     */
    
    cout << "\n=== Section 1: C++ Basics ===\n";
    cout << "Let's start with fundamental concepts:\n";
    
    // Variables and Data Types
    /*
     * VARIABLES store data in memory. They must be declared with a type.
     * Common primitive types:
     * - int: whole numbers
     * - double: floating-point numbers
     * - char: single characters
     * - bool: true/false
     * - string: sequence of characters (from std library)
     */
    int dna_length = 100;          // Integer
    double gc_content = 58.3;      // Floating point
    char nucleotide = 'A';         // Single character
    bool is_coding = true;         // Boolean
    string gene_name = "BRCA1";    // String
    
    cout << "\nVariable examples:\n";
    cout << "DNA length: " << dna_length << " bp\n";
    cout << "GC content: " << gc_content << "%\n";
    cout << "Nucleotide: " << nucleotide << "\n";
    cout << "Is coding? " << (is_coding ? "Yes" : "No") << "\n";
    cout << "Gene name: " << gene_name << "\n";
    
    // Constants
    const double PI = 3.14159;     // const means value can't change
    const int CODONS_PER_AMINO_ACID = 3;
    
    // User Input
    cout << "\nLet's get some input. Enter a DNA sequence: ";
    string dna_sequence;
    cin >> dna_sequence;
    cout << "You entered: " << dna_sequence << "\n";
}

void control_structures() {
    /*
     * CONTROL STRUCTURES
     * 
     * Allow programs to make decisions and repeat actions
     */
    
    cout << "\n=== Control Structures ===\n";
    
    // If-else statements
    int read_count;
    cout << "Enter number of sequencing reads: ";
    cin >> read_count;
    
    if (read_count > 1000000) {
        cout << "High-throughput sequencing!\n";
    } else if (read_count > 1000) {
        cout << "Moderate sequencing depth\n";
    } else {
        cout << "Low coverage\n";
    }
    
    // Switch statement
    char base;
    cout << "Enter a nucleotide (A, T, C, or G): ";
    cin >> base;
    
    switch (base) {
        case 'A': cout << "Adenine\n"; break;
        case 'T': cout << "Thymine\n"; break;
        case 'C': cout << "Cytosine\n"; break;
        case 'G': cout << "Guanine\n"; break;
        default: cout << "Invalid nucleotide\n";
    }
    
    // Loops
    cout << "\n=== Loops ===\n";
    
    // For loop - when you know how many iterations
    cout << "Counting codons:\n";
    for (int i = 1; i <= 5; i++) {
        cout << "Codon " << i << "\n";
    }
    
    // While loop - when condition is unknown
    cout << "Enter nucleotides one by one (0 to stop):\n";
    char nt;
    while (cin >> nt && nt != '0') {
        cout << "Added " << nt << " to sequence\n";
    }
}

void functions_and_scope() {
    /*
     * FUNCTIONS AND SCOPE
     * 
     * Functions:
     * - Reusable blocks of code
     * - Can take parameters and return values
     * - Help organize code
     * 
     * Scope:
     * - Where variables are accessible
     * - Variables declared inside {} are local to that block
     */
    
    cout << "\n=== Functions and Scope ===\n";
    
    // Function declaration
    auto calculate_gc_content = [](const string& sequence) -> double {
        int gc_count = 0;
        for (char nt : sequence) {
            if (nt == 'G' || nt == 'C') gc_count++;
        }
        return (sequence.empty()) ? 0 : (100.0 * gc_count / sequence.length());
    };
    
    string seq;
    cout << "Enter a DNA sequence to calculate GC content: ";
    cin >> seq;
    double gc = calculate_gc_content(seq);
    cout << "GC content: " << gc << "%\n";
    
    // Function with multiple parameters
    auto transcribe_dna_to_rna = [](string dna) -> string {
        for (char& c : dna) {
            if (c == 'T') c = 'U';
        }
        return dna;
    };
    
    cout << "Transcribed RNA: " << transcribe_dna_to_rna(seq) << "\n";
}

// ----------------- SECTION 2: COMPOUND DATA TYPES -----------------

void vectors_and_arrays() {
    /*
     * VECTORS AND ARRAYS
     * 
     * Arrays: fixed-size collection of elements
     * Vectors: dynamic-size collection (from std library)
     */
    
    cout << "\n=== Section 2: Compound Data Types ===\n";
    
    // Array example (fixed size)
    const int NUM_NUCLEOTIDES = 4;
    char nucleotides[NUM_NUCLEOTIDES] = {'A', 'T', 'C', 'G'};
    
    cout << "Nucleotides in DNA:\n";
    for (int i = 0; i < NUM_NUCLEOTIDES; i++) {
        cout << nucleotides[i] << " ";
    }
    cout << "\n";
    
    // Vector example (dynamic size)
    vector<string> genes;
    genes.push_back("BRCA1");
    genes.push_back("TP53");
    genes.push_back("EGFR");
    
    cout << "\nGene list:\n";
    for (const auto& gene : genes) {
        cout << gene << "\n";
    }
    
    // Bioinformatics example: k-mer counting
    string dna_sequence = "ATGCGATCGATCGATCG";
    int k = 3;
    vector<string> kmers;
    
    for (int i = 0; i <= dna_sequence.length() - k; i++) {
        kmers.push_back(dna_sequence.substr(i, k));
    }
    
    cout << "\n3-mers in sequence " << dna_sequence << ":\n";
    for (const auto& kmer : kmers) {
        cout << kmer << " ";
    }
    cout << "\n";
}

void maps_and_hash_tables() {
    /*
     * MAPS AND HASH TABLES
     * 
     * Key-value pairs for efficient lookup
     * - map: ordered (tree-based)
     * - unordered_map: hash table (usually faster)
     */
    
    cout << "\n=== Maps and Hash Tables ===\n";
    
    // Count nucleotide frequencies
    string sequence = "ATGCGATCGATCGATCG";
    unordered_map<char, int> nucleotide_counts;
    
    for (char nt : sequence) {
        nucleotide_counts[nt]++;
    }
    
    cout << "Nucleotide counts in " << sequence << ":\n";
    for (const auto& pair : nucleotide_counts) {
        cout << pair.first << ": " << pair.second << "\n";
    }
    
    // Codon to amino acid mapping
    map<string, string> codon_table = {
        {"ATG", "Met"}, {"TTC", "Phe"}, {"GAT", "Asp"},
        {"TGG", "Trp"}, {"TAG", "STOP"}, {"TAA", "STOP"}
    };
    
    cout << "\nCodon table examples:\n";
    cout << "ATG -> " << codon_table["ATG"] << "\n";
    cout << "TAG -> " << codon_table["TAG"] << "\n";
}

// ----------------- SECTION 3: FILE I/O AND ERROR HANDLING -----------------

void file_io_example() {
    /*
     * FILE INPUT/OUTPUT
     * 
     * Reading from and writing to files is essential for bioinformatics
     * - ifstream for input files
     * - ofstream for output files
     */
    
    cout << "\n=== Section 3: File I/O ===\n";
    
    // Writing to a file
    ofstream outfile("sequence.fasta");
    if (outfile.is_open()) {
        outfile << ">Sample1\n";
        outfile << "ATGCGATCGATCGATCG\n";
        outfile.close();
        cout << "Created FASTA file: sequence.fasta\n";
    } else {
        cerr << "Unable to create file\n";
    }
    
    // Reading from a file
    ifstream infile("sequence.fasta");
    string line;
    cout << "\nReading FASTA file:\n";
    
    if (infile.is_open()) {
        while (getline(infile, line)) {
            if (line[0] == '>') {
                cout << "Header: " << line.substr(1) << "\n";
            } else {
                cout << "Sequence: " << line << "\n";
            }
        }
        infile.close();
    } else {
        cerr << "Unable to open file\n";
    }
}

void error_handling() {
    /*
     * ERROR HANDLING
     * 
     * Important for robust bioinformatics tools
     * - Exceptions handle unexpected conditions
     * - try/catch blocks manage errors
     */
    
    cout << "\n=== Error Handling ===\n";
    
    auto validate_dna_sequence = [](const string& seq) {
        for (char c : seq) {
            if (c != 'A' && c != 'T' && c != 'C' && c != 'G') {
                throw invalid_argument("Invalid nucleotide in sequence: " + string(1, c));
            }
        }
    };
    
    string test_seq = "ATGCXTG";  // Contains invalid 'X'
    
    try {
        cout << "Validating sequence: " << test_seq << "\n";
        validate_dna_sequence(test_seq);
        cout << "Sequence is valid\n";
    } catch (const invalid_argument& e) {
        cerr << "Error: " << e.what() << "\n";
    }
}

// ----------------- SECTION 4: OBJECT-ORIENTED PROGRAMMING -----------------

class Gene {
private:
    string name;
    string sequence;
    int length;
    
public:
    // Constructor
    Gene(const string& n, const string& seq) : name(n), sequence(seq) {
        length = seq.length();
    }
    
    // Member functions
    void print_info() const {
        cout << "Gene: " << name << "\n";
        cout << "Length: " << length << " bp\n";
        cout << "GC content: " << calculate_gc_content() << "%\n";
    }
    
    double calculate_gc_content() const {
        int gc_count = 0;
        for (char nt : sequence) {
            if (nt == 'G' || nt == 'C') gc_count++;
        }
        return (length > 0) ? (100.0 * gc_count / length) : 0;
    }
    
    string transcribe_to_rna() const {
        string rna = sequence;
        for (char& c : rna) {
            if (c == 'T') c = 'U';
        }
        return rna;
    }
    
    // Getters
    string get_name() const { return name; }
    string get_sequence() const { return sequence; }
    int get_length() const { return length; }
};

void oop_example() {
    /*
     * OBJECT-ORIENTED PROGRAMMING
     * 
     * Key concepts:
     * - Classes: blueprints for objects
     * - Objects: instances of classes
     * - Encapsulation: bundling data and methods
     * - Inheritance: creating hierarchies
     */
    
    cout << "\n=== Section 4: Object-Oriented Programming ===\n";
    
    // Create Gene object
    Gene brca1("BRCA1", "ATGCGATCGATCGATCG");
    brca1.print_info();
    
    cout << "\nTranscribed RNA: " << brca1.transcribe_to_rna() << "\n";
    
    // Inheritance example
    class CodingGene : public Gene {
    private:
        string protein_sequence;
        
    public:
        CodingGene(const string& n, const string& dna, const string& prot)
            : Gene(n, dna), protein_sequence(prot) {}
            
        void print_info() const {
            Gene::print_info();
            cout << "Protein length: " << protein_sequence.length() << " aa\n";
        }
    };
    
    CodingGene tp53("TP53", "ATGCGATCGATCGATCG", "MEEPQSDPSV");
    cout << "\nCoding gene info:\n";
    tp53.print_info();
}

// ----------------- SECTION 5: BIOINFORMATICS APPLICATIONS -----------------

void sequence_analysis() {
    /*
     * SEQUENCE ANALYSIS
     * 
     * Practical bioinformatics examples
     */
    
    cout << "\n=== Section 5: Bioinformatics Applications ===\n";
    
    // Read FASTA file (simplified)
    vector<pair<string, string>> sequences;
    sequences.emplace_back("Gene1", "ATGCGATCGATCGATCG");
    sequences.emplace_back("Gene2", "TTGGCCTAGCTAGCTAG");
    
    // Analyze each sequence
    for (const auto& seq_pair : sequences) {
        const string& name = seq_pair.first;
        const string& sequence = seq_pair.second;
        
        cout << "\nAnalyzing " << name << " (" << sequence.length() << " bp)\n";
        
        // Count nucleotides
        map<char, int> counts;
        for (char nt : sequence) counts[nt]++;
        
        cout << "Nucleotide counts:\n";
        for (const auto& pair : counts) {
            cout << pair.first << ": " << pair.second << "\n";
        }
        
        // Calculate GC content
        double gc_content = 0;
        if (sequence.length() > 0) {
            int gc = counts['G'] + counts['C'];
            gc_content = 100.0 * gc / sequence.length();
        }
        cout << "GC content: " << gc_content << "%\n";
    }
    
    // Find ORFs (simplified)
    string dna_seq = "ATGAAATGAATAG";
    cout << "\nFinding ORFs in sequence: " << dna_seq << "\n";
    
    for (int i = 0; i < dna_seq.length() - 2; i++) {
        string codon = dna_seq.substr(i, 3);
        if (codon == "ATG") {
            cout << "Start codon at position " << i << "\n";
        } else if (codon == "TAA" || codon == "TAG" || codon == "TGA") {
            cout << "Stop codon at position " << i << "\n";
        }
    }
}

void kmer_analysis() {
    /*
     * KMER ANALYSIS
     * 
     * Counting subsequences of length k
     */
    
    cout << "\n=== K-mer Analysis ===\n";
    
    string sequence = "ATGCGATCGATCGATCG";
    int k = 2;  // dinucleotides
    
    unordered_map<string, int> kmer_counts;
    
    for (int i = 0; i <= sequence.length() - k; i++) {
        string kmer = sequence.substr(i, k);
        kmer_counts[kmer]++;
    }
    
    cout << k << "-mer counts for " << sequence << ":\n";
    for (const auto& pair : kmer_counts) {
        cout << pair.first << ": " << pair.second << "\n";
    }
}

// ----------------- MAIN PROGRAM -----------------

int main() {
    cout << "Welcome to BioC++ Learning Pipeline!\n";
    cout << "This program will teach you C++ through bioinformatics examples.\n";
    
    // Section 1: Fundamentals
    basics_introduction();
    control_structures();
    functions_and_scope();
    
    // Section 2: Compound Data Types
    vectors_and_arrays();
    maps_and_hash_tables();
    
    // Section 3: File I/O and Error Handling
    file_io_example();
    error_handling();
    
    // Section 4: Object-Oriented Programming
    oop_example();
    
    // Section 5: Bioinformatics Applications
    sequence_analysis();
    kmer_analysis();
    
    cout << "\nCongratulations! You've completed the BioC++ Learning Pipeline.\n";
    cout << "Try modifying the code examples to explore further.\n";
    
    return 0;
}
