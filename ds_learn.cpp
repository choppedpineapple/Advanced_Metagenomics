/*
Bioinformatics C++ Learning Pipeline
This program teaches core C++ concepts through bioinformatics examples.
Compile with: g++ -o biolearn biolearn.cpp && ./biolearn
*/

// Section 1: Fundamental Includes
#include <iostream>   // Basic input/output operations
#include <fstream>    // File handling operations
#include <string>     // String manipulation
#include <vector>     // Dynamic array functionality

using namespace std; // Standard namespace (note: avoid in large projects)

// Section 2: Function Declaration
double calculateGCContent(const string& sequence); // Function prototype

// Section 3: Class Definition
class DNASequence {
private:
    string sequence;
    string name;

public:
    // Constructor
    DNASequence(const string& seqName, const string& seq) {
        name = seqName;
        sequence = seq;
    }

    // Method to calculate GC content
    double getGCContent() const {
        return calculateGCContent(sequence);
    }

    // Display sequence information
    void displayInfo() const {
        cout << "Sequence: " << name << endl;
        cout << "Length: " << sequence.length() << " bp" << endl;
        cout << "GC Content: " << getGCContent() * 100 << "%" << endl;
    }
};

// Section 4: Main Program Execution
int main() {
    /* ----------------------------
    Concept 1: Variables & Data Types
    ----------------------------
    Fundamental building blocks for storing data */
    string dnaSequence = "ATGCTAGCTAACGT"; // DNA sequence storage
    int sequenceLength = dnaSequence.length(); // Integer variable
    double gcPercentage = 0.0; // Floating-point variable

    /* ----------------------------
    Concept 2: Control Structures
    ----------------------------
    Decision making and repetition */
    // If-else statement
    if (sequenceLength == 0) {
        cerr << "Error: Empty sequence!" << endl;
        return 1;
    }

    // For loop with nucleotide counting
    int aCount = 0, tCount = 0, cCount = 0, gCount = 0;
    for (char nucleotide : dnaSequence) {
        switch (nucleotide) {
            case 'A': aCount++; break;
            case 'T': tCount++; break;
            case 'C': cCount++; break;
            case 'G': gCount++; break;
        }
    }

    /* ----------------------------
    Concept 3: Functions
    ----------------------------
    Reusable code blocks for specific tasks */
    gcPercentage = calculateGCContent(dnaSequence);
    cout << "GC Content: " << gcPercentage * 100 << "%" << endl;

    /* ----------------------------
    Concept 4: File I/O
    ----------------------------
    Reading biological data from files */
    ifstream fastaFile("sequence.fasta");
    string line, header, sequence;

    if (fastaFile.is_open()) {
        while (getline(fastaFile, line)) {
            if (line[0] == '>') { // Header line
                header = line.substr(1); // Remove '>' character
            } else {
                sequence += line; // Concatenate sequence lines
            }
        }
        fastaFile.close();
    } else {
        cerr << "Error opening file!" << endl;
        return 1;
    }

    /* ----------------------------
    Concept 5: Object-Oriented Programming
    ----------------------------
    Creating and using biological objects */
    DNASequence mySequence("Example Gene", sequence);
    mySequence.displayInfo();

    return 0;
}

// Section 5: Function Implementation
double calculateGCContent(const string& sequence) {
    /* Function: Calculate GC content of DNA sequence
    Parameters: const string& - reference to sequence (prevents copying)
    Returns: double - GC percentage (0.0-1.0) */
    
    if (sequence.empty()) return 0.0;

    int gcCount = 0;
    for (char nuc : sequence) {
        if (nuc == 'G' || nuc == 'C') {
            gcCount++;
        }
    }
    return static_cast<double>(gcCount) / sequence.length();
}
