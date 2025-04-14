// bioinformatics_pipeline_tutorial.cpp
// A learning-focused C++ program demonstrating core concepts through DNA analysis

#include <iostream>   // For input/output operations
#include <fstream>    // For file handling
#include <string>     // For string manipulation
#include <cmath>      // For math functions (e.g., rounding)

using namespace std;  // Standard namespace to avoid std:: prefixes

// Function prototype (declaration)
double calculate_gc_content(const string& sequence);

int main() {
    /* 1. VARIABLES AND DATA TYPES */
    string dna_sequence = "";  // Store DNA sequence
    int gc_count = 0;          // Count of G/C nucleotides
    double gc_percent = 0.0;   // GC percentage (decimal)
    const float MAX_GC = 100.0;// Constant for maximum possible GC%

    /* 2. INPUT/OUTPUT OPERATIONS */
    cout << "Enter DNA sequence (A/T/G/C only): ";
    cin >> dna_sequence;

    /* 3. FUNCTION CALL */
    gc_percent = calculate_gc_content(dna_sequence);

    /* 4. CONTROL STRUCTURES - IF/ELSE */
    if(gc_percent > 60.0) {
        cout << "High GC content detected!" << endl;
    } else if(gc_percent < 40.0) {
        cout << "Low GC content detected!" << endl;
    } else {
        cout << "Moderate GC content" << endl;
    }

    /* 5. FILE HANDLING */
    ofstream output_file("dna_analysis.txt");
    if(output_file.is_open()) {
        output_file << "Sequence: " << dna_sequence << "\n";
        output_file << "GC Content: " << round(gc_percent) << "%\n";
        output_file.close();
    } else {
        cerr << "Error creating output file!" << endl;
    }

    return 0;  // Main function should return integer
}

/* 6. FUNCTION DEFINITION */
double calculate_gc_content(const string& sequence) {
    int gc = 0;
    // Range-based for loop (C++11 feature)
    for(char nucleotide : sequence) {
        /* 7. CONDITIONAL STATEMENTS */
        switch(toupper(nucleotide)) {
            case 'G':
            case 'C':
                gc++;
                break;
            case 'A':
            case 'T':
                break;
            default:
                cerr << "Invalid nucleotide: " << nucleotide << endl;
        }
    }
    
    /* 8. TYPE CASTING AND ERROR HANDLING */
    if(sequence.empty()) {
        cerr << "Empty sequence provided!" << endl;
        return 0.0;
    }
    
    return (static_cast<double>(gc) / sequence.length()) * 100.0;
}


g++ -std=c++11 bioinformatics_pipeline_tutorial.cpp -o dna_analyzer
./dna_analyzer
