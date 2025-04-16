
---

# Outline

1. **Hello World & Compilation**
2. **Variables, Data Types, and Input/Output**
3. **Control Structures (if, loops)**
4. **Functions**
5. **Vectors and Basic STL**
6. **File I/O (Reading FASTA files)**
7. **Structs and Classes (Basic OOP)**
8. **Simple Pipeline: GC Content Calculator**

---

## 1. Hello World & Compilation

```cpp
#include <iostream> // Include input/output stream library

// Main function: Entry point of every C++ program
int main() {
    // Print "Hello, Bioinformatics!" to the console
    std::cout << "Hello, Bioinformatics!" << std::endl;
    return 0; // Return 0 signals successful execution
}
```
**How to compile and run:**
```sh
g++ hello.cpp -o hello
./hello
```
- `#include <iostream>`: Adds standard input/output functionality.
- `int main() {}`: C++ programs start execution from `main`.
- `std::cout` prints to the console.

---

## 2. Variables, Data Types, and Input/Output

```cpp
#include <iostream>

int main() {
    int sequenceLength = 0; // Integer variable declaration
    double gcContent = 0.0; // Floating-point variable

    std::cout << "Enter sequence length: ";
    std::cin >> sequenceLength; // User input

    std::cout << "Enter GC content (%): ";
    std::cin >> gcContent;

    std::cout << "Sequence information:\n";
    std::cout << "Length: " << sequenceLength << "\n";
    std::cout << "GC Content: " << gcContent << "%\n";

    return 0;
}
```
- `int`, `double`: Data types.
- `std::cin`: Reads user input.
- `<<` and `>>`: Output and input operators.

---

## 3. Control Structures (If Statements, Loops)

```cpp
#include <iostream>

int main() {
    int n;
    std::cout << "How many sequences? ";
    std::cin >> n;

    // For loop to process multiple sequences
    for (int i = 1; i <= n; ++i) {
        std::string seq;
        std::cout << "Enter sequence #" << i << ": ";
        std::cin >> seq;

        // If statement to check length
        if (seq.length() < 5) {
            std::cout << "Sequence too short!\n";
        } else {
            std::cout << "Valid sequence: " << seq << "\n";
        }
    }
    return 0;
}
```
- `for` loop: Repeats code block.
- `if` statement: Decision-making.

---

## 4. Functions (Reusable Code)

```cpp
#include <iostream>
#include <string>

// Function to calculate GC content
double calcGCContent(const std::string& seq) {
    int gcCount = 0;
    for (char base : seq) {
        if (base == 'G' || base == 'C' || base == 'g' || base == 'c') {
            gcCount++;
        }
    }
    // Calculate percentage
    return 100.0 * gcCount / seq.length();
}

int main() {
    std::string sequence;
    std::cout << "Enter a DNA sequence: ";
    std::cin >> sequence;

    double gc = calcGCContent(sequence);

    std::cout << "GC Content: " << gc << "%\n";
    return 0;
}
```
- Functions allow code reuse.
- `const std::string& seq`: Passes the sequence by reference, avoids copying, and doesn’t modify it.

---

## 5. Vectors and Basic STL (Standard Template Library)

```cpp
#include <iostream>
#include <vector>
#include <string>

int main() {
    std::vector<std::string> sequences; // Dynamic array of strings
    int n;
    std::cout << "How many sequences? ";
    std::cin >> n;

    // Read n sequences into the vector
    for (int i = 0; i < n; ++i) {
        std::string seq;
        std::cout << "Enter sequence #" << (i+1) << ": ";
        std::cin >> seq;
        sequences.push_back(seq); // Add sequence to the vector
    }

    // Print all sequences
    std::cout << "You entered:\n";
    for (const std::string& seq : sequences) {
        std::cout << seq << "\n";
    }
    return 0;
}
```
- `std::vector`: Dynamic array, grows as needed.
- `push_back()`: Adds element to vector.
- Range-based for loop: `for (const auto& item : container)`

---

## 6. File I/O: Reading a FASTA File

```cpp
#include <iostream>
#include <fstream>
#include <string>

int main() {
    std::ifstream infile("example.fasta"); // Open FASTA file for reading
    if (!infile) {
        std::cerr << "Cannot open file.\n";
        return 1;
    }

    std::string line;
    std::string sequence = "";
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        if (line[0] == '>') {
            // Header line in FASTA
            std::cout << "Header: " << line << "\n";
            if (!sequence.empty()) {
                std::cout << "Sequence: " << sequence << "\n";
                sequence = "";
            }
        } else {
            sequence += line; // Append sequence lines
        }
    }
    // Print last sequence
    if (!sequence.empty()) {
        std::cout << "Sequence: " << sequence << "\n";
    }
    infile.close();
    return 0;
}
```
- `ifstream`: Input file stream.
- `getline`: Reads a line from the file.
- FASTA format: `>` header, followed by sequence.

---

## 7. Structs and Classes (OOP)

```cpp
#include <iostream>
#include <vector>
#include <string>

// Struct for a FASTA sequence
struct FastaSequence {
    std::string header;
    std::string sequence;
};

int main() {
    FastaSequence fs;
    fs.header = ">example";
    fs.sequence = "ATGCGTACG";

    std::cout << "Header: " << fs.header << "\n";
    std::cout << "Sequence: " << fs.sequence << "\n";
    return 0;
}
```
- `struct`: Groups related data.
- `class` is similar but members are private by default.

---

## 8. Simple Bioinformatics Pipeline: GC Content for All Sequences in a FASTA File

```cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// Struct to store each FASTA sequence
struct FastaSequence {
    std::string header;
    std::string sequence;
};

// Function to calculate GC content
double calcGCContent(const std::string& seq) {
    int gcCount = 0;
    for (char base : seq) {
        if (base == 'G' || base == 'C' || base == 'g' || base == 'c') {
            gcCount++;
        }
    }
    if (seq.length() == 0) return 0;
    return 100.0 * gcCount / seq.length();
}

// Function to read FASTA file and return a vector of FastaSequence
std::vector<FastaSequence> readFasta(const std::string& filename) {
    std::ifstream infile(filename);
    std::vector<FastaSequence> sequences;
    if (!infile) {
        std::cerr << "Cannot open file.\n";
        return sequences;
    }
    std::string line, currentHeader, currentSeq;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        if (line[0] == '>') {
            if (!currentHeader.empty()) {
                sequences.push_back({currentHeader, currentSeq});
                currentSeq = "";
            }
            currentHeader = line;
        } else {
            currentSeq += line;
        }
    }
    if (!currentHeader.empty()) {
        sequences.push_back({currentHeader, currentSeq});
    }
    return sequences;
}

int main() {
    std::string filename;
    std::cout << "Enter FASTA filename: ";
    std::cin >> filename;

    std::vector<FastaSequence> sequences = readFasta(filename);

    for (const FastaSequence& fs : sequences) {
        double gc = calcGCContent(fs.sequence);
        std::cout << fs.header << "\n";
        std::cout << "GC Content: " << gc << "%\n\n";
    }

    return 0;
}
```

---

# **How to Use This Pipeline**

1. **Copy each code section** into its own file, e.g., `step1.cpp`, `step2.cpp`, etc.
2. **Compile** using `g++ stepN.cpp -o stepN`, replacing N with the step number.
3. **Run** the compiled program: `./stepN`.
4. **Experiment:** Change variables, add print statements, or modify logic to see how things work.
5. **Progress:** Once confident, move to the next step.

---

## **Next Steps in Learning**

- Explore **STL containers** (`map`, `set`, etc.).
- Learn about **pointers** and **dynamic memory** (advanced).
- Practice writing **parsers** for other formats (e.g., FASTQ).
- Try adding **error handling**, **multi-threading**, or connecting C++ code to **Python** (via bindings like pybind11).

---

## **Summary**

This pipeline teaches you:
- How to structure C++ code
- How to use variables, loops, functions, and basic OOP
- How to read/write files and process bioinformatics formats
- How to build up from simple programs to real-world tasks

**Pro tip:** Always read and experiment with the code. The comments will guide your learning and build your confidence in C++ for bioinformatics!

---

**If you want a specific step explained in even greater detail or want to add a new bioinformatics feature, just ask!**
