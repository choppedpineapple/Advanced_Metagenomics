#include <iostream>
#include <fstream>
#include <string>

// helper to print usage and exit
void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <in.fastq> <out.fastq> <min_length>\n";
    std::exit(1);
}

int main(int argc, char* argv[]) {
    if (argc != 4) usage(argv[0]);
    std::ifstream fin { argv[1] };
    std::ofstream fout{ argv[2] };
    int min_len = std::stoi(argv[3]);

    if (!fin || !fout) {
        std::cerr << "Error opening files.\n";
        return 1;
    }
    std::string header, seq, plus, qual;
    while (std::getline(fin, header) &&
           std::getline(fin, seq)    &&
           std::getline(fin, plus)   &&
           std::getline(fin, qual)) {
        if ((int)seq.size() >= min_len) {
            fout << header << '\n'
                 << seq    << '\n'
                 << plus   << '\n'
                 << qual   << '\n';
        }
    }
    return 0;
}
