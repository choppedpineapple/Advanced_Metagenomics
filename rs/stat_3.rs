use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let filename = "sequences.fasta";
    let min_length: usize = 100;

    let file = match File::open(filename) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Unable to open file {}: {}", filename, e);
            std::process::exit(1);
        }
    };

    let reader = BufReader::new(file);

    let sentinel = std::iter::once(Ok(String::from(">")));
    let lines = reader.lines().chain(sentinel);

    let mut header = String::new();
    let mut sequence = String::new();

    for line in lines {
        let line = line.expect("Failed to read line");

        if line.starts_with('>') {
            if !sequence.is_empty() && sequence.len() >= min_length {
                println!("{}", header);
                println!("{}", sequence);
            }
            sequence.clear();
            header = line;
        } else {
            sequence.push_str(&line);
        }
    }
}
