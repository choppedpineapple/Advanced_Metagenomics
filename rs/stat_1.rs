use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    // Get the filename from command line arguments
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        println!("Usage: {} <filename>", args[0]);
        return;
    }
    
    let filename = &args[1];
    
    // Open the file
    let file = File::open(filename).expect("Failed to open file");
    let reader = BufReader::new(file);
    
    // Read lines and print the 4th one
    let lines: Vec<String> = reader.lines()
        .map(|l| l.expect("Failed to read line"))
        .collect();
    
    if lines.len() >= 4 {
        println!("{}", lines[3]);  // 4th line is at index 3
    } else {
        println!("File has fewer than 4 lines");
    }
}
