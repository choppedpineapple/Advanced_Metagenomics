from collections import Counter
import pandas as pd

# Step 1: Your list of DNA sequences
sequences = [
    'ATGCN',
    'ATGNN',
    'ATGNC',
    'ATGNN'
]

# Step 2: Initialize an empty list to hold the count data
data = []

# Step 3: Loop through each position in the sequences
for position in range(len(sequences[0])):  # assumes all sequences have the same length
    # Collect all the bases at this position
    bases_at_position = [seq[position] for seq in sequences]
    
    # Count how many times each base appears at this position
    counts = Counter(bases_at_position)
    
    # Create a dictionary with all bases, even if a base doesn't appear at all
    row = {base: counts.get(base, 0) for base in 'ATGCN'}
    data.append(row)

# Step 4: Convert the list of dictionaries to a pandas DataFrame
df = pd.DataFrame(data)

# Step 5: Optional - Add position index for clarity
df.index.name = 'Position'

# Step 6: Print the result
print(df)
