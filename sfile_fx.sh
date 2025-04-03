# Collect results
for _, remaining_strings in results:
    failed.extend(remaining_strings)

# Add this line to ensure deterministic ordering
failed.sort()  

# Update remaining strings for next iteration
remain_strings = failed.copy()




-------

# In the main section of your script, change this line:
file_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.txt')]

# To this:
file_list = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.txt')])


