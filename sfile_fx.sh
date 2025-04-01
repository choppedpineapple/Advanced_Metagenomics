# Collect results
for _, remaining_strings in results:
    failed.extend(remaining_strings)

# Add this line to ensure deterministic ordering
failed.sort()  

# Update remaining strings for next iteration
remain_strings = failed.copy()
