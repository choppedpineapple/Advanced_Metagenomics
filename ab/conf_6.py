def is_boundary_wobble(seq1, seq2):
    # Sort by length: s1 is shorter, s2 is longer
    s1, s2 = sorted([seq1, seq2], key=len)
    
    # Check if the length difference is exactly 1
    if len(s2) - len(s1) != 1:
        return False
        
    # Check if it's an extra amino acid at the C-terminus (end)
    if s2.startswith(s1):
        return True
        
    # Check if it's an extra amino acid at the N-terminus (beginning)
    if s2.endswith(s1):
        return True
        
    return False

# Testing your mess:
print(is_boundary_wobble("GAYEJEJ", "GAYEJEJW"))      # True (IgBLAST's fault)
print(is_boundary_wobble("WTHSJR", "WTHKJR"))        # False (Real biology)
print(is_boundary_wobble("EYEJDIROE", "EYWEJDIROE"))  # False (Real biology)
