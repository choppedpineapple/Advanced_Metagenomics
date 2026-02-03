### Block 1: Installation and Import
Standard convention is to import it as `np`.

```python
# Run in terminal: pip install numpy

import numpy as np

print("NumPy version:", np.__version__)
```

### Block 2: Creating Arrays
The core object is the `ndarray` (N-dimensional array). You can create them from Python lists or use NumPy's built-in generators.

```python
# 1. Creating from a standard Python list
arr_list = np.array([1, 2, 3, 4, 5])

# 2. Creating arrays with built-in methods
# Zeros: Filled with 0s
zeros = np.zeros((2, 3)) 

# Ones: Filled with 1s
ones = np.ones((3, 3))

# Arange: Like range(), but returns an array
sequence = np.arange(0, 10, 2) # Start, Stop (exclusive), Step

# Random: Random floats between 0 and 1
random_arr = np.random.rand(2, 2)

print("From list:", arr_list)
print("\nZeros (2x3):\n", zeros)
print("\nSequence (0 to 8 step 2):", sequence)
```

### Block 3: Inspecting Array Attributes
Before doing math, you must know the *shape* (dimensions) and *data type* of your array. This solves most errors.

```python
matrix = np.array([[1, 2, 3], 
                   [4, 5, 6]])

# ndim: Number of dimensions (1D, 2D, 3D...)
print("Dimensions:", matrix.ndim) # Output: 2

# shape: Tuple of (rows, columns)
print("Shape:", matrix.shape)     # Output: (2, 3)

# size: Total number of elements
print("Size:", matrix.size)       # Output: 6

# dtype: The type of data (int32, float64, etc.)
print("Data type:", matrix.dtype) # Output: int64 (or int32)
```

### Block 4: Indexing and Slicing
Accessing specific elements. Works like Python lists but extends to 2D (matrices).

```python
data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

# 1D Slicing: [start:stop:step]
subset = data[1:5] # Get index 1 to 4
print("Slice:", subset)

# 2D Matrix Slicing
matrix = np.array([[1, 2, 3], 
                   [4, 5, 6], 
                   [7, 8, 9]])

# matrix[row, col]
element = matrix[0, 2] # Row 0, Col 2 -> Output: 3
print("Specific element:", element)

# Get the first two rows and the first two columns
sub_matrix = matrix[:2, :2]
print("\nTop-left 2x2 submatrix:\n", sub_matrix)

# IMPORTANT: NumPy slices are "views", not copies.
# Modifying 'sub_matrix' will change 'matrix'.
```

### Block 5: Boolean Masking (Advanced Filtering)
This is a powerful feature. You create a "mask" of True/False values to filter data without writing loops.

```python
ages = np.array([18, 22, 15, 30, 12, 40])

# 1. Create a condition (returns True/False array)
adults_mask = ages >= 18
print("Mask:", adults_mask)

# 2. Apply the mask to the array (returns only values where True)
adults_only = ages[adults_mask]
print("Adults only:", adults_only)

# You can do this in one line
print("Young people:", ages[ages < 18])
```

### Block 6: Reshaping and Transposing
Data often needs to be reorganized to fit into machine learning models (e.g., turning a list into a column vector).

```python
arr = np.arange(1, 13) # Numbers 1 to 12
print("Original (1D):", arr)

# Reshape to 3 rows, 4 columns (must match total size)
reshaped = arr.reshape(3, 4)
print("\nReshaped to (3,4):\n", reshaped)

# Flatten: Convert multi-dimensional back to 1D
flattened = reshaped.flatten()
print("\nFlattened:", flattened)

# Transpose: Swap rows and columns (columns become rows)
transposed = reshaped.T
print("\nTransposed shape:", transposed.shape) # (4, 3)
```

### Block 7: Array Mathematics (Element-wise)
NumPy allows you to do math on entire arrays at once (Vectorization), which is much faster than loops.

```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

# Basic Arithmetic
print("Addition:", a + b)      # [11, 22, 33]
print("Multiplication:", a * b) # [10, 40, 90]
print("Scalar Multiply:", a * 5) # [5, 10, 15]

# Universal Functions (Ufuncs) apply math element-wise
# Square root
print("Square root:", np.sqrt(a))

# Exponents (e^x)
print("Exp:", np.exp(a))
```

### Block 8: Broadcasting
This is what makes NumPy magical. It allows you to perform arithmetic on arrays of different shapes.

```python
# Scenario: Add 5 to every element in the array
# NumPy automatically "stretches" the scalar 5 to match the array shape.
arr = np.array([1, 2, 3])
result = arr + 5
print("Broadcasting scalar:", result)

# Scenario: Adding a (3,1) matrix to a (3,) array
matrix = np.array([[1], [2], [3]]) # Shape (3, 1)
vector = np.array([10, 20, 30])    # Shape (3,)

# NumPy broadcasts the vector across the rows
result = matrix + vector
print("\nMatrix + Vector Broadcasting:\n", result)
```

### Block 9: Aggregation and Statistics
Calculating summary statistics across axes.

```python
data = np.array([[10, 20, 30],
                 [40, 50, 60],
                 [70, 80, 90]])

# Sum of ALL elements
total_sum = np.sum(data)
print("Total Sum:", total_sum)

# Min and Max
print("Min:", np.min(data))
print("Max:", np.max(data))

# Mean (Average)
print("Mean:", np.mean(data))

# --- AXIS ARGUMENT IS CRUCIAL ---
# axis=0: Operation down the rows (collapse rows, get column stats)
# axis=1: Operation across columns (collapse columns, get row stats)

col_sum = np.sum(data, axis=0) 
print("\nSum of columns (axis=0):", col_sum) # [120, 150, 180]

row_sum = np.sum(data, axis=1)
print("Sum of rows (axis=1):", row_sum)       # [60, 150, 240]
```

### Block 10: Linear Algebra (Dot Product)
Essential for machine learning, physics, and simulations.

```python
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])

# Dot Product (1*4 + 2*5 + 3*6)
dot_product = np.dot(vec1, vec2)
print("Dot Product:", dot_product) # Output: 32

# Matrix Multiplication (@ operator is the modern standard)
mat_a = np.array([[1, 2], [3, 4]])
mat_b = np.array([[5, 6], [7, 8]])

mat_mult = mat_a @ mat_b 
print("\nMatrix Multiplication:\n", mat_mult)
```

### Block 11: Saving and Loading
You can save arrays to disk to process them later without recomputing.

```python
arr = np.array([1, 2, 3, 4, 5])

# Save to .npy file (NumPy specific format, fast)
np.save('my_array.npy', arr)

# Load from .npy file
loaded_arr = np.load('my_array.npy')

print("Loaded Array:", loaded_arr)
```

### Block 12: Putting it together (Simulation)
Let's simulate a dataset of 100 students with random scores, filter them, and calculate statistics.

```python
# 1. Generate Data: 100 students, 3 subjects (Math, Science, English)
# Random integers between 0 and 100
scores = np.random.randint(0, 101, size=(100, 3)) 

# 2. Calculate average score per student (Axis 1)
student_averages = np.mean(scores, axis=1)

# 3. Filter: Who passed? (Average >= 50)
passing_mask = student_averages >= 50
passing_students_count = np.sum(passing_mask)

# 4. What is the highest average score?
highest_avg = np.max(student_averages)

print(f"Total shape: {scores.shape}")
print(f"Number of students passing: {passing_students_count}")
print(f"Highest class average: {highest_avg:.2f}")
```
