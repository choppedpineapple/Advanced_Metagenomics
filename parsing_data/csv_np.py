import numpy as np

# Create sample data and save to CSV
data = np.array([
    [1, 25, 50000, 1],
    [2, 30, 60000, 0],
    [3, 35, 75000, 1], 
    [4, 28, 45000, 0],
    [5, 42, 85000, 1]
])

# Save to CSV with headers
header = "id,age,salary,promoted"
np.savetxt('employees.csv', data, delimiter=',', header=header, 
           comments='', fmt='%d')

print("Created employees.csv")

# Read CSV back
# Skip header row and convert to appropriate data types
csv_data = np.loadtxt('employees.csv', delimiter=',', skiprows=1)
print("\nLoaded data:")
print(csv_data)

# Basic manipulations
print(f"\nData shape: {csv_data.shape}")
print(f"Number of employees: {len(csv_data)}")

# Extract specific columns (0-indexed)
ages = csv_data[:, 1]  # Age column
salaries = csv_data[:, 2]  # Salary column

print(f"\nAverage age: {np.mean(ages):.1f}")
print(f"Average salary: ${np.mean(salaries):,.0f}")
print(f"Max salary: ${np.max(salaries):,.0f}")

# Filter data - employees over 30
over_30 = csv_data[csv_data[:, 1] > 30]
print(f"\nEmployees over 30:")
print(over_30)

# Add a new column (bonus = salary * 0.1)
bonuses = csv_data[:, 2] * 0.1
enhanced_data = np.column_stack([csv_data, bonuses])
print(f"\nData with bonuses:")
print(enhanced_data)

# Save modified data
new_header = "id,age,salary,promoted,bonus"
np.savetxt('employees_with_bonus.csv', enhanced_data, delimiter=',', 
           header=new_header, comments='', fmt='%.0f')

print("\nSaved enhanced data to employees_with_bonus.csv")

# Read with mixed data types using structured arrays
dtype = [('id', 'i4'), ('age', 'i4'), ('salary', 'i4'), ('promoted', 'i4')]
structured_data = np.loadtxt('employees.csv', delimiter=',', skiprows=1, dtype=dtype)

print(f"\nStructured array - accessing by field name:")
print(f"All ages: {structured_data['age']}")
print(f"Promoted employees: {structured_data[structured_data['promoted'] == 1]['id']}")
