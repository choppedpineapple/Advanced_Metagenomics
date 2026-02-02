### Block 1: Installation and Import
Before doing anything, you need pandas. It is standard convention to import it as `pd`.

```python
# Run this in your terminal if you haven't installed it yet:
# pip install pandas

import pandas as pd
import numpy as np  # Often used with pandas

print("Pandas version:", pd.__version__)
```

### Block 2: Creating a DataFrame (The Core Object)
The **DataFrame** is the main data structure in pandas (like a table in Excel). Let's create one from a dictionary to practice.

```python
# Creating a dictionary of data
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 40, 45],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'Salary': [70000, 80000, 120000, 90000, 85000]
}

# Convert dictionary to a DataFrame
df = pd.DataFrame(data)

# Display the dataframe
print(df)
```

### Block 3: Reading and Writing Data (I/O)
In the real world, you rarely type data manually. You load it from CSV or Excel files.

```python
# SAVING the dataframe we made above to a CSV file
df.to_csv('employees.csv', index=False) 

# LOADING data from a CSV file
loaded_df = pd.read_csv('employees.csv')

# Note: For Excel files, use pd.read_excel('file.xlsx')
print("Data loaded successfully:\n", loaded_df.head())
```

### Block 4: Inspecting the Data
The first thing you do with a new dataset is understand its size and the types of data it contains.

```python
# 1. Check the first 5 rows
print("--- First 5 Rows ---")
print(df.head())

# 2. Check the last 3 rows
print("\n--- Last 3 Rows ---")
print(df.tail(3))

# 3. Get a concise summary (columns, data types, non-null counts)
print("\n--- Info ---")
print(df.info())

# 4. Get statistical summary of numeric columns
print("\n--- Describe ---")
print(df.describe())
```

### Block 5: Selecting Columns
There are two main ways to grab a specific column: Dot notation (easiest) and Bracket notation (safer).

```python
# Method 1: Dot notation (Good for simple column names without spaces)
names = df.Name
print(names) 

# Method 2: Bracket notation (Required if column name has spaces)
salaries = df['Salary']
print(salaries)

# Selecting MULTIPLE columns (Must use bracket notation with a list)
subset = df[['Name', 'City']]
print("\n--- Subset (Name and City) ---")
print(subset)
```

### Block 6: Selecting Rows (`loc` vs `iloc`)
This is often the most confusing part for beginners.
*   **`.iloc`**: Index-based (Use integer numbers like 0, 1, 2).
*   **`.loc`**: Label-based (Use the names in the index).

```python
# iloc: Get the first row (index 0)
first_row = df.iloc[0]
print("--- First Row (iloc) ---")
print(first_row)

# iloc: Get rows 0 to 2 (Exclusive of the end index, like Python lists)
rows_slice = df.iloc[0:3]
print("\n--- Rows 0, 1, 2 (iloc) ---")
print(rows_slice)

# loc: Get row where index label is 0 (usually same as iloc if index is default)
row_loc = df.loc[0]
print("\n--- Row with label 0 (loc) ---")
print(row_loc)
```

### Block 7: Filtering Data (Boolean Indexing)
You use conditional logic to select rows that meet specific criteria.

```python
# Find employees older than 30
older_than_30 = df[df['Age'] > 30]
print("--- Older than 30 ---")
print(older_than_30)

# Find employees in Chicago who earn less than 100k
# Note: Use '&' for 'and', '|' for 'or'
complex_filter = df[(df['City'] == 'Chicago') & (df['Salary'] < 100000)]
print("\n--- Chicago earning < 100k ---")
print(complex_filter)

# Filter using the .isin() method
specific_cities = df[df['City'].isin(['New York', 'Phoenix'])]
print("\n--- Only NY or Phoenix ---")
print(specific_cities)
```

### Block 8: Handling Missing Data
Real data is messy. You will often find `NaN` (Not a Number) values.

```python
# Let's introduce some missing values
df_with_nan = df.copy()
df_with_nan.loc[1, 'Salary'] = np.nan # Set Bob's salary to missing
df_with_nan.loc[3, 'City'] = np.nan   # Set David's city to missing

print("--- Data with Missing Values ---")
print(df_with_nan)

# 1. Drop rows with ANY missing values
clean_df = df_with_nan.dropna()
print("\n--- Drop Rows with NaN ---")
print(clean_df)

# 2. Fill missing values (Imputation)
# Fill missing salaries with the average salary
mean_salary = df_with_nan['Salary'].mean()
filled_df = df_with_nan.fillna({'Salary': mean_salary, 'City': 'Unknown'})
print("\n--- Filled NaN ---")
print(filled_df)
```

### Block 9: Adding and Modifying Columns
Data manipulation often involves creating new columns based on existing ones.

```python
# Create a new column 'Tax' which is 10% of Salary
df['Tax'] = df['Salary'] * 0.10
print(df)

# Create a column 'Net_Salary'
df['Net_Salary'] = df['Salary'] - df['Tax']
print("\n--- With Net Salary ---")
print(df)

# Modify an existing column (Give everyone a $5000 raise)
df['Salary'] = df['Salary'] + 5000
print("\n--- After Raise ---")
print(df)
```

### Block 10: Sorting
Ordering your data helps with analysis and presentation.

```python
# Sort by Age (Ascending)
sorted_age = df.sort_values(by='Age')

# Sort by Salary (Descending - highest first)
sorted_salary = df.sort_values(by='Salary', ascending=False)

print("--- Sorted by Salary (High to Low) ---")
print(sorted_salary[['Name', 'Salary']])
```

### Block 11: Grouping and Aggregation
This is the "SQL" part of pandas. You group data by categories and calculate stats (sum, mean, count).

```python
# Let's add some duplicate cities to demonstrate grouping better
df.loc[1, 'City'] = 'New York' 
df.loc[2, 'City'] = 'New York'

# Group by City and get the Average Salary
city_group = df.groupby('City')['Salary'].mean()

print("--- Average Salary per City ---")
print(city_group)

# Group by City and get multiple stats (Count and Mean)
city_stats = df.groupby('City')['Salary'].agg(['count', 'mean', 'max'])
print("\n--- Detailed City Stats ---")
print(city_stats)
```

### Block 12: Applying Functions (`apply`)
Sometimes built-in functions aren't enough, and you want to apply a custom function to every row or column.

```python
# Define a custom function
def categorize_age(age):
    if age < 30:
        return 'Young'
    elif age < 40:
        return 'Mid-Age'
    else:
        return 'Senior'

# Apply this function to the 'Age' column and create a new column
df['Age_Group'] = df['Age'].apply(categorize_age)

print("--- Data with Age Groups ---")
print(df[['Name', 'Age', 'Age_Group']])
```

### Block 13: Merging DataFrames
Combining data from different sources is a common task.

```python
# Create a second dataframe with bonus information
bonus_data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Bonus': [5000, 2000, 8000]
}
df_bonus = pd.DataFrame(bonus_data)

# Merge the original df with df_bonus based on the 'Name' column
# 'how=inner' means keep only names found in BOTH dataframes
merged_df = pd.merge(df, df_bonus, on='Name', how='left')

print("--- Merged DataFrame with Bonus ---")
print(merged_df)
```

### Block 14: Final Review (Putting it together)
Here is a quick snippet summarizing a standard analysis workflow.

```python
# 1. Load data
# 2. Clean data (dropna)
# 3. Filter data (Age > 25)
# 4. Group data (Get Average Net Salary by Age Group)

analysis_df = merged_df.copy()
result = analysis_df[analysis_df['Age'] > 25].groupby('Age_Group')['Net_Salary'].mean()

print("--- Final Analysis Result (Avg Net Salary by Age Group) ---")
print(result)
```
