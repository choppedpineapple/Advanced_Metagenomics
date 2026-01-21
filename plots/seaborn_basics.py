import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ===== 1. READ THE DATA =====
# Read TSV file with no headers
# We assign column names manually since there are no headers in the file
df = pd.read_csv('shannon_diversity.tsv', 
                 sep='\t',              # Tab-separated
                 header=None,           # No header row
                 names=['Sample', 'Shannon_Index'])  # Assign column names

print(f"Loaded {len(df)} samples")
print(df.head())  # Preview first few rows

# ===== 2. IDENTIFY OUTLIERS (OPTIONAL) =====
# Calculate statistics to understand the data distribution
# This helps us identify Zymo controls or other outliers
q1 = df['Shannon_Index'].quantile(0.25)  # 25th percentile
q3 = df['Shannon_Index'].quantile(0.75)  # 75th percentile
iqr = q3 - q1  # Interquartile range
median = df['Shannon_Index'].median()

# Typical Shannon values are 0-5, so we'll focus the plot on this range
# But let's see what's in our data
print(f"
Data statistics:")
print(f"Min: {df['Shannon_Index'].min():.2f}")
print(f"Max: {df['Shannon_Index'].max():.2f}")
print(f"Median: {median:.2f}")
print(f"Q1-Q3: {q1:.2f} - {q3:.2f}")

# Identify potential outliers (Zymo controls)
# Standard definition: outliers are beyond 1.5 * IQR from Q1/Q3
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = df[(df['Shannon_Index'] < lower_bound) | 
              (df['Shannon_Index'] > upper_bound)]
print(f"
Potential outliers/controls: {len(outliers)}")
if len(outliers) > 0:
    print(outliers)

# ===== 3. SET UP SEABORN STYLE =====
# Seaborn has built-in themes that make plots look professional
# Popular styles: 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'
sns.set_style('whitegrid')  # Clean style with grid lines

# Set color palette (optional)
# Try: 'deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind'
sns.set_palette('muted')

# ===== 4. CREATE THE PLOT =====
# Option A: Bar plot (shows each sample as a bar)
# Good for: comparing individual samples, fewer samples (<30)
fig, ax = plt.subplots(figsize=(12, 6))  # Set figure size

# Create bar plot
# x = categorical variable (sample names)
# y = numerical variable (Shannon index values)
# data = the dataframe
# ax = which axis to plot on
sns.barplot(data=df, 
            x='Sample', 
            y='Shannon_Index',
            color='steelblue',  # Single color for all bars
            ax=ax)

# Rotate x-axis labels for readability (sample names can be long)
plt.xticks(rotation=45, ha='right')  # ha = horizontal alignment

# Set y-axis limits to focus on normal range (0-5 for Shannon index)
# This will "zoom in" on the typical values, hiding extreme outliers
# Adjust these values based on your data
ax.set_ylim(0, 5.5)  # Leave a bit of space above max typical value

# Add labels and title
ax.set_xlabel('Sample Name', fontsize=12)
ax.set_ylabel('Shannon Diversity Index', fontsize=12)
ax.set_title('Shannon Diversity Index Across Samples', 
             fontsize=14, fontweight='bold')

# Add a horizontal line at median (optional - helps see trends)
ax.axhline(y=median, color='red', linestyle='--', 
           linewidth=1, alpha=0.7, label=f'Median ({median:.2f})')
ax.legend()

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
plt.savefig('shannon_diversity_barplot.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== 5. ALTERNATIVE PLOT TYPES =====
# Option B: Strip plot (shows each sample as a point)
# Good for: seeing distribution, many samples, identifying outliers
fig, ax = plt.subplots(figsize=(10, 6))

sns.stripplot(data=df, 
              y='Shannon_Index',  # Note: only y, no x
              color='steelblue',
              size=8,
              alpha=0.7,  # Transparency
              jitter=True,  # Adds random horizontal spread
              ax=ax)

ax.set_ylim(0, 5.5)
ax.set_ylabel('Shannon Diversity Index', fontsize=12)
ax.set_title('Shannon Diversity Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shannon_diversity_stripplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Option C: Horizontal bar plot (better for many samples with long names)
fig, ax = plt.subplots(figsize=(8, len(df) * 0.3))  # Height scales with samples

sns.barplot(data=df,
            x='Shannon_Index',  # Note: x and y are swapped
            y='Sample',
            color='steelblue',
            ax=ax)

ax.set_xlim(0, 5.5)
ax.set_xlabel('Shannon Diversity Index', fontsize=12)
ax.set_ylabel('Sample Name', fontsize=12)
ax.set_title('Shannon Diversity Index', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shannon_diversity_horizontal.png', dpi=300, bbox_inches='tight')
plt.show()

print("
Plots saved successfully!")
