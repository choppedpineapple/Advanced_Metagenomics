# Full Script: Simple Boxplot for F/B Ratios with High-Value Sidebar
# Run this in Jupyter or as a .py file – assumes 'fb_ratios.tsv' in your dir

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv('fb_ratios.tsv', sep='\t', header=None, names=['sample', 'value'])
print(f"Loaded {len(df)} samples.")
print(df.head())
print(df['value'].describe())

# 2. Filter High Values
cutoff = 10
high_samples = df[df['value'] > cutoff][['sample', 'value']].copy()
if not high_samples.empty:
    high_list = "High F/B (>10):\n" + "\n".join(
        [f"{row['sample']}: {row['value']:.1f}" for _, row in high_samples.iterrows()]
    )
    num_high = len(high_samples)
else:
    high_list = "No samples above cutoff."
    num_high = 0
print(f"Found {num_high} samples above {cutoff}.")
if num_high > 0:
    print(high_samples)

# 3. Create and Style Plot
n_samples = len(df)
fig_width = max(6, min(12, 4 + 0.02 * n_samples))
fig_height = 6
label_size = max(8, min(12, 14 - 0.01 * n_samples))
title_size = label_size + 2

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
box_plot = ax.boxplot(df['value'], vert=True, patch_artist=False, labels=['All Samples'])
ax.set_ylabel('F/B Ratio', fontsize=label_size)
ax.set_title('F/B Ratio Distribution Across Metagenomic Samples', fontsize=title_size)
ax.tick_params(axis='both', labelsize=label_size - 1)

# 4. Add Sidebar Text
ax.text(1.05, 0.95, high_list, transform=ax.transAxes, fontsize=label_size - 1, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5, pad=0.3))

# 5. Save
plt.tight_layout()
output_file = 'fb_ratio_boxplot.pdf'
plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
print(f"Plot saved as '{output_file}' – Ready for your next lab meeting!")
# plt.show()  # Uncomment for on-screen preview
