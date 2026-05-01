import pandas as pd
from statsmodels.tsa.api import VAR
import warnings
import os
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels.tsa.base.tsa_model")

# --- DYNAMIC PATH SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(script_dir), "data")
# Ensure the folder exists before saving anything
os.makedirs(data_dir, exist_ok=True)

# Load the TURKISH data from the parallel data folder
input_path = os.path.join(data_dir, "var_input_tr.csv")
df = pd.read_csv(input_path, index_col='date_publish', parse_dates=True)

# Apply Differencing 
df_diff = df.diff().dropna()
print("Data differenced. Modeling daily 'shocks' for Turkish data.")

# List of grievance columns
grievance_columns = [
    'narrative_gov', 'narrative_dem_reform', 'narrative_global', 
    'narrative_religion', 'narrative_elections', 'narrative_basic_needs', 
    'narrative_coup', 'narrative_violence'
]

protest_starts = [
    pd.Timestamp("2017-06-15"),
    pd.Timestamp("2024-10-31"),
    pd.Timestamp("2025-03-19"),
]

max_lag = 28

# extract pre-protest window for each episode and align by lag
all_windows = []

for start in protest_starts:
    window = df_diff[
        (df_diff.index >= start - pd.Timedelta(days=max_lag)) &
        (df_diff.index <= start)
    ][grievance_columns].copy()
    
    # assign lag labels so 0 = protest day and 28 = 28 days before
    window = window.iloc[::-1].reset_index(drop=True)  #index 0 is protest day
    window.index.name = 'lag'
    all_windows.append(window)

#averaging
avg_window = pd.concat(all_windows).groupby(level=0).mean()

fig, ax = plt.subplots(figsize=(12, 5))
colors = plt.cm.tab10.colors

for i, grievance in enumerate(grievance_columns):
    label = grievance.replace("narrative_", "").upper()
    ax.plot(avg_window.index, avg_window[grievance], marker='o', label=label,
            color=colors[i % len(colors)], linewidth=1.8, markersize=4)

ax.axvline(0, color='red', linestyle='--', linewidth=1, label='Protest Start')
ax.set_xlabel("Days before protest (0 = protest start)")
ax.set_ylabel("Avg differenced narrative score")
ax.set_title("Average Grievance Trajectory Across 3 Pre-Protest Windows (TR)")
ax.invert_xaxis() 
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
plt.tight_layout()

# SAVE PATH UPDATED
plt.savefig(os.path.join(data_dir, "avg_preprotest_trajectory_tr.png"), dpi=150)
plt.show()

# Initialize the VAR Model
model = VAR(df_diff)

# AUTOMATED CAUSALITY TESTING (7 & 14 Lags)
lags_to_test = [7, 14]

for lag in lags_to_test:
    results = model.fit(lag)
    
    print(f"\n========== TURKISH GRANGER CAUSALITY RESULTS ({lag}-DAY LAG) ==========")
    for grievance in grievance_columns:
        if grievance in df_diff.columns:
            test_result = results.test_causality('narrative_protest_outcome', [grievance], kind='f')
            p_val = test_result.pvalue
            significance = "⭐ SIGNIFICANT" if p_val < 0.05 else "Not Significant"
            clean_name = grievance.replace("narrative_", "").upper()
            print(f"{clean_name.ljust(15)} -> PROTESTS : p-value = {p_val:.4f} | {significance}")

# Temporal Pearson logic...
records = []
pearson_lag = 14 

for start in protest_starts:
    window = df_diff[
        (df_diff.index >= start - pd.Timedelta(days=28)) &
        (df_diff.index <= start)
    ][grievance_columns].copy()
    
    for grievance in grievance_columns:
        if grievance in window.columns:
            for lag in range(1, pearson_lag + 1):
                shifted = window[grievance].shift(lag)
                protest_binary = (window.index == start).astype(int)
                aligned = pd.concat([shifted, pd.Series(protest_binary, index=window.index)], axis=1).dropna()

                if len(aligned) > 0:
                    corr, p_val = pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                    records.append({
                        'grievance': grievance, 'lag': lag,
                        'protest': str(start.date()), 'correlation': corr, 'p_value': p_val
                    })

results_df = pd.DataFrame(records)
avg_results = results_df.groupby(['grievance', 'lag'])[['correlation', 'p_value']].mean().reset_index()

# SAVE PATH UPDATED
output_path = os.path.join(data_dir, "lagged_correlations_tr.csv")
results_df.to_csv(output_path, index=False)
print(f"Saved lagged correlations to: {output_path}")

# Heatmap Section
pivot_corr = avg_results.pivot(index='grievance', columns='lag', values='correlation')
pivot_pval = avg_results.pivot(index='grievance', columns='lag', values='p_value')
pivot_corr.index = [i.replace("narrative_", "").upper() for i in pivot_corr.index]
pivot_pval.index = pivot_corr.index

fig, ax = plt.subplots(figsize=(14, 6))
cmap = plt.cm.RdYlGn 
norm = mcolors.TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)
im = ax.imshow(pivot_corr.values, aspect='auto', cmap=cmap, norm=norm)

ax.set_xticks(range(len(pivot_corr.columns)))
ax.set_xticklabels([f"Lag {c}" for c in pivot_corr.columns], rotation=45, ha='right')
ax.set_yticks(range(len(pivot_corr.index)))
ax.set_yticklabels(pivot_corr.index)

for i in range(pivot_corr.shape[0]):
    for j in range(pivot_corr.shape[1]):
        r, p = pivot_corr.values[i, j], pivot_pval.values[i, j]
        txt = f"{r:.2f}{'*' if p < 0.05 else ''}"
        ax.text(j, i, txt, ha='center', va='center', fontsize=7, color='black' if abs(r) < 0.2 else 'white')

plt.colorbar(im, ax=ax, label='Pearson r')
ax.set_title("Lagged Correlation of Grievance Narratives - Protests (tr)\n* = p < 0.05")
plt.tight_layout()

# SAVE PATH UPDATED
plt.savefig(os.path.join(data_dir, "heatmap_lagged_corr_tr.png"), dpi=150)
plt.show()

# Final Plot Section
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, (protest_date, group) in zip(axes, results_df.groupby('protest')):
    for i, (grievance, grp) in enumerate(group.groupby('grievance')):
        grp = grp.sort_values('lag')
        label = grievance.replace("narrative_", "").upper()
        ax.plot(grp['lag'], grp['correlation'], marker='o', label=label, linewidth=1.8, markersize=4)

axes[0].set_ylabel("Pearson r")
plt.tight_layout()

# SAVE PATH UPDATED
plt.savefig(os.path.join(data_dir, "lineplot_lagged_corr_per_protest_tr.png"), dpi=150)
plt.show()