import pandas as pd
from statsmodels.tsa.api import VAR
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels.tsa.base.tsa_model")

# Load the TURKISH data
df = pd.read_csv(os.path.join("data", "var_input_tr.csv"), index_col='date_publish', parse_dates=True)

# Apply Differencing 
df_diff = df.diff().dropna()
print("Data differenced. Modeling daily 'shocks' for Turkish data.")

# Initialize the VAR Model
model = VAR(df_diff)

# List of grievance columns
grievance_columns = [
    'narrative_gov', 'narrative_dem_reform', 'narrative_global', 
    'narrative_religion', 'narrative_elections', 'narrative_basic_needs', 
    'narrative_coup', 'narrative_violence'
]

# AUTOMATED CAUSALITY TESTING (7 & 14 Lags)
lags_to_test = [7, 14]

for lag in lags_to_test:
    results = model.fit(lag)
    
    print(f"\n========== TURKISH GRANGER CAUSALITY RESULTS ({lag}-DAY LAG) ==========")
    print("Null Hypothesis: The Grievance does NOT cause Protests")
    print("---------------------------------------------------------------")

    for grievance in grievance_columns:
        if grievance in df_diff.columns:
            test_result = results.test_causality('narrative_protest_outcome', [grievance], kind='f')
            p_val = test_result.pvalue
            
            significance = "⭐ SIGNIFICANT" if p_val < 0.05 else "Not Significant"
            clean_name = grievance.replace("narrative_", "").upper()
            print(f"{clean_name.ljust(15)} -> PROTESTS : p-value = {p_val:.4f} | {significance}")
        else:
            print(f"{grievance.ljust(15)} -> NOT FOUND IN DATA")

    print("===============================================================\n")


#the temporal pearson
from scipy.stats import pearsonr

records = []
max_lag = 14

for grievance in grievance_columns:
    if grievance in df_diff.columns:
        for lag in range(1, max_lag + 1):
            shifted = df_diff[grievance].shift(lag)
            aligned = pd.concat(
                [shifted, df_diff['narrative_protest_outcome']], axis=1
            ).dropna()

            if len(aligned) > 0:
                corr, p_val = pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                records.append({
                    'grievance': grievance,
                    'lag': lag,
                    'correlation': corr,
                    'p_value': p_val
                })

results_df = pd.DataFrame(records)

output_path = os.path.join("data", "lagged_correlations_tr.csv")
results_df.to_csv(output_path, index=False)
print(f"Saved lagged correlations to: {output_path}")

print("\n========== BEST LAG PER GRIEVANCE (Temporal Pearson) ==========")
print(f"{'Grievance':<20} {'Best Lag':>9} {'Correlation':>12} {'p-value':>10} {'Sig':>5}")
print("-" * 60)

#summary for terminal
for grievance, group in results_df.groupby('grievance'):
    group_sig = group[group['p_value'] < 0.05]

    if not group_sig.empty:
        best_row = group_sig.loc[group_sig['correlation'].abs().idxmax()]
    else:
        best_row = group.loc[group['correlation'].abs().idxmax()]

    sig_marker = "YES" if best_row['p_value'] < 0.05 else ""
    clean_name = grievance.replace("narrative_", "").upper()
    print(
        f"{clean_name:<20} {int(best_row['lag']):>9} "
        f"{best_row['correlation']:>12.4f} {best_row['p_value']:>10.4f} {sig_marker:>5}"
    )

print("=" * 60)


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
#this is a heatmap
pivot_corr = results_df.pivot(index='grievance', columns='lag', values='correlation')
pivot_pval = results_df.pivot(index='grievance', columns='lag', values='p_value')

pivot_corr.index = [i.replace("narrative_", "").upper() for i in pivot_corr.index]
pivot_pval.index = pivot_corr.index

fig, ax = plt.subplots(figsize=(14, 6))

cmap = plt.cm.RdYlGn #red means negative and green means positive
norm = mcolors.TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)

im = ax.imshow(pivot_corr.values, aspect='auto', cmap=cmap, norm=norm)

ax.set_xticks(range(len(pivot_corr.columns)))
ax.set_xticklabels([f"Lag {c}" for c in pivot_corr.columns], rotation=45, ha='right')
ax.set_yticks(range(len(pivot_corr.index)))
ax.set_yticklabels(pivot_corr.index)

#shows r value and star (*) significant ones
for i in range(pivot_corr.shape[0]):
    for j in range(pivot_corr.shape[1]):
        r   = pivot_corr.values[i, j]
        p   = pivot_pval.values[i, j]
        txt = f"{r:.2f}{'*' if p < 0.05 else ''}"
        ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                color='black' if abs(r) < 0.2 else 'white')

plt.colorbar(im, ax=ax, label='Pearson r')
ax.set_title("Lagged Correlation of Grievance Narratives - Protests (tr)\n* = p < 0.05", 
             fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(os.path.join("data", "heatmap_lagged_corr_tr.png"), dpi=150)
plt.show()
print("Saved heatmap.")


#correlation across lags per grievance
fig, ax = plt.subplots(figsize=(12, 5))

colors = plt.cm.tab10.colors
for i, grievance in enumerate(results_df['grievance'].unique()):
    grp = results_df[results_df['grievance'] == grievance].sort_values('lag')
    label = grievance.replace("narrative_", "").upper()
    ax.plot(grp['lag'], grp['correlation'], marker='o', label=label,
            color=colors[i % len(colors)], linewidth=1.8, markersize=4)
    
    #mark the stat significant lags with a filled dot
    sig = grp[grp['p_value'] < 0.05]
    ax.scatter(sig['lag'], sig['correlation'], color=colors[i % len(colors)],
               s=60, zorder=5, edgecolors='black', linewidths=0.6)

ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Pearson r")
ax.set_title("Correlation vs. Lag of Grievance Narratives - Protests (tr)\nFilled dots = p < 0.05")
ax.set_xticks(range(1, max_lag + 1))
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join("data", "lineplot_lagged_corr_tr.png"), dpi=150)
plt.show()
print("Saved line plot.")

