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

# Load the TURKISH data
df = pd.read_csv(os.path.join("data", "var_input_tr.csv"), index_col='date_publish', parse_dates=True)

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
ax.invert_xaxis()  # so it reads left=14 days before and right=protest day
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join("data", "avg_preprotest_trajectory_tr.png"), dpi=150)
plt.show()

# Initialize the VAR Model
model = VAR(df_diff)



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

print("\n========== GRANGER CAUSALITY - PRE-PROTEST WINDOWS ONLY ==========")
print("Null Hypothesis: The Grievance does NOT cause Protests")
print("---------------------------------------------------------------")

# combine all pre-protest windows into one dataframe
windowed_frames = []
for start in protest_starts:
    w = df_diff[
        (df_diff.index >= start - pd.Timedelta(days=28)) &
        (df_diff.index <= start)
    ].copy()
    w['protest_binary'] = (w.index == start).astype(int)
    windowed_frames.append(w)

df_windowed = pd.concat(windowed_frames).sort_index()

# run VAR on windowed data
cols_for_var = grievance_columns + ['protest_binary']
df_windowed_var = df_windowed[cols_for_var].dropna()

model_windowed = VAR(df_windowed_var)

for lag in [7, 14]:
    try:
        results_w = model_windowed.fit(lag)
        print(f"\n--- {lag}-DAY LAG (windowed) ---")
        for grievance in grievance_columns:
            test_result = results_w.test_causality('protest_binary', [grievance], kind='f')
            p_val = test_result.pvalue
            significance = "⭐ SIGNIFICANT" if p_val < 0.05 else "Not Significant"
            clean_name = grievance.replace("narrative_", "").upper()
            print(f"{clean_name.ljust(15)} -> PROTESTS : p-value = {p_val:.4f} | {significance}")
    except Exception as e:
        print(f"lag={lag} failed: {e}")

print("===============================================================\n")


#the temporal pearson


records = []
pearson_lag = 14  # reduced because windows are only 28 days

for start in protest_starts:
    window = df_diff[
        (df_diff.index >= start - pd.Timedelta(days=28)) &
        (df_diff.index <= start)
    ][grievance_columns].copy()
    
    for grievance in grievance_columns:
        if grievance in window.columns:
            for lag in range(1, pearson_lag + 1):
                shifted = window[grievance].shift(lag)
                # protest_binary: 1 only on the start day (index position 0 after reversal)
                protest_binary = (window.index == start).astype(int)
                
                aligned = pd.concat(
                    [shifted, pd.Series(protest_binary, index=window.index)], axis=1
                ).dropna()

                if len(aligned) > 0:
                    corr, p_val = pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                    records.append({
                        'grievance': grievance,
                        'lag': lag,
                        'protest': str(start.date()),
                        'correlation': corr,
                        'p_value': p_val
                    })

results_df = pd.DataFrame(records)
# average correlation across the 3 episodes per grievance+lag
avg_results = results_df.groupby(['grievance', 'lag'])[['correlation', 'p_value']].mean().reset_index()

output_path = os.path.join("data", "lagged_correlations_tr.csv")
results_df.to_csv(output_path, index=False)
print(f"Saved lagged correlations to: {output_path}")

# per protest
print("\n========== BEST LAG PER GRIEVANCE - PER PROTEST (Temporal Pearson) ==========")
for protest_date, protest_group in results_df.groupby('protest'):
    print(f"\n--- Protest: {protest_date} ---")
    print(f"{'Grievance':<20} {'Best Lag':>9} {'Correlation':>12} {'p-value':>10} {'Sig':>5}")
    print("-" * 60)
    for grievance, group in protest_group.groupby('grievance'):
        group_sig = group[group['p_value'] < 0.05]
        if not group_sig.empty:
            best_row = group_sig.loc[group_sig['correlation'].abs().idxmax()]
        else:
            best_row = group.loc[group['correlation'].abs().idxmax()]
        sig_marker = "YES" if best_row['p_value'] < 0.05 else ""
        clean_name = grievance.replace("narrative_", "").upper()
        print(f"{clean_name:<20} {int(best_row['lag']):>9} {best_row['correlation']:>12.4f} {best_row['p_value']:>10.4f} {sig_marker:>5}")

# averaged
print("\n========== BEST LAG PER GRIEVANCE - AVERAGED (Temporal Pearson) ==========")
print(f"{'Grievance':<20} {'Best Lag':>9} {'Correlation':>12} {'p-value':>10} {'Sig':>5}")
print("-" * 60)
for grievance, group in avg_results.groupby('grievance'):
    group_sig = group[group['p_value'] < 0.05]
    if not group_sig.empty:
        best_row = group_sig.loc[group_sig['correlation'].abs().idxmax()]
    else:
        best_row = group.loc[group['correlation'].abs().idxmax()]
    sig_marker = "YES" if best_row['p_value'] < 0.05 else ""
    clean_name = grievance.replace("narrative_", "").upper()
    print(f"{clean_name:<20} {int(best_row['lag']):>9} {best_row['correlation']:>12.4f} {best_row['p_value']:>10.4f} {sig_marker:>5}")
print("=" * 60)


#this is a heatmap
pivot_corr = avg_results.pivot(index='grievance', columns='lag', values='correlation')
pivot_pval = avg_results.pivot(index='grievance', columns='lag', values='p_value')

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
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, (protest_date, group) in zip(axes, results_df.groupby('protest')):
    colors = plt.cm.tab10.colors
    for i, (grievance, grp) in enumerate(group.groupby('grievance')):
        grp = grp.sort_values('lag')
        label = grievance.replace("narrative_", "").upper()
        ax.plot(grp['lag'], grp['correlation'], marker='o', label=label,
                color=colors[i % len(colors)], linewidth=1.8, markersize=4)
        
        sig = grp[grp['p_value'] < 0.05]
        ax.scatter(sig['lag'], sig['correlation'], color=colors[i % len(colors)],
                   s=60, zorder=5, edgecolors='black', linewidths=0.6)

    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_title(f"Protest: {protest_date}")
    ax.set_xlabel("Lag (days)")
    ax.set_xticks(range(1, pearson_lag + 1))

axes[0].set_ylabel("Pearson r")
axes[0].legend(bbox_to_anchor=(0, -0.3), loc='upper left', fontsize=8, ncol=2)
plt.suptitle("Lagged Correlation per Protest Episode (TR)\nFilled dots = p < 0.05", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join("data", "lineplot_lagged_corr_per_protest_tr.png"), dpi=150)
plt.show()

