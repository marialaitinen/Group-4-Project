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
warnings.filterwarnings("ignore")

# --- DYNAMIC PATH SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(script_dir), "data")
os.makedirs(data_dir, exist_ok=True)

# Load the Italian data
input_path = os.path.join(data_dir, "var_input_it.csv")
if not os.path.exists(input_path):
    print(f"File not found: {input_path}")
    exit()

df = pd.read_csv(input_path, index_col='date_publish', parse_dates=True)

# Apply Differencing 
df_diff = df.diff().dropna()
print(f"Original Data Limits: {df.index.min().date()} to {df.index.max().date()}")
print(f"Differenced Data Limits: {df_diff.index.min().date()} to {df_diff.index.max().date()}")
print(f"Total Rows (Differenced): {len(df_diff)}")
print("Data differenced. Modeling daily shocks for Italian data.")

# List of grievance columns
grievance_columns = [
    'narrative_gov', 'narrative_dem_reform', 'narrative_global', 
    'narrative_religion', 'narrative_elections', 'narrative_basic_needs', 
    'narrative_coup', 'narrative_violence'
]

# Italian protest events
protest_starts_all = [
    pd.Timestamp("2020-03-25"), 
    pd.Timestamp("2011-10-15")
]

# Auto-filter to events within differenced data range
protest_starts = [t for t in protest_starts_all 
                  if t >= df_diff.index.min() and t <= df_diff.index.max()]

print(f"\nUsing {len(protest_starts)} protest events within data range:")
for t in protest_starts:
    print(f"  {t.date()}")

n_events = max(1, len(protest_starts))

# =====================================================================
# CATEGORY 1: PRE-PROTEST TRAJECTORY PLOT
# =====================================================================
max_lag = 28
all_windows = []

for start in protest_starts:
    # Continuous daily index so 1 row exactly = 1 day (handles missing weekends)
    window_idx = pd.date_range(start=start - pd.Timedelta(days=max_lag), end=start, freq='D')
    
    # Missing dates are filled with 0 (meaning no change in narrative from previous day)
    window = df_diff.reindex(window_idx, fill_value=0)[grievance_columns].copy()
    
    window = window.iloc[::-1].reset_index(drop=True)
    window.index.name = 'lag'
    all_windows.append(window)

if all_windows:
    fig_traj, ax_traj = plt.subplots(figsize=(12, 6), layout='constrained')
    avg_window = pd.concat(all_windows).groupby(level=0).mean()
    colors = plt.cm.tab10.colors
    
    for i, col in enumerate(grievance_columns):
        label = col.replace("narrative_", "").upper()
        ax_traj.plot(avg_window.index, avg_window[col], marker='o', label=label,
                     color=colors[i], linewidth=2.5, markersize=5)
        
    ax_traj.axvline(0, color='red', linestyle='--', linewidth=2, label='Protest Start')
    ax_traj.set_xlabel("Days before protest (0 = protest start)", fontsize=12)
    ax_traj.set_ylabel("Avg differenced narrative score", fontsize=12)
    ax_traj.set_title(f"Average Grievance Trajectory Across {len(protest_starts)} Pre-Protest Windows (IT)", fontsize=16, pad=15)
    ax_traj.invert_xaxis()
    ax_traj.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=11)
    
    traj_path = os.path.join(data_dir, "trajectory_it.png")
    fig_traj.savefig(traj_path, dpi=150, bbox_inches='tight')
    plt.close(fig_traj)
    print(f"\nSaved Category 1: {traj_path}")

# =====================================================================
# GRANGER CAUSALITY
# =====================================================================
model = VAR(df_diff)
lags_to_test = [5, 6, 7, 8, 9, 14, 15, 16]
granger_records = []

for lag in lags_to_test:
    try:
        results = model.fit(lag)
        print(f"\n========== ITALIAN GRANGER CAUSALITY RESULTS ({lag}-DAY LAG) ==========")
        for grievance in grievance_columns:
            if grievance in df_diff.columns:
                test_result = results.test_causality('narrative_protest_outcome', [grievance], kind='f')
                p_val = test_result.pvalue
                significant = 1 if p_val < 0.05 else 0
                clean_name = grievance.replace("narrative_", "").upper()
                print(f"{clean_name.ljust(15)} -> PROTESTS : p-value = {p_val:.4f} | significant = {significant}")
                
                granger_records.append({
                    'language': 'it',
                    'lag': lag,
                    'grievance': clean_name,
                    'p_value': round(p_val, 4),
                    'significant': significant
                })
    except Exception as e:
        print(f"Lag {lag} skipped: {e}")

granger_df = pd.DataFrame(granger_records)
if not granger_df.empty:
    granger_df.to_csv(os.path.join(data_dir, "granger_results_it.csv"), index=False)
else:
    print("\nWARNING: No Granger records generated.")

# =====================================================================
# PEARSON LAGGED CORRELATION (FIXED ZERO-VARIANCE & TIME GAPS)
# =====================================================================
PRE_DAYS = 16
POST_DAYS = 9
records = []

for start in protest_starts:
    # Continuous calendar index ensures accurate lagging and prevents empty binary arrays
    window_idx = pd.date_range(start=start - pd.Timedelta(days=PRE_DAYS), 
                               end=start + pd.Timedelta(days=POST_DAYS), freq='D')
    
    window = df_diff.reindex(window_idx, fill_value=0)[grievance_columns].copy()
    protest_binary = (window.index == start).astype(int)
    
    for grievance in grievance_columns:
        if grievance not in window.columns:
            continue
            
        for lag in range(-POST_DAYS, PRE_DAYS + 1):
            shifted = window[grievance].shift(lag)
            aligned = pd.concat([shifted, pd.Series(protest_binary, index=window.index)], axis=1).dropna()
            
            if len(aligned) > 2:
                # Check for zero variance to prevent math errors returning NaN
                std_x = aligned.iloc[:, 0].std()
                std_y = aligned.iloc[:, 1].std()
                
                if std_x == 0 or std_y == 0:
                    corr, p_val = 0.0, 1.0  # Flatline -> explicitly zero correlation
                else:
                    corr, p_val = pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                    
                records.append({
                    'grievance': grievance,
                    'lag': lag,
                    'protest': str(start.date()),
                    'correlation': corr,
                    'p_value': p_val,
                    'significant': 1 if p_val < 0.05 else 0
                })

results_df = pd.DataFrame(records)
if not results_df.empty:
    avg_results = results_df.groupby(['grievance', 'lag'])[['correlation', 'p_value', 'significant']].mean().reset_index()
    avg_results['significant'] = (avg_results['significant'] >= 0.5).astype(int)
    results_df.to_csv(os.path.join(data_dir, "lagged_correlations_it.csv"), index=False)

# =====================================================================
# CATEGORY 2: HEATMAPS (Full, 5-Day, 3-Day)
# =====================================================================
def plot_heatmap_on_ax(avg_df, lag_range, title_suffix, ax):
    subset = avg_df[avg_df['lag'].isin(lag_range)]
    if subset.empty:
        ax.set_title(f"No data for {title_suffix}")
        ax.axis('off')
        return
        
    pivot_corr = subset.pivot(index='grievance', columns='lag', values='correlation')
    pivot_pval = subset.pivot(index='grievance', columns='lag', values='p_value')
    pivot_corr.index = [i.replace("narrative_", "").upper() for i in pivot_corr.index]
    pivot_pval.index = pivot_corr.index

    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)
    im = ax.imshow(pivot_corr.values, aspect='auto', cmap=cmap, norm=norm)

    ax.set_xticks(range(len(pivot_corr.columns)))
    ax.set_xticklabels(
        [f"+{c}d" if c > 0 else (f"{c}d" if c < 0 else "0d") for c in pivot_corr.columns], 
        rotation=45, ha='right', fontsize=11
    )
    ax.set_yticks(range(len(pivot_corr.index)))
    ax.set_yticklabels(pivot_corr.index, fontsize=11)

    for i in range(pivot_corr.shape[0]):
        for j in range(pivot_corr.shape[1]):
            r = pivot_corr.values[i, j]
            p = pivot_pval.values[i, j]
            if np.isnan(r):
                continue
            txt = f"{r:.2f}{'*' if p < 0.05 else ''}"
            ax.text(j, i, txt, ha='center', va='center', 
                    fontsize=10, color='black' if abs(r) < 0.2 else 'white')

    cbar = plt.colorbar(im, ax=ax, label='Pearson r')
    cbar.ax.tick_params(labelsize=10) 
    
    ax.set_title(
        f"Lagged Correlation - Protests (it) | {title_suffix}\n"
        f"+Nd = N days before protest   -Nd = N days after   * = p < 0.05",
        pad=10, fontsize=14
    )

if not results_df.empty:
    fig_hm, axes_hm = plt.subplots(3, 1, figsize=(14, 18), layout='constrained')
    fig_hm.suptitle("Pearson Lagged Correlation Heatmaps (IT)", fontsize=20, fontweight='bold', y=1.02)

    plot_heatmap_on_ax(avg_results, list(range(-POST_DAYS, PRE_DAYS + 1)), "Full window (-9 to +16 days)", axes_hm[0])
    plot_heatmap_on_ax(avg_results, list(range(-5, 6)), "±5 day window", axes_hm[1])
    plot_heatmap_on_ax(avg_results, list(range(-3, 4)), "±3 day window", axes_hm[2])

    hm_path = os.path.join(data_dir, "heatmaps_it.png")
    fig_hm.savefig(hm_path, dpi=150, bbox_inches='tight')
    plt.close(fig_hm)
    print(f"Saved Category 2: {hm_path}")

# =====================================================================
# CATEGORY 3: PER-EVENT LINE PLOTS
# =====================================================================
if not results_df.empty:
    fig_events, axes_events = plt.subplots(1, n_events, figsize=(5 * max(n_events, 2), 6), sharey=True, layout='constrained', squeeze=False)
    axes_events = axes_events.flatten()
    
    fig_events.suptitle("Per-Event Correlation Breakdown (IT)", fontsize=20, fontweight='bold', y=1.05)

    for ax, (protest_date, group) in zip(axes_events, results_df.groupby('protest')):
        group = group.dropna(subset=['correlation']) 
        
        if group.empty:
            ax.set_title(f"{protest_date}\n(Not enough valid data)", fontsize=12)
            continue
            
        for grievance, grp in group.groupby('grievance'):
            grp = grp.sort_values('lag')
            label = grievance.replace("narrative_", "").upper()
            ax.plot(grp['lag'], grp['correlation'], marker='o', label=label, linewidth=2, markersize=4)
        
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax.axhline(0, color='gray', linestyle=':', linewidth=1.5)
        ax.set_title(protest_date, fontsize=14, pad=10)
        ax.set_xlabel("Lag (+before / -after protest)", fontsize=12)
        ax.grid(True, alpha=0.3)
        
    axes_events[0].set_ylabel("Pearson r", fontsize=13)
    
    if len(axes_events) > 0:
        axes_events[-1].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=11)

    events_path = os.path.join(data_dir, "per_event_lines_it.png")
    fig_events.savefig(events_path, dpi=150, bbox_inches='tight')
    plt.close(fig_events)
    print(f"Saved Category 3: {events_path}")

print("\nProcessing complete.")