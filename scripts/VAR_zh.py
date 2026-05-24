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

# Load the Chinese data
input_path = os.path.join(data_dir, "var_input_zh.csv")
if not os.path.exists(input_path):
    print(f"File not found: {input_path}")
    exit()

df = pd.read_csv(input_path, index_col='date_publish', parse_dates=True)

# Apply Differencing
df_diff = df.diff().dropna()
print(f"Original Data Limits: {df.index.min().date()} to {df.index.max().date()}")
print(f"Differenced Data Limits: {df_diff.index.min().date()} to {df_diff.index.max().date()}")
print(f"Total Rows (Differenced): {len(df_diff)}")
print("Data differenced. Modeling daily shocks for Chinese data.")

# List of grievance columns
grievance_columns = [
    'narrative_gov', 'narrative_dem_reform', 'narrative_global',
    'narrative_religion', 'narrative_elections', 'narrative_basic_needs',
    'narrative_coup', 'narrative_violence'
]

# FEEDBACK #3: Chinese protest events (PRESERVED)
protest_starts_all = [
    pd.Timestamp("2019-06-09"),   # HK anti-extradition protests
    pd.Timestamp("2019-07-01"),   # HK Legislative Council storming
    pd.Timestamp("2019-08-05"),   # HK general strike
    pd.Timestamp("2020-05-24"),   # HK anti-NSL protests
    pd.Timestamp("2021-07-20"),   # Zhengzhou flood aftermath protests
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
    ax_traj.set_ylabel("Agg Avg differenced narrative score", fontsize=12)
    ax_traj.set_title(f"Aggregated Average Grievance Trajectory Across {len(protest_starts)} Pre-Protest Windows (ZH)", fontsize=16, pad=15)
    ax_traj.invert_xaxis()
    ax_traj.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=11)
    
    traj_path = os.path.join(data_dir, "trajectory_zh.png")
    fig_traj.savefig(traj_path, dpi=150, bbox_inches='tight')
    plt.close(fig_traj)
    print(f"\nSaved Category 1: {traj_path}")

# =====================================================================
# CATEGORY 1B: INDIVIDUAL ARITHMETIC TRAJECTORIES
# =====================================================================
if all_windows:
    fig_indiv, axes_indiv = plt.subplots(1, len(protest_starts), 
                                         figsize=(6 * len(protest_starts), 6), 
                                         sharey=True, layout='constrained', squeeze=False)
    axes_indiv = axes_indiv.flatten()
    fig_indiv.suptitle("Individual Grievance Trajectories Per Protest Event (ZH)", 
                        fontsize=18, fontweight='bold', y=1.02)

    for ax, window, start in zip(axes_indiv, all_windows, protest_starts):
        for i, col in enumerate(grievance_columns):
            label = col.replace("narrative_", "").upper()
            ax.plot(window.index, window[col], marker='o', label=label,
                    color=colors[i], linewidth=2, markersize=4)

        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Protest Start')
        ax.set_title(str(start.date()), fontsize=13, pad=10)
        ax.set_xlabel("Days before protest", fontsize=11)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

    axes_indiv[0].set_ylabel("Differenced narrative score", fontsize=12)
    axes_indiv[-1].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=10)

    indiv_traj_path = os.path.join(data_dir, "trajectory_individual_zh.png")
    fig_indiv.savefig(indiv_traj_path, dpi=150, bbox_inches='tight')
    plt.close(fig_indiv)
    print(f"Saved Category 1B: {indiv_traj_path}")

# =====================================================================
# GRANGER CAUSALITY
# =====================================================================
model = VAR(df_diff)
lags_to_test = [5, 6, 7, 8, 9, 14, 15, 16]
granger_records = []

for lag in lags_to_test:
    try:
        results = model.fit(lag)
        print(f"\n========== CHINESE GRANGER CAUSALITY RESULTS ({lag}-DAY LAG) ==========")
        for grievance in grievance_columns:
            if grievance in df_diff.columns:
                test_result = results.test_causality('narrative_protest_outcome', [grievance], kind='f')
                p_val = test_result.pvalue
                significant = 1 if p_val < 0.05 else 0
                clean_name = grievance.replace("narrative_", "").upper()
                print(f"{clean_name.ljust(15)} -> PROTESTS : p-value = {p_val:.4f} | significant = {significant}")
                
                granger_records.append({
                    'language': 'zh',
                    'lag': lag,
                    'grievance': clean_name,
                    'p_value': round(p_val, 4),
                    'significant': significant
                })
    except Exception as e:
        print(f"Lag {lag} skipped: {e}")

granger_df = pd.DataFrame(granger_records)
if not granger_df.empty:
    granger_df.to_csv(os.path.join(data_dir, "granger_results_zh.csv"), index=False)
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
    results_df.to_csv(os.path.join(data_dir, "lagged_correlations_zh.csv"), index=False)

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
        f"Lagged Correlation - Protests (zh) | {title_suffix}\n"
        f"+Nd = N days before protest   -Nd = N days after   * = p < 0.05",
        pad=10, fontsize=14
    )

if not results_df.empty:
    fig_hm, axes_hm = plt.subplots(3, 1, figsize=(14, 18), layout='constrained')
    fig_hm.suptitle("Pearson Lagged Correlation Heatmaps (ZH)", fontsize=20, fontweight='bold', y=1.02)

    plot_heatmap_on_ax(avg_results, list(range(-POST_DAYS, PRE_DAYS + 1)), "Full window (-9 to +16 days)", axes_hm[0])
    plot_heatmap_on_ax(avg_results, list(range(-5, 6)), "±5 day window", axes_hm[1])
    plot_heatmap_on_ax(avg_results, list(range(-3, 4)), "±3 day window", axes_hm[2])

    hm_path = os.path.join(data_dir, "heatmaps_zh.png")
    fig_hm.savefig(hm_path, dpi=150, bbox_inches='tight')
    plt.close(fig_hm)
    print(f"Saved Category 2: {hm_path}")

# =====================================================================
# CATEGORY 3: PER-EVENT LINE PLOTS
# =====================================================================
if not results_df.empty:
    fig_events, axes_events = plt.subplots(1, n_events, figsize=(5 * max(n_events, 2), 6), sharey=True, layout='constrained', squeeze=False)
    axes_events = axes_events.flatten()
    
    fig_events.suptitle("Per-Event Correlation Breakdown (ZH)", fontsize=20, fontweight='bold', y=1.05)

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

    events_path = os.path.join(data_dir, "per_event_lines_zh.png")
    fig_events.savefig(events_path, dpi=150, bbox_inches='tight')
    plt.close(fig_events)
    print(f"Saved Category 3: {events_path}")

# =====================================================================
# CATEGORY 4: IMPULSE RESPONSE FUNCTIONS
# =====================================================================
FIXED_LAG = 7
IRF_HORIZON = 16
TARGET = 'narrative_protest_outcome'
WINDOW_DAYS = 100 

irf_records = {}

for start in protest_starts:
    try:
        window_start = start - pd.Timedelta(days=WINDOW_DAYS)
        window_idx = pd.date_range(start=window_start, end=start - pd.Timedelta(days=1), freq='D')
        local_df = df_diff.reindex(window_idx, fill_value=0)

        # Drop columns that are entirely zero — they cause singular matrices
        local_df = local_df.loc[:, (local_df != 0).any(axis=0)]

        if len(local_df) < FIXED_LAG + 10:
            print(f"Skipping {start.date()}: insufficient data ({len(local_df)} rows)")
            continue

        if local_df.shape[1] < 2:
            print(f"Skipping {start.date()}: too few non-zero columns")
            continue

        results = VAR(local_df).fit(FIXED_LAG)
        irf = results.irf(IRF_HORIZON)

        var_names = local_df.columns.tolist()

        if TARGET not in var_names:
            print(f"Skipping {start.date()}: TARGET column was all-zero and got dropped")
            continue

        target_idx = var_names.index(TARGET)

        event_irfs = {}
        for grievance in grievance_columns:
            if grievance in var_names:
                shock_idx = var_names.index(grievance)
                event_irfs[grievance] = irf.orth_irfs[:, target_idx, shock_idx]

        irf_records[str(start.date())] = event_irfs
        print(f"IRF fitted for {start.date()} on {len(local_df)} rows, {local_df.shape[1]} cols")
    except Exception as e:
        print(f"IRF failed for {start.date()}: {e}")

# Aggregated IRF
if irf_records:
    agg_irf = {}
    for grievance in grievance_columns:
        arrays = [irf_records[d][grievance] for d in irf_records if grievance in irf_records[d]]
        if arrays:
            agg_irf[grievance] = np.mean(arrays, axis=0)

    # Plot
    n_panels = len(irf_records) + 1
    fig_irf, axes_irf = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5),
                                      sharey=True, layout='constrained', squeeze=False)
    axes_irf = axes_irf.flatten()
    fig_irf.suptitle("Impulse Response Functions: Shock to Protest Outcome (ZH)",
                      fontsize=18, fontweight='bold', y=1.02)

    horizon_x = range(IRF_HORIZON + 1)

    for ax, (protest_date, event_irfs) in zip(axes_irf, irf_records.items()):
        for i, grievance in enumerate(grievance_columns):
            if grievance in event_irfs:
                label = grievance.replace("narrative_", "").upper()
                ax.plot(horizon_x, event_irfs[grievance],
                        marker='o', markersize=3, linewidth=2,
                        label=label, color=colors[i % len(colors)])
        ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
        ax.set_title(protest_date, fontsize=12, pad=8)
        ax.set_xlabel("Days after shock", fontsize=10)
        ax.grid(True, alpha=0.3)

    ax_agg = axes_irf[len(irf_records)]
    for i, grievance in enumerate(grievance_columns):
        if grievance in agg_irf:
            label = grievance.replace("narrative_", "").upper()
            ax_agg.plot(horizon_x, agg_irf[grievance],
                        marker='o', markersize=3, linewidth=2.5,
                        label=label, color=colors[i % len(colors)])
    ax_agg.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax_agg.set_title("Aggregated (mean)", fontsize=12, pad=8)
    ax_agg.set_xlabel("Days after shock", fontsize=10)
    ax_agg.grid(True, alpha=0.3)

    axes_irf[0].set_ylabel("Response in protest outcome", fontsize=11)
    axes_irf[-1].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=10)

    irf_path = os.path.join(data_dir, "irf_zh.png")
    fig_irf.savefig(irf_path, dpi=150, bbox_inches='tight')
    plt.close(fig_irf)
    print(f"Saved Category 4: {irf_path}")
else:
    print("WARNING: No IRF records fitted, IRF plot skipped")

# =====================================================================
# CATEGORY 5: VAR COEFFICIENT HEATMAPS
# =====================================================================
def plot_coeff_heatmap(ax, df, title_suffix):
    cmap = plt.cm.RdBu
    max_val = max(abs(df.values.min()), abs(df.values.max()), 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)
    
    im = ax.imshow(df.values, aspect='auto', cmap=cmap, norm=norm)
    
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=10)
    
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = df.values[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:.3f}", ha='center', va='center', 
                    fontsize=9, color='black' if abs(val) < max_val * 0.5 else 'white')
            
    plt.colorbar(im, ax=ax, label='Coefficient Value')
    ax.set_title(f"VAR Coefficients | {title_suffix}", pad=10, fontsize=12)

coeff_records = {}

for start in protest_starts:
    try:
        window_start = start - pd.Timedelta(days=WINDOW_DAYS)
        window_idx = pd.date_range(start=window_start, end=start - pd.Timedelta(days=1), freq='D')
        local_df = df_diff.reindex(window_idx, fill_value=0)

        if len(local_df) < FIXED_LAG + 10:
            print(f"Skipping {start.date()}: insufficient data ({len(local_df)} rows)")
            continue

        results = VAR(local_df).fit(FIXED_LAG)
        var_names = local_df.columns.tolist()
        target_idx = var_names.index(TARGET)

        rows = {}
        for grievance in grievance_columns:
            if grievance in var_names:
                shock_idx = var_names.index(grievance)
                lag_coefs = [results.coefs[lag, target_idx, shock_idx]
                             for lag in range(FIXED_LAG)]
                rows[grievance.replace("narrative_", "").upper()] = lag_coefs

        coeff_df = pd.DataFrame(rows, index=[f"lag{i+1}" for i in range(FIXED_LAG)]).T
        coeff_records[str(start.date())] = coeff_df
        print(f"Coefficients fitted for {start.date()}")
    except Exception as e:
        print(f"Coeff heatmap failed for {start.date()}: {e}")

if coeff_records:
    all_coeff_arrays = np.array([df.values for df in coeff_records.values()])
    agg_coeff_df = pd.DataFrame(
        np.mean(all_coeff_arrays, axis=0),
        index=list(coeff_records.values())[0].index,
        columns=list(coeff_records.values())[0].columns
    )

    n_hm = len(coeff_records) + 1
    fig_hm2, axes_hm2 = plt.subplots(1, n_hm, figsize=(4.5 * n_hm, 6),
                                     layout='constrained', squeeze=False)
    axes_hm2 = axes_hm2.flatten()
    fig_hm2.suptitle("VAR Coefficients: Grievances to Protest Outcome (ZH)",
                      fontsize=18, fontweight='bold', y=1.02)

    for ax, (protest_date, coeff_df) in zip(axes_hm2, coeff_records.items()):
        plot_coeff_heatmap(ax, coeff_df, protest_date)

    plot_coeff_heatmap(axes_hm2[len(coeff_records)], agg_coeff_df, "Aggregated (mean)")

    coeff_path = os.path.join(data_dir, "coeff_heatmap_zh.png")
    fig_hm2.savefig(coeff_path, dpi=150, bbox_inches='tight')
    plt.close(fig_hm2)
    print(f"Saved Category 5: {coeff_path}")
else:
    print("WARNING: No coefficient records fitted, coeff heatmap skipped")

print("\nProcessing complete.")