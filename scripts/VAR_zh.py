import pandas as pd
from statsmodels.tsa.api import VAR
import warnings
import os
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

warnings.filterwarnings("ignore")

# --- DYNAMIC PATH SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(script_dir), "data")
os.makedirs(data_dir, exist_ok=True)

input_path = os.path.join(data_dir, "var_input_zh.csv")
df = pd.read_csv(input_path, index_col='date_publish', parse_dates=True)

df_diff = df.diff().dropna()
print(f"Data loaded: {df.index.min().date()} to {df.index.max().date()}, {len(df)} rows")
print("Data differenced. Modeling daily shocks for Chinese data.")
print(f"Columns: {list(df.columns)}")

grievance_columns = [
    'narrative_gov', 'narrative_dem_reform', 'narrative_global',
    'narrative_religion', 'narrative_elections', 'narrative_basic_needs',
    'narrative_coup', 'narrative_violence'
]

# FEEDBACK #3: Chinese protest events (within 2018-2021 data range)
protest_starts_all = [
    pd.Timestamp("2019-06-09"),   # HK anti-extradition protests - 1M people
    pd.Timestamp("2019-07-01"),   # HK Legislative Council storming
    pd.Timestamp("2019-08-05"),   # HK general strike
    pd.Timestamp("2020-05-24"),   # HK anti-NSL protests
    pd.Timestamp("2021-07-20"),   # Zhengzhou flood aftermath protests
]

# auto-filter to events within data range
protest_starts = [t for t in protest_starts_all
                  if t >= df_diff.index.min() and t <= df_diff.index.max()]
print(f"\nUsing {len(protest_starts)} protest events within data range:")
for t in protest_starts:
    print(f"  {t.date()}")

# --- PRE-PROTEST TRAJECTORY PLOT ---
max_lag = 28
all_windows = []

for start in protest_starts:
    window = df_diff[
        (df_diff.index >= start - pd.Timedelta(days=max_lag)) &
        (df_diff.index <= start)
    ][grievance_columns].copy()
    window = window.iloc[::-1].reset_index(drop=True)
    window.index.name = 'lag'
    all_windows.append(window)

if all_windows:
    avg_window = pd.concat(all_windows).groupby(level=0).mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10.colors
    for i, col in enumerate(grievance_columns):
        label = col.replace("narrative_", "").upper()
        ax.plot(avg_window.index, avg_window[col], marker='o', label=label,
                color=colors[i], linewidth=1.8, markersize=4)
    ax.axvline(0, color='red', linestyle='--', linewidth=1, label='Protest Start')
    ax.set_xlabel("Days before protest (0 = protest start)")
    ax.set_ylabel("Avg differenced narrative score")
    ax.set_title(f"Arithmetic Average Grievance Trajectory Across {len(protest_starts)} Pre-Protest Windows (ZH)")
    ax.invert_xaxis()
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "avg_preprotest_trajectory_zh.png"), dpi=150)
    plt.show()

# --- GRANGER CAUSALITY ---
model = VAR(df_diff)

# FEEDBACK #1: expanded lag list
lags_to_test = [5, 6, 7, 8, 9, 14, 15, 16]

granger_records = []

for lag in lags_to_test:
    try:
        results = model.fit(lag)
        print(f"\n========== CHINESE GRANGER CAUSALITY RESULTS ({lag}-DAY LAG) ==========")
        for grievance in grievance_columns:
            if grievance in df_diff.columns:
                test_result = results.test_causality(narrative_protest_outcome, [grievance], kind='f')
                p_val = test_result.pvalue
                # FEEDBACK #2: 0/1
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
granger_df.to_csv(os.path.join(data_dir, "granger_results_zh.csv"), index=False)
print(f"\nSaved Granger results to: granger_results_zh.csv")

# --- PEARSON LAGGED CORRELATION ---
# FEEDBACK #1: full range -9 to +16, plus ±5 and ±3 windows
PRE_DAYS = 16
POST_DAYS = 9

records = []

for start in protest_starts:
    window = df_diff[
        (df_diff.index >= start - pd.Timedelta(days=PRE_DAYS)) &
        (df_diff.index <= start + pd.Timedelta(days=POST_DAYS))
    ][grievance_columns].copy()

    protest_binary = (window.index == start).astype(int)

    for grievance in grievance_columns:
        if grievance not in window.columns:
            continue
        for lag in range(-POST_DAYS, PRE_DAYS + 1):
            shifted = window[grievance].shift(lag)
            aligned = pd.concat(
                [shifted, pd.Series(protest_binary, index=window.index)], axis=1
            ).dropna()
            if len(aligned) > 2:
                corr, p_val = pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                records.append({
                    'grievance': grievance,
                    'lag': lag,
                    'protest': str(start.date()),
                    'correlation': corr,
                    'p_value': p_val,
                    # FEEDBACK #2: 0/1
                    'significant': 1 if p_val < 0.05 else 0
                })

results_df = pd.DataFrame(records)
avg_results = results_df.groupby(['grievance', 'lag'])[['correlation', 'p_value', 'significant']].mean().reset_index()
avg_results['significant'] = (avg_results['significant'] >= 0.5).astype(int)

results_df.to_csv(os.path.join(data_dir, "lagged_correlations_zh.csv"), index=False)
print(f"Saved lagged correlations to: lagged_correlations_zh.csv")


def plot_heatmap(avg_df, lag_range, title_suffix, filename):
    subset = avg_df[avg_df['lag'].isin(lag_range)]
    if subset.empty:
        print(f"No data for {title_suffix}, skipping.")
        return
    pivot_corr = subset.pivot(index='grievance', columns='lag', values='correlation')
    pivot_pval = subset.pivot(index='grievance', columns='lag', values='p_value')
    pivot_corr.index = [i.replace("narrative_", "").upper() for i in pivot_corr.index]
    pivot_pval.index = pivot_corr.index

    fig, ax = plt.subplots(figsize=(max(8, len(lag_range) * 0.9), 4))
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)
    im = ax.imshow(pivot_corr.values, aspect='auto', cmap=cmap, norm=norm)

    ax.set_xticks(range(len(pivot_corr.columns)))
    ax.set_xticklabels(
        [f"+{c}d" if c > 0 else (f"{c}d" if c < 0 else "0d") for c in pivot_corr.columns],
        rotation=45, ha='right'
    )
    ax.set_yticks(range(len(pivot_corr.index)))
    ax.set_yticklabels(pivot_corr.index)

    for i in range(pivot_corr.shape[0]):
        for j in range(pivot_corr.shape[1]):
            r = pivot_corr.values[i, j]
            p = pivot_pval.values[i, j]
            if np.isnan(r):
                continue
            txt = f"{r:.2f}{'*' if p < 0.05 else ''}"
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=7, color='black' if abs(r) < 0.2 else 'white')

    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.set_title(
        f"Lagged Correlation - Protests (zh) | {title_suffix}\n"
        f"+Nd = N days before protest   -Nd = N days after   * = p < 0.05"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, filename), dpi=150)
    plt.show()
    print(f"Saved: {filename}")


# FEEDBACK #1: three window sizes
full_range  = list(range(-POST_DAYS, PRE_DAYS + 1))
five_range  = list(range(-5, 6))
three_range = list(range(-3, 4))

plot_heatmap(avg_results, full_range,  "Full window (-9 to +16 days)", "heatmap_lagged_corr_zh_full.png")
plot_heatmap(avg_results, five_range,  "±5 day window",                 "heatmap_lagged_corr_zh_5day.png")
plot_heatmap(avg_results, three_range, "±3 day window",                 "heatmap_lagged_corr_zh_3day.png")

# --- PER-EVENT LINE PLOTS ---
if not results_df.empty:
    n_events = results_df['protest'].nunique()
    fig, axes = plt.subplots(1, n_events, figsize=(6 * n_events, 5), sharey=True)
    if n_events == 1:
        axes = [axes]
    for ax, (protest_date, group) in zip(axes, results_df.groupby('protest')):
        for grievance, grp in group.groupby('grievance'):
            grp = grp.sort_values('lag')
            label = grievance.replace("narrative_", "").upper()
            ax.plot(grp['lag'], grp['correlation'], marker='o', label=label, linewidth=1.8, markersize=4)
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
        ax.set_title(protest_date, fontsize=9)
        ax.set_xlabel("Lag (+before / -after protest)")
    axes[0].set_ylabel("Pearson r")
    axes[-1].legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    plt.suptitle("Per-Event Lagged Correlation (ZH)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "lineplot_lagged_corr_per_protest_zh.png"), dpi=150)
    plt.show()

print("\nDone. Output files saved in zh/ folder.")
