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
os.makedirs(data_dir, exist_ok=True)

# Load the Korean data
input_path = os.path.join(data_dir, "var_input_ko.csv")
if not os.path.exists(input_path):
    print(f"File not found: {input_path}")
    exit()

df = pd.read_csv(input_path, index_col='date_publish', parse_dates=True)

# Apply Differencing 
df_diff = df.diff().dropna()
print("Data differenced. Modeling daily 'shocks' for Korean data.")

# List of grievance columns
grievance_columns = [
    'narrative_gov', 'narrative_dem_reform', 'narrative_global', 
    'narrative_religion', 'narrative_elections', 'narrative_basic_needs', 
    'narrative_coup', 'narrative_violence'
]

# NOTE: Update these timestamps for Korean specific protest events!
protest_starts = [
    pd.Timestamp("2024-01-01"), 
]

max_lag = 28
all_windows = []

for start in protest_starts:
    try:
        window = df_diff[
            (df_diff.index >= start - pd.Timedelta(days=max_lag)) &
            (df_diff.index <= start)
        ][grievance_columns].copy()
        
        if not window.empty:
            window = window.iloc[::-1].reset_index(drop=True)
            window.index.name = 'lag'
            all_windows.append(window)
    except Exception as e:
        print(f"Could not process window for {start}: {e}")

if all_windows:
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
    ax.set_title(f"Average Grievance Trajectory - Korean ({len(all_windows)} Windows)")
    ax.invert_xaxis() 
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "avg_preprotest_trajectory_ko.png"), dpi=150)
    plt.close()

# Initialize the VAR Model
model = VAR(df_diff)
lags_to_test = [7, 14]

for lag in lags_to_test:
    results = model.fit(lag)
    print(f"\n========== KOREAN GRANGER CAUSALITY RESULTS ({lag}-DAY LAG) ==========")
    for grievance in grievance_columns:
        if grievance in df_diff.columns:
            target = 'narrative_protest_outcome' 
            if target in df_diff.columns:
                test_result = results.test_causality(target, [grievance], kind='f')
                p_val = test_result.pvalue
                significance = "⭐ SIGNIFICANT" if p_val < 0.05 else "Not Significant"
                clean_name = grievance.replace("narrative_", "").upper()
                print(f"{clean_name.ljust(15)} -> PROTESTS : p-value = {p_val:.4f} | {significance}")

# Temporal Pearson logic
records = []
pearson_lag = 14 

for start in protest_starts:
    mask = (df_diff.index >= start - pd.Timedelta(days=28)) & (df_diff.index <= start)
    window = df_diff[mask][grievance_columns].copy()
    
    for grievance in grievance_columns:
        if grievance in window.columns:
            for lag in range(1, pearson_lag + 1):
                shifted = window[grievance].shift(lag)
                protest_binary = (window.index == start).astype(int)
                aligned = pd.concat([shifted, pd.Series(protest_binary, index=window.index)], axis=1).dropna()
                if len(aligned) > 5:
                    corr, p_val = pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                    records.append({
                        'grievance': grievance, 'lag': lag,
                        'protest': str(start.date()), 'correlation': corr, 'p_value': p_val
                    })

if records:
    results_df = pd.DataFrame(records)
    output_path = os.path.join(data_dir, "lagged_correlations_ko.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Saved lagged correlations to: {output_path}")
