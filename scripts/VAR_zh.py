import pandas as pd
from statsmodels.tsa.api import VAR
import warnings
import os

warnings.filterwarnings("ignore")

# Load Data
df = pd.read_csv(os.path.join("data", "var_input_zh.csv"), index_col='date_publish', parse_dates=True)
df_diff = df.diff().dropna()

# Fit VAR
model = VAR(df_diff)
grievance_columns = [
    'narrative_gov', 'narrative_dem_reform', 'narrative_global', 
    'narrative_religion', 'narrative_elections', 'narrative_basic_needs', 
    'narrative_coup', 'narrative_violence'
]

# Test Lags
for lag in [7, 14]:
    results = model.fit(lag)
    print(f"\n========== CANTONESE GRANGER CAUSALITY ({lag}-DAY LAG) ==========")
    for grievance in grievance_columns:
        if grievance in df_diff.columns:
            p_val = results.test_causality('narrative_protest_outcome', [grievance], kind='f').pvalue
            sig = "⭐ SIGNIFICANT" if p_val < 0.05 else "Not Significant"
            print(f"{grievance.replace('narrative_', '').upper().ljust(15)} -> PROTESTS : p = {p_val:.4f} | {sig}")
    print("===============================================================")