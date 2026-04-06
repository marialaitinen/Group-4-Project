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