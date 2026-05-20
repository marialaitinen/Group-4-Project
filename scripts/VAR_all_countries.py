import pandas as pd
from statsmodels.tsa.api import VAR
import warnings
import os

# Suppress warnings for a clean terminal
warnings.filterwarnings("ignore")

# --- PATH SETUP: Parallel Folder Logic ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))

print(f"INITIALIZING VAR PIPELINE")
print(f"Reading from: {data_dir}\n")

# The mapped categories
topic_category_map = {
    0: "narrative_protest_outcome", 1: "narrative_gov", 2: "narrative_dem_reform",
    3: "narrative_global", 4: "narrative_religion", 5: "narrative_elections",
    6: "narrative_basic_needs", 7: "narrative_coup", 8: "narrative_violence"
}

grievance_columns = [
    'narrative_gov', 'narrative_dem_reform', 'narrative_global', 
    'narrative_religion', 'narrative_elections', 'narrative_basic_needs', 
    'narrative_coup', 'narrative_violence'
]

languages = ['en', 'es', 'fr', 'tr', 'zh']
lags_to_test = [7, 14]

# Hold results for final export
master_results = []

for lang in languages:
    # Use the absolute data_dir path
    file_path = os.path.join(data_dir, f"classified_articles_{lang}.csv")
    
    if not os.path.exists(file_path):
        print(f"Skipping {lang.upper()} - File not found.")
        continue
        
    print(f"--- LOADING {lang.upper()} DATASET ---")
    df = pd.read_csv(file_path)
    df['date_publish'] = pd.to_datetime(df['date_publish'])
    
    # Filter: Unique countries, dropping NaNs and explicitly removing 'undefined'
    countries = df['country'].dropna().unique()
    countries = [c for c in countries if str(c).lower() != 'undefined']
    
    for country in countries:
        country_df = df[df['country'] == country]
        
        # Statistical Guardrail
        if len(country_df) < 100:
            continue
            
        # Aggregating daily
        daily_ts = (
            country_df.groupby([pd.Grouper(key='date_publish', freq='D'), 'topic'])
            .size()
            .unstack(fill_value=0)
        )
        
        daily_ts = daily_ts.rename(columns=topic_category_map)
        columns_to_keep = [col for col in topic_category_map.values() if col in daily_ts.columns]
        daily_ts = daily_ts[columns_to_keep]
        
        # Guardrail: Must have the outcome variable
        if 'narrative_protest_outcome' not in daily_ts.columns:
            continue
            
        # Differencing for stationarity
        df_diff = daily_ts.diff().dropna()
        
        try:
            model = VAR(df_diff)
            
            for lag in lags_to_test:
                results = model.fit(lag)
                found_sig = False
                
                for grievance in grievance_columns:
                    if grievance in df_diff.columns:
                        test_result = results.test_causality('narrative_protest_outcome', [grievance], kind='f')
                        p_val = test_result.pvalue
                        
                        clean_name = grievance.replace("narrative_", "").upper()
                        
                        # Logging ALL results to terminal
                        if p_val < 0.05:
                            print(f" ⭐ {country.upper()} | {clean_name} causes Protests (p={p_val:.4f}) at {lag} lags")
                            found_sig = True
                        else:
                            print(f"  - {country.upper()} | {clean_name} not significant (p={p_val:.4f}) at {lag} lags")

                        master_results.append({
                            'Language': lang.upper(),
                            'Country': country.upper(),
                            'Lag_Days': lag,
                            'Grievance': clean_name,
                            'P_Value': round(p_val, 4),
                            'Significant': 'Yes' if p_val < 0.05 else 'No'
                        })
                
                # You might want to keep or remove this line now that everything prints
                if not found_sig and lag == 14:
                    print(f"  . {country.upper()} checked (no significant results overall).")

        except Exception:
            # Skip countries where math fails due to zero variance
            continue

# FINAL EXPORT
print("\n" + "="*40)
print("GLOBAL PIPELINE COMPLETE!")

if master_results:
    results_df = pd.DataFrame(master_results)
    results_df = results_df.sort_values(by=['Significant', 'P_Value'], ascending=[False, True]) 
    
    output_path = os.path.join(data_dir, "FINAL_GLOBAL_VAR_RESULTS_ALL.csv")
    results_df.to_csv(output_path, index=False)
    print(f"All testing complete. Results saved to: {output_path}")
else:
    print("No valid models were produced. Check data volume per country.")