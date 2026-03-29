import pandas as pd

# Load and clean
df = pd.read_csv("filtered_grievance_data_en.csv")
df['date_publish'] = pd.to_datetime(df['date_publish'], errors='coerce')

# 1. Filter out the "Time Travelers" (Keep only 2010 to 2026)
mask = (df['date_publish'] > '2010-01-01') & (df['date_publish'] < '2026-12-31')
df = df[mask].copy()

# 2. Create "Narrative Indicators" (Boolean flags)
# We use your keywords to mark which articles belong to which 'grievance'
df['is_protest'] = df['maintext'].str.contains('protest|riot|strike', case=False, na=False)
df['is_corruption'] = df['maintext'].str.contains('corruption|fraud|bribery', case=False, na=False)
df['is_economic'] = df['maintext'].str.contains('austerity|inflation|prices|energy', case=False, na=False)

# 3. Resample to DAILY counts
# This is the "Pulse" table
daily_ts = df.resample('D', on='date_publish').agg({
    'is_protest': 'sum',
    'is_corruption': 'sum',
    'is_economic': 'sum'
}).rename(columns={
    'is_protest': 'narrative_protest',
    'is_corruption': 'narrative_corruption',
    'is_economic': 'narrative_economic'
})

# 4. Fill gaps (days with zero articles)
daily_ts = daily_ts.fillna(0)

# Save the final time-series for the VAR model
daily_ts.to_csv("narrative_timeseries_daily.csv")

print("---  TIME-SERIES PREVIEW ---")
print(daily_ts.tail(10))
print(f"\nSaved {len(daily_ts)} days of narrative data to 'narrative_timeseries_daily.csv'")