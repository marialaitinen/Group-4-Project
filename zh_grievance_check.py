import pandas as pd

print("Loading data, large file please wait...")
df = pd.read_csv("filtered_grievance_data_zh.csv")
print(f"Total rows: {len(df):,}")

# 1. Clean dates
df['date_publish'] = pd.to_datetime(df['date_publish'], errors='coerce')
before = len(df)
df = df.dropna(subset=['date_publish'])
after = len(df)
print(f"Valid dates: {after:,} articles (dropped {before - after:,} articles with no date)")

# 2. Filter date range
mask = (df['date_publish'] > '2010-01-01') & (df['date_publish'] < '2026-12-31')
df = df[mask].copy()
print(f"Within 2010-2026 range: {len(df):,} articles")

# 3. Narrative classification (Traditional Chinese keywords)
df['is_protest'] = df['maintext'].str.contains(
    '抗議|示威|游行|暴動|騷亂|罷工|佔領', case=False, na=False)

df['is_corruption'] = df['maintext'].str.contains(
    '腐敗|貪污|賄賂|詐欺|醜聞|反腐敗', case=False, na=False)

df['is_economic'] = df['maintext'].str.contains(
    '緊縮|通貨膨脹|物價|失業|能源|天然氣|工會', case=False, na=False)

print(f"\nNarrative distribution:")
print(f"  Protest:    {df['is_protest'].sum():,} articles")
print(f"  Corruption: {df['is_corruption'].sum():,} articles")
print(f"  Economic:   {df['is_economic'].sum():,} articles")

# 4. Resample to daily counts
daily_ts = df.resample('D', on='date_publish').agg({
    'is_protest': 'sum',
    'is_corruption': 'sum',
    'is_economic': 'sum'
}).rename(columns={
    'is_protest': 'narrative_protest',
    'is_corruption': 'narrative_corruption',
    'is_economic': 'narrative_economic'
})

daily_ts = daily_ts.fillna(0)
daily_ts.to_csv("zh_narrative_timeseries_daily.csv")

print(f"\nTime series preview (last 10 active days):")
print(daily_ts[daily_ts.sum(axis=1) > 0].tail(10))
print(f"\n✅ Saved to zh_narrative_timeseries_daily.csv")