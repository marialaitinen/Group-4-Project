import pandas as pd
import matplotlib.pyplot as plt

# 1. Load daily data
df = pd.read_csv("zh_narrative_timeseries_daily.csv", 
                 index_col='date_publish', parse_dates=True)

# 2. Focus on the high-density window 2018-2021
df_window = df['2018-01-01':'2021-12-31'].copy()

# 3. Apply 7-day rolling average (weekly smoothing, better for sustained data)
df_smooth = df_window.rolling(window=7).mean().dropna()

print(f"Window: {len(df_smooth)} days of data")
print(f"\nSample (first 5 rows):")
print(df_smooth.head())

# 4. Plot the narrative pulse
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(df_smooth.index, df_smooth['narrative_protest'], 
             color='red', linewidth=1.5)
axes[0].set_ylabel('Protest')
axes[0].set_title('Chinese Grievance Narrative Pulse (2018-2021)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(df_smooth.index, df_smooth['narrative_corruption'], 
             color='orange', linewidth=1.5)
axes[1].set_ylabel('Corruption')
axes[1].grid(True, alpha=0.3)

axes[2].plot(df_smooth.index, df_smooth['narrative_economic'], 
             color='blue', linewidth=1.5)
axes[2].set_ylabel('Economic')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('zh_narrative_pulse.png', dpi=150)
plt.show()

# 5. Save for VAR model
df_smooth.to_csv("zh_var_input_ready.csv")
print("\n✅ Saved to zh_var_input_ready.csv")