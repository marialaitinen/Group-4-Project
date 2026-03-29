import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the daily counts
df = pd.read_csv("narrative_timeseries_daily.csv", index_col='date_publish', parse_dates=True)

# 2. ZOOM IN: Focus only on the window where you actually have data (2016)
# Based on your output, Sept 2016 is your "Hot Zone"
df_2016 = df['2016-08-15':'2016-10-15'].copy()

# 3. SMOOTHING: Create a 3-day rolling average to handle the 1-to-300 jumps
df_smooth = df_2016.rolling(window=3).mean().dropna()

print(f"✅ Created a 'Golden Window' of {len(df_smooth)} high-volume days.")

# 4. QUICK PLOT: See your narratives "Pulse"
df_smooth.plot(figsize=(12, 6), title="Grievance Narrative Pulse (Sept 2016)")
plt.ylabel("Article Volume (3-day avg)")
plt.grid(True)
plt.show()

# Save this for the VAR model
df_smooth.to_csv("var_input_ready.csv")