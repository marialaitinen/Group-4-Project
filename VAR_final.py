import pandas as pd
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv("var_input_ready.csv", index_col='date_publish', parse_dates=True)

# 2. Apply Differencing 

df_diff = df.diff().dropna()

print(" Data differenced. We are now modeling daily 'shocks' rather than total volume.")

# 3. Fit the VAR Model

model = VAR(df_diff)
results = model.fit(5)

# 4. THE CAUSALITY TEST
print("\n --- GRANGER CAUSALITY RESULTS (On Differenced Data) ---")

# Test: Does Economic Change lead to Protest Change?
econ_test = results.test_causality('narrative_protest', ['narrative_economic'], kind='f')
print(f"Economic -> Protest: p-value = {econ_test.pvalue:.4f}")

# Test: Does Corruption Change lead to Protest Change?
corr_test = results.test_causality('narrative_protest', ['narrative_corruption'], kind='f')
print(f"Corruption -> Protest: p-value = {corr_test.pvalue:.4f}")

# 5. Visualizing the "Lead-Lag" Relationship
irf = results.irf(10)
irf.plot(impulse='narrative_economic', response='narrative_protest')
plt.title("Response of Protests to an Economic Narrative Shock")
plt.show()