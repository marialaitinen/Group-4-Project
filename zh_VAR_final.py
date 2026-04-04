import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("zh_var_input_ready.csv", 
                 index_col='date_publish', parse_dates=True)

# 2. Stationarity check (ADF test)
print("=== ADF Stationarity Test (before differencing) ===")
for col in df.columns:
    result = adfuller(df[col].dropna())
    status = "✅ Stationary" if result[1] < 0.05 else "❌ Non-stationary"
    print(f"{col}: p={result[1]:.4f} {status}")

# 3. Apply differencing
df_diff = df.diff().dropna()

print("\n=== ADF Stationarity Test (after differencing) ===")
for col in df_diff.columns:
    result = adfuller(df_diff[col].dropna())
    status = "✅ Stationary" if result[1] < 0.05 else "❌ Non-stationary"
    print(f"{col}: p={result[1]:.4f} {status}")

# 4. Fit VAR model
print("\n=== Fitting VAR Model ===")
model = VAR(df_diff)
results = model.fit(5)
print(results.summary())

# 5. Granger Causality Tests
print("\n=== GRANGER CAUSALITY RESULTS ===")

# Does Economic narrative lead Protest narrative?
econ_test = results.test_causality(
    'narrative_protest', ['narrative_economic'], kind='f')
print(f"Economic -> Protest:    p={econ_test.pvalue:.4f} "
      f"{'✅ Significant' if econ_test.pvalue < 0.05 else '❌ Not significant'}")

# Does Corruption narrative lead Protest narrative?
corr_test = results.test_causality(
    'narrative_protest', ['narrative_corruption'], kind='f')
print(f"Corruption -> Protest:  p={corr_test.pvalue:.4f} "
      f"{'✅ Significant' if corr_test.pvalue < 0.05 else '❌ Not significant'}")

# Does Protest narrative lead Economic narrative?
prot_econ_test = results.test_causality(
    'narrative_economic', ['narrative_protest'], kind='f')
print(f"Protest -> Economic:    p={prot_econ_test.pvalue:.4f} "
      f"{'✅ Significant' if prot_econ_test.pvalue < 0.05 else '❌ Not significant'}")

# 6. Impulse Response Function
print("\nGenerating IRF plots...")
irf = results.irf(14)

irf.plot(impulse='narrative_economic', response='narrative_protest')
plt.suptitle('Response of Protest to Economic Narrative Shock', y=1.02)
plt.tight_layout()
plt.savefig('zh_irf_econ_to_protest.png', dpi=150, bbox_inches='tight')

irf.plot(impulse='narrative_corruption', response='narrative_protest')
plt.suptitle('Response of Protest to Corruption Narrative Shock', y=1.02)
plt.tight_layout()
plt.savefig('zh_irf_corr_to_protest.png', dpi=150, bbox_inches='tight')

plt.show()
print("\n✅ Done! Check zh_irf_*.png for impulse response plots")