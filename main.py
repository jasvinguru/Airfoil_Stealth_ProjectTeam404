ximport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Set plot aesthetic style
sns.set_theme(style="whitegrid")

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv('AirfoilSelfNoise.csv')

print("\n--- Dataset Summary ---")
print("Data Shape:", df.shape)
print("Missing Values per column:\n", df.isnull().sum())
print("\nDescriptive Statistics:\n", df.describe())

# 2. EDA & Visualizations
print("\nGenerating charts...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# jasvin part
# Chart 1: Feature Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0, 0])
axes[0, 0].set_title('Figure 1: Feature Correlation Heatmap', fontsize=12, fontweight='bold')

# This code creates our first graph, the correlation heatmap.
# df.corr() calculates the correlation between the variables, 
# and sns.heatmap() displays it as a heatmap


#karthi paart
# Chart 2: Angle of Attack vs Air Disruption
sns.scatterplot(data=df, x='alpha', y='delta', hue='U_infinity', palette='viridis', ax=axes[0, 1])
axes[0, 1].set_title('Figure 2: Angle of Attack (alpha) vs Air Disruption (delta)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Angle of Attack (deg)')
axes[0, 1].set_ylabel('Displacement Thickness delta (m)')

# This code creates our second graph.
# We use a scatter plot to show the relationship between angle of attack, alpha, and displacement thickness, delta. 
# The different colours represent different air velocities


#kirthic part
# Chart 3: Distribution of SSPL
sns.histplot(df['SSPL'], kde=True, color='crimson', ax=axes[1, 0])
axes[1, 0].set_title('Figure 3: Distribution of Sound Pressure Level (SSPL)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Sound Pressure Level (dB)')

# This code creates our third graph. It is a histogram showing how the SSPL values are distributed in our dataset.
# It helps us understand the range and frequency of sound pressure levels.”


#janak part
# Chart 4: Stealth Envelope Identification
stealth_mask = (df['delta'] <= 0.0025) & (df['SSPL'] <= 118)
df['Signature_Category'] = np.where(stealth_mask, 'Stealth Envelope', 'Standard Flight')

sns.scatterplot(data=df, x='delta', y='SSPL', hue='Signature_Category', 
                palette={'Standard Flight': 'gray', 'Stealth Envelope': 'green'}, s=50, ax=axes[1, 1])
axes[1, 1].axhline(118, color='red', linestyle='--', label='Acoustic Threshold (118 dB)')
axes[1, 1].axvline(0.0025, color='blue', linestyle='--', label='Disruption Threshold (0.0025m)')
axes[1, 1].set_title('Figure 4: Stealth Envelope (Low Delta & Low SSPL)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Displacement Thickness delta (m)')
axes[1, 1].set_ylabel('Sound Pressure Level SSPL (dB)')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('airfoil_stealth_analysis.png', dpi=300)
print("Saved visualization figure as 'airfoil_stealth_analysis.png'")
plt.show()

# 3. Machine Learning Model Training
print("\nTraining Machine Learning Model...")
X = df[['f', 'alpha', 'c', 'U_infinity']]
y = df[['delta', 'SSPL']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")
print(f"R^2 Score for Air Disruption (delta): {r2_score(y_test['delta'], y_pred[:, 0]):.4f}")
print(f"R^2 Score for Acoustic Noise (SSPL): {r2_score(y_test['SSPL'], y_pred[:, 1]):.4f}")
