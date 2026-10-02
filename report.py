import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv('AirfoilSelfNoise.csv')

X = df[['f', 'alpha', 'c', 'U_infinity']]
y = df[['delta', 'SSPL']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n========================================")
print("       COMPUTER REPORT CARD             ")
print("========================================")
print(f"Air Disruption (delta) Score: {r2_score(y_test['delta'], y_pred[:, 0]) * 100:.2f}%")
print(f"Noise Level (SSPL) Score:     {r2_score(y_test['SSPL'], y_pred[:, 1]) * 100:.2f}%")
print("========================================\n")