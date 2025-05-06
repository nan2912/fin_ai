import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Load the data
df = pd.read_csv('Financial Statements.csv')
print(df.head())

# Clean column names
df.columns = [col.strip() for col in df.columns]

# Display a sample
print("Sample data:\n", df.head())

# 2. Select features and target for regression
features = [
    'Revenue', 'Gross Profit', 'Earning Per Share', 'EBITDA',
    'Share Holder Equity', 'Current Ratio', 'Debt/Equity Ratio',
    'ROE', 'ROA', 'ROI', 'Net Profit Margin', 'Return on Tangible Equity'
]
target = 'Net Income'

# Remove rows with missing values in selected columns
df_model = df.dropna(subset=features + [target])

X = df_model[features]
y = df_model[target]

# 3. Random Forest & XGBoost Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("\nRandom Forest MAE:", mean_absolute_error(y_test, y_pred_rf))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))
print("XGBoost MAE:", mean_absolute_error(y_test, y_pred_xgb))
print("XGBoost R2:", r2_score(y_test, y_pred_xgb))

# 4. LSTM Time Series Forecasting (for one company, e.g., AAPL)
company = "AAPL"
df_aapl = df[df['Company'] == company].sort_values('Year')
net_income = df_aapl['Net Income'].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
net_income_scaled = scaler.fit_transform(net_income)

# Create sequences
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

n_steps = 3
X_seq, y_seq = create_sequences(net_income_scaled, n_steps)

# Train/test split
split = int(0.8 * len(X_seq))
X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
y_train_seq, y_test_seq = y_seq[:split], y_seq[split:]

# Build LSTM model
model = Sequential([
    LSTM(32, input_shape=(n_steps, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mae')
model.fit(X_train_seq, y_train_seq, epochs=50, verbose=0)

# Predict
y_pred_seq = model.predict(X_test_seq)
y_pred_seq_inv = scaler.inverse_transform(y_pred_seq)
y_test_seq_inv = scaler.inverse_transform(y_test_seq)

print("\nLSTM MAE:", mean_absolute_error(y_test_seq_inv, y_pred_seq_inv))
print("LSTM R2:", r2_score(y_test_seq_inv, y_pred_seq_inv))

# 5. Simple Financial Report Generation
print("\n--- Sample Financial Report (Random Forest) ---")
for i in range(3):
    idx = X_test.index[i]
    print(f"\nYear: {df_model.loc[idx, 'Year']}, Company: {df_model.loc[idx, 'Company']}")
    print(f"Revenue: {df_model.loc[idx, 'Revenue']}")
    print(f"Gross Profit: {df_model.loc[idx, 'Gross Profit']}")
    print(f"EBITDA: {df_model.loc[idx, 'EBITDA']}")
    print(f"Predicted Net Income: {y_pred_rf[i]:.2f}")
    print(f"Actual Net Income: {df_model.loc[idx, 'Net Income']}")

# 6. Visualization
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred_rf, label='Random Forest')
plt.scatter(y_test, y_pred_xgb, label='XGBoost', marker='x')
plt.xlabel("Actual Net Income")
plt.ylabel("Predicted Net Income")
plt.title("Actual vs Predicted Net Income")
plt.legend()
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

plt.figure(figsize=(8,4))
plt.plot(range(len(y_test_seq_inv)), y_test_seq_inv, label='Actual')
plt.plot(range(len(y_pred_seq_inv)), y_pred_seq_inv, label='LSTM Predicted')
plt.title("AAPL Net Income Time Series Forecast (LSTM)")
plt.xlabel("Time Step")
plt.ylabel("Net Income")
plt.legend()
plt.show()
