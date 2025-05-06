import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

st.set_page_config(page_title="Finance ML Report Generator", layout="wide")
st.title("📊 Finance Report Generation with Machine Learning")

uploaded_file = st.file_uploader("Upload your Financial Statements CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]
    st.write("Sample Data", df.head())

    features = [
        'Revenue', 'Gross Profit', 'Earning Per Share', 'EBITDA',
        'Share Holder Equity', 'Current Ratio', 'Debt/Equity Ratio',
        'ROE', 'ROA', 'ROI', 'Net Profit Margin', 'Return on Tangible Equity'
    ]
    target = 'Net Income'

    # Remove missing values
    df_model = df.dropna(subset=features + [target])
    X = df_model[features]
    y = df_model[target]

    # Random Forest & XGBoost Regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    st.write("Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Random Forest MAE", f"{mean_absolute_error(y_test, y_pred_rf):,.2f}")
    col1.metric("Random Forest R²", f"{r2_score(y_test, y_pred_rf):.4f}")
    col2.metric("XGBoost MAE", f"{mean_absolute_error(y_test, y_pred_xgb):,.2f}")
    col2.metric("XGBoost R²", f"{r2_score(y_test, y_pred_xgb):.4f}")

    # Predict for all rows (Random Forest)
    rf_all = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_all.fit(X, y)
    df_model['Predicted Net Income'] = rf_all.predict(X)

    st.write("## 🔎 Filter Financial Report by Company")
    company_list = df_model['Company'].unique()
    selected_company = st.selectbox("Select a company for financial report:", company_list)
    filtered_df = df_model[df_model['Company'] == selected_company]
    st.dataframe(filtered_df[['Year', 'Company', 'Revenue', 'Gross Profit', 'EBITDA', 'Predicted Net Income', 'Net Income']])


    # Plots
    st.write("## 📈 Actual vs Predicted Net Income")
    fig1, ax1 = plt.subplots()
    ax1.scatter(y_test, y_pred_rf, label='Random Forest')
    ax1.scatter(y_test, y_pred_xgb, label='XGBoost', marker='x')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
    ax1.set_xlabel("Actual Net Income")
    ax1.set_ylabel("Predicted Net Income")
    ax1.set_title("Actual vs Predicted Net Income")
    ax1.legend()
    st.pyplot(fig1)

    # LSTM Time Series (for user-selected company)
    st.write("## 🔮 LSTM Net Income Forecast (Single Company)")
    company_list = df_model['Company'].unique()
    company = st.selectbox("Select a company for LSTM time series forecasting:", company_list)
    df_company = df[df['Company'] == company].sort_values('Year')
    net_income = df_company['Net Income'].values.reshape(-1, 1)
    if len(net_income) > 5:
        scaler = MinMaxScaler()
        net_income_scaled = scaler.fit_transform(net_income)
        def create_sequences(data, n_steps):
            X, y = [], []
            for i in range(len(data) - n_steps):
                X.append(data[i:i+n_steps])
                y.append(data[i+n_steps])
            return np.array(X), np.array(y)
        n_steps = 3
        X_seq, y_seq = create_sequences(net_income_scaled, n_steps)
        if len(X_seq) > 0:
            split = int(0.8 * len(X_seq))
            X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
            y_train_seq, y_test_seq = y_seq[:split], y_seq[split:]
            model = Sequential([
                LSTM(32, input_shape=(n_steps, 1)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mae')
            model.fit(X_train_seq, y_train_seq, epochs=50, verbose=0)
            y_pred_seq = model.predict(X_test_seq)
            y_pred_seq_inv = scaler.inverse_transform(y_pred_seq)
            y_test_seq_inv = scaler.inverse_transform(y_test_seq)
            st.write(f"LSTM MAE: {mean_absolute_error(y_test_seq_inv, y_pred_seq_inv):,.2f}")
            st.write(f"LSTM R²: {r2_score(y_test_seq_inv, y_pred_seq_inv):.4f}")
            fig2, ax2 = plt.subplots()
            ax2.plot(range(len(y_test_seq_inv)), y_test_seq_inv, label='Actual')
            ax2.plot(range(len(y_pred_seq_inv)), y_pred_seq_inv, label='LSTM Predicted')
            ax2.set_title(f"{company} Net Income Time Series Forecast (LSTM)")
            ax2.set_xlabel("Time Step")
            ax2.set_ylabel("Net Income")
            ax2.legend()
            st.pyplot(fig2)
        else:
            st.warning("Not enough data points for LSTM time series forecasting.")
    else:
        st.warning("Not enough data points for LSTM time series forecasting.")

    # Downloadable CSV
    st.write("## ⬇️ Download Results")
    st.download_button(
        label="Download predictions as CSV",
        data=df_model.to_csv(index=False),
        file_name="Financial_Statements_with_Predictions.csv",
        mime="text/csv"
    )
else:
    st.info("Please upload a CSV file to begin.")
