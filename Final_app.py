# main_app.py
import streamlit as st

st.set_page_config(page_title="Business Intelligence Suite", layout="wide")
st.markdown("""
    <style>
        .main {background-color: #f9f9fb;}
        .css-1d391kg {background-color: #2c3e50;}
        .stButton>button {background-color: #2980b9; color: white;}
        .stRadio>div>label {font-weight: bold;}
        .stSidebar {background-color: #23272f;}
        .stSidebar .sidebar-content {color: #fff;}
    </style>
""", unsafe_allow_html=True)

st.title("Business Intelligence Suite")
st.markdown("A unified platform for Accounting, Tax Risk Analysis, and Financial ML Reporting.")

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "📘 Accounting Module",
        "📊 Tax Risk Analysis",
        "📈 Finance ML Report Generator"
    ]
)

if page == "📘 Accounting Module":
    # --- ACCOUNTING MODULE (from acctingf1.py) ---
    import pandas as pd
    import sqlite3
    from datetime import datetime

    conn = sqlite3.connect("accounting.db", check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            account TEXT,
            debit REAL,
            credit REAL,
            narration TEXT
        )
    ''')
    conn.commit()

    def add_transaction(date, account, debit, credit, narration):
        c.execute("INSERT INTO transactions (date, account, debit, credit, narration) VALUES (?, ?, ?, ?, ?)",
                  (date, account, debit, credit, narration))
        conn.commit()

    def get_all_transactions():
        return pd.read_sql_query("SELECT * FROM transactions ORDER BY date", conn)

    def get_ledger(account):
        return pd.read_sql_query("""
            SELECT date, debit, credit, narration
            FROM transactions
            WHERE account = ?
            ORDER BY date
        """, conn, params=(account,))

    def generate_trial_balance():
        df = pd.read_sql_query("""
            SELECT account,
                   SUM(debit) AS total_debit,
                   SUM(credit) AS total_credit
            FROM transactions
            GROUP BY account
        """, conn)
        df["balance"] = df["total_debit"] - df["total_credit"]
        return df

    st.header("📘 Accounting Module with CSV Support")
    menu = st.radio("Select Action", [
        "Add Transaction", "Upload Dataset", "View Ledger", "Trial Balance", "Export Data"
    ])

    if menu == "Add Transaction":
        st.subheader("➕ Add New Transaction")
        col1, col2 = st.columns(2)
        with col1:
            date = st.date_input("Date", datetime.today())
            account = st.text_input("Account Name")
        with col2:
            debit = st.number_input("Debit Amount", min_value=0.0)
            credit = st.number_input("Credit Amount", min_value=0.0)
            narration = st.text_area("Narration")
        if st.button("Add Transaction"):
            if not account or (debit == 0 and credit == 0):
                st.error("Please fill in all required fields.")
            else:
                add_transaction(str(date), account, debit, credit, narration)
                st.success("✅ Transaction added successfully!")

    elif menu == "Upload Dataset":
        st.subheader("⬆️ Upload Transactions CSV")
        file = st.file_uploader("Upload your CSV file (columns: date, account, debit, credit, narration)", type="csv")
        if file is not None:
            df = pd.read_csv(file)
            required_cols = {"date", "account", "debit", "credit", "narration"}
            if required_cols.issubset(df.columns):
                st.dataframe(df.head())
                for _, row in df.iterrows():
                    try:
                        add_transaction(row['date'], row['account'], float(row['debit']), float(row['credit']), str(row['narration']))
                    except Exception as e:
                        st.error(f"Error in row: {row} - {e}")
                st.success("✅ All transactions added to database!")
            else:
                st.error(f"CSV must contain columns: {required_cols}")

    elif menu == "View Ledger":
        st.subheader("📒 View Account Ledger")
        account_name = st.text_input("Enter Account Name:")
        if st.button("Show Ledger"):
            ledger_df = get_ledger(account_name)
            if ledger_df.empty:
                st.warning("No transactions found for this account.")
            else:
                st.dataframe(ledger_df)

    elif menu == "Trial Balance":
        st.subheader("📊 Trial Balance")
        trial_df = generate_trial_balance()
        total_debit = trial_df["total_debit"].sum()
        total_credit = trial_df["total_credit"].sum()
        st.dataframe(trial_df)
        st.metric("🔢 Total Debit", f"{total_debit:,.2f}")
        st.metric("🔢 Total Credit", f"{total_credit:,.2f}")

    elif menu == "Export Data":
        st.subheader("⬇️ Export All Transactions")
        df = get_all_transactions()
        st.dataframe(df)
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv_data, "transactions.csv", "text/csv")

elif page == "📊 Tax Risk Analysis":
    # --- TAX RISK ANALYSIS DASHBOARD (from tax_risk_pred.py) ---
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier, XGBRegressor
    from sklearn.model_selection import train_test_split


    # --- Add CSV File Uploader ---
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your Tax Risk CSV file",
        type=["csv"],
        help="Columns should match the sample dataset"  
    )

    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        # Drop unnecessary columns if present
        for col in ['Taxpayer_ID', 'Audit_to_Tax_Ratio']:
            if col in df.columns:
                df = df.drop([col], axis=1)
            df = df.drop_duplicates()
        return df

    df = None
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.sidebar.success("✅ File uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
    elif 'tax_risk_dataset.csv' in os.listdir():
        df = load_data('tax_risk_dataset.csv')
        st.sidebar.info("Using sample data (tax_risk_dataset.csv).")
    else:
        st.error("No data loaded! Please upload a CSV file with the required columns.")
        st.stop()

    le_industry = LabelEncoder()
    df['Industry'] = le_industry.fit_transform(df['Industry'])
    le_risk = LabelEncoder()
    df['Risk_Label'] = le_risk.fit_transform(df['Risk_Label'])

    X_filing = df.drop(['Tax_Liability', 'Risk_Label'], axis=1)
    y_filing = df['Tax_Liability']
    X_risk = df.drop(['Risk_Label'], axis=1)
    y_risk = df['Risk_Label']

    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_filing, y_filing, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_risk, y_risk, test_size=0.2, stratify=y_risk, random_state=42)

    clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    clf.fit(X_train_r, y_train_r)
    reg = XGBRegressor(random_state=42)
    reg.fit(X_train_f, y_train_f)

    option = st.radio("Go to", ["🏁 Overview", "📊 EDA", "📈 Feature Importance", "🔍 Predict Tax Risk"])

    if option == "🏁 Overview":
        st.title("📊 Tax Risk Analysis Dashboard")
        st.markdown("""
        Welcome to the interactive Tax Risk Analysis Dashboard.
        This tool uses machine learning to analyze taxpayer data and:
        - Predict risk categories (Low, Medium, High)
        - Estimate tax liabilities
        Navigate using the sidebar to explore the data and try live predictions!
        """)

    elif option == "📊 EDA":
        st.title("📊 Exploratory Data Analysis")
        df['Risk_Label_Name'] = le_risk.inverse_transform(df['Risk_Label'])
        st.subheader("Risk Label Distribution with Enhanced Visualization")
        risk_counts = df['Risk_Label_Name'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        bars = ax1.bar(risk_counts.index, risk_counts.values, color=['#66c2a5', '#fc8d62', '#8da0cb'])
        ax1.set_title("Number of Taxpayers by Risk Category", fontsize=14)
        ax1.set_xlabel("Risk Category")
        ax1.set_ylabel("Count")
        ax1.bar_label(bars, padding=3)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig1)

        st.subheader("Correlation Heatmap")
        df_numeric = df.select_dtypes(include=[float, int])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, ax=ax)
        st.pyplot(fig)

        st.subheader("Revenue Distribution by Risk Label (Boxplot + Violin)")
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 6))
        sns.boxplot(x='Risk_Label_Name', y='Revenue', data=df, ax=ax3a, palette='Set3')
        ax3a.set_title("Boxplot of Revenue by Risk Label")
        sns.violinplot(x='Risk_Label_Name', y='Revenue', data=df, ax=ax3b, palette='Set2')
        ax3b.set_title("Violin Plot of Revenue by Risk Label")
        st.pyplot(fig3)

        st.subheader("Industry vs Risk Label (Stacked Bar Chart)")
        industry_risk = df.groupby(['Industry', 'Risk_Label_Name']).size().unstack().fillna(0)
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        industry_risk.plot(kind='bar', stacked=True, ax=ax4, colormap='Pastel1')
        ax4.set_title("Taxpayer Risk Distribution by Industry")
        ax4.set_xlabel("Industry Code")
        ax4.set_ylabel("Count")
        ax4.legend(title="Risk Label")
        st.pyplot(fig4)

    elif option == "📈 Feature Importance":
        st.title("📈 Feature Importances")
        st.subheader("For Tax Risk Classification")
        importances_clf = clf.feature_importances_
        fig4, ax4 = plt.subplots()
        sns.barplot(x=importances_clf, y=X_risk.columns, ax=ax4, palette='viridis')
        ax4.set_title("XG BOOST - Risk Model")
        st.pyplot(fig4)
        st.subheader("For Tax Liability Regression")
        importances_reg = reg.feature_importances_
        fig5, ax5 = plt.subplots()
        sns.barplot(x=importances_reg, y=X_filing.columns, ax=ax5, palette='crest')
        ax5.set_title("XG BOOST - Liability Model")
        st.pyplot(fig5)

    elif option == "🔍 Predict Tax Risk":
        st.title("🔍 Predict Tax Risk")
        st.markdown("Input taxpayer features to predict their **risk category**.")
        input_data = {}
        for col in X_risk.columns:
            if df[col].dtype in ['float64', 'int64']:
                input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))
            else:
                input_data[col] = st.selectbox(f"{col}", options=sorted(df[col].unique()))
        if st.button("Predict Risk Label"):
            try:
                input_df = pd.DataFrame([input_data])
                pred_risk = clf.predict(input_df)[0]
                st.success(f"🎯 Predicted Risk Category: **{le_risk.inverse_transform([pred_risk])[0]}**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

elif page == "📈 Finance ML Report Generator":
    # --- FINANCE ML REPORT GENERATOR (from app.py) ---
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
        df_model = df.dropna(subset=features + [target])
        X = df_model[features]
        y = df_model[target]
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
        rf_all = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_all.fit(X, y)
        df_model['Predicted Net Income'] = rf_all.predict(X)
        st.write("## 🔎 Filter Financial Report by Company")
        company_list = df_model['Company'].unique()
        selected_company = st.selectbox("Select a company for financial report:", company_list)
        filtered_df = df_model[df_model['Company'] == selected_company]
        st.dataframe(filtered_df[['Year', 'Company', 'Revenue', 'Gross Profit', 'EBITDA', 'Predicted Net Income', 'Net Income']])
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
        st.write("## 🔮 LSTM Net Income Forecast (Single Company)")
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
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_train_seq, y_train_seq, epochs=20, verbose=0)
                y_pred_seq = model.predict(X_test_seq)
                y_pred_seq_inv = scaler.inverse_transform(y_pred_seq)
                y_test_seq_inv = scaler.inverse_transform(y_test_seq)
                fig2, ax2 = plt.subplots()
                ax2.plot(range(len(y_test_seq_inv)), y_test_seq_inv, label='Actual')
                ax2.plot(range(len(y_pred_seq_inv)), y_pred_seq_inv, label='Predicted')
                ax2.set_title("LSTM Net Income Forecast")
                ax2.legend()
                st.pyplot(fig2)
        else:
            st.info("Not enough data points for LSTM forecasting (need >5 years per company).")

