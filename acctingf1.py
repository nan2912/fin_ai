import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime

# SQLite DB setup
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

# Functions
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

# UI setup
st.set_page_config(page_title="Accounting App", layout="wide")
st.title("📘 Accounting Module with CSV Support")

# Sidebar navigation
menu = st.sidebar.radio("Navigation", [
    "Add Transaction",
    "Upload Dataset",
    "Load Sample Dataset",
    "View Ledger",
    "Trial Balance",
    "Export Data"
])

# 1. Add Transaction
if menu == "Add Transaction":
    st.header("➕ Add New Transaction")
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

# 2. Upload Dataset (via browser)
elif menu == "Upload Dataset":
    st.header("⬆️ Upload Transactions CSV")
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

# 3. Load Sample Dataset (hardcoded path)
elif menu == "Load Sample Dataset":
    st.header("📥 Load Sample Dataset")
    file_path = r"C:\Users\matha\Downloads\Comprehensive_Banking_Database.csv"

    try:
        df = pd.read_csv(file_path)
        st.dataframe(df.head())

        required_cols = {"date", "account", "debit", "credit", "narration"}
        if required_cols.issubset(df.columns):
            for _, row in df.iterrows():
                try:
                    add_transaction(row['date'], row['account'], float(row['debit']), float(row['credit']), str(row['narration']))
                except Exception as e:
                    st.error(f"Error in row: {row} - {e}")
            st.success("✅ Sample data loaded successfully!")
        else:
            st.error(f"CSV must contain columns: {required_cols}")
    except Exception as e:
        st.error(f"Error loading file: {e}")

# 4. View Ledger
elif menu == "View Ledger":
    st.header("📒 View Account Ledger")
    account_name = st.text_input("Enter Account Name:")
    if st.button("Show Ledger"):
        ledger_df = get_ledger(account_name)
        if ledger_df.empty:
            st.warning("No transactions found for this account.")
        else:
            st.dataframe(ledger_df)

# 5. Trial Balance
elif menu == "Trial Balance":
    st.header("📊 Trial Balance")
    trial_df = generate_trial_balance()
    total_debit = trial_df["total_debit"].sum()
    total_credit = trial_df["total_credit"].sum()

    st.dataframe(trial_df)
    st.metric("🔢 Total Debit", f"{total_debit:,.2f}")
    st.metric("🔢 Total Credit", f"{total_credit:,.2f}")

# 6. Export Data
elif menu == "Export Data":
    st.header("⬇️ Export All Transactions")
    df = get_all_transactions()
    st.dataframe(df)
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_data, "transactions.csv", "text/csv")
