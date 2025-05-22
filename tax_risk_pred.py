import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report
import io
from xgboost import XGBClassifier, XGBRegressor


# Streamlit configuration
st.set_page_config(page_title="Tax Risk Analysis Dashboard", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('tax_risk_dataset.csv')

    # Drop unused columns
    df = df.drop(['Taxpayer_ID', 'Audit_to_Tax_Ratio'], axis=1)
    df = df.drop_duplicates()

    # Reduce 'High' labels by 50%
    high_risk_df = df[df['Risk_Label'] == 'High']
    df = df.drop(high_risk_df.sample(frac=0.5, random_state=42).index)

    return df

df = load_data()

# Encode categorical variables
le_industry = LabelEncoder()
df['Industry'] = le_industry.fit_transform(df['Industry'])

le_risk = LabelEncoder()
df['Risk_Label'] = le_risk.fit_transform(df['Risk_Label'])

# Feature-target split
X_filing = df.drop(['Tax_Liability', 'Risk_Label'], axis=1)
y_filing = df['Tax_Liability']

X_risk = df.drop(['Risk_Label'], axis=1)
y_risk = df['Risk_Label']

# Train/test split
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_filing, y_filing, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_risk, y_risk, test_size=0.2, stratify=y_risk, random_state=42)

# Train models
# Train models using XGBoost
clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
clf.fit(X_train_r, y_train_r)

reg = XGBRegressor(random_state=42)
reg.fit(X_train_f, y_train_f)

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["🏁 Overview", "📊 EDA", "📈 Feature Importance", "🔍 Predict Tax Risk"])


# 1. Overview Section

if option == "🏁 Overview":
    st.title("📊 Tax Risk Analysis Dashboard")
    st.markdown("""
    Welcome to the interactive Tax Risk Analysis Dashboard.  
    This tool uses machine learning to analyze taxpayer data and:
    - Predict risk categories (Low, Medium, High)
    - Estimate tax liabilities  
      
    Navigate using the sidebar to explore the data and try live predictions!
    """)


# 2. EDA Section

elif option == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    # Add Risk_Label Name for readable plots
    df['Risk_Label_Name'] = le_risk.inverse_transform(df['Risk_Label'])

    st.subheader("Scatter Plot: Revenue vs Profit with Risk Label")

    # Create numerical codes for the Risk Labels
    risk_codes = df['Risk_Label']  # Assuming Risk_Label is already encoded
    label_names = le_risk.classes_  # Get the human-readable labels from the encoder

    fig0, ax0 = plt.subplots(figsize=(10, 6))

    # Create scatter plot
    scatter = ax0.scatter(
        df['Revenue'],
        df['Profit'],
        c=risk_codes,
        cmap='coolwarm',   # Or try 'viridis', 'plasma', etc.
        alpha=0.7
    )

    # Create legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=label,
               markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=8)
        for i, label in enumerate(label_names)
    ]
    ax0.legend(handles=legend_elements, title="Risk Label")

    ax0.set_xlabel("Revenue")
    ax0.set_ylabel("Profit")
    ax0.set_title("Revenue vs Profit Colored by Risk Label")
    ax0.grid(True, linestyle='--', alpha=0.6)

    st.pyplot(fig0)




    st.subheader("Correlation Heatmap")
    df_numeric = df.select_dtypes(include=[float, int])

    fig, ax = plt.subplots(figsize=(10, 8))  # You can adjust the size
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



# 3. Feature Importance

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


# # 4. Model Evaluation

# elif option == "🧪 Model Metrics":
#     st.title("🧪 Model Performance")

#     # Classification
#     y_pred_risk = clf.predict(X_test_r)
#     report = classification_report(y_test_r, y_pred_risk, target_names=le_risk.classes_, output_dict=True)
#     report_df = pd.DataFrame(report).transpose()
#     st.subheader("Classification Report (Tax Risk)")
#     st.dataframe(report_df.style.format("{:.2f}"))

#     # Download option
#     csv = report_df.to_csv(index=True).encode('utf-8')
#     st.download_button("📥 Download Classification Report", csv, "classification_report.csv", "text/csv")

#     # Regression
#     st.subheader("Regression Metrics (Tax Liability)")
#     y_pred_filing = reg.predict(X_test_f)
#     rmse = np.sqrt(mean_squared_error(y_test_f, y_pred_filing))
#     r2 = reg.score(X_test_f, y_test_f)
#     st.metric(label="RMSE", value=f"₹{rmse:,.2f}")
#     st.metric(label="R² Score", value=f"{r2:.2%}")


# 5. Predict Risk Label

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


# 6. Predict Tax Liability

# elif option == "💰 Predict Tax Liability":
#     st.title("💰 Predict Tax Liability")
#     st.markdown("Estimate the **tax liability** based on user input.")

#     input_data = {}
#     for col in X_filing.columns:
#         if df[col].dtype in ['float64', 'int64']:
#             input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))
#         else:
#             input_data[col] = st.selectbox(f"{col}", options=sorted(df[col].unique()))

#     if st.button("Predict Tax Liability"):
#         try:
#             input_df = pd.DataFrame([input_data])
#             prediction = reg.predict(input_df)[0]
#             st.success(f"💵 Estimated Tax Liability: ₹{prediction:,.2f}")
#         except Exception as e:
#             st.error(f"Prediction failed: {e}")
