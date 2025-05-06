import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv('tax_risk_dataset.csv')

# 2. Data Cleaning
# Drop non-informative columns
df = df.drop(['Taxpayer_ID', 'Audit_to_Tax_Ratio'], axis=1)

# Encode categorical variables
le_industry = LabelEncoder()
df['Industry'] = le_industry.fit_transform(df['Industry'])

le_risk = LabelEncoder()
df['Risk_Label'] = le_risk.fit_transform(df['Risk_Label'])

# Remove duplicates if any
df = df.drop_duplicates()

# Check missing values (optional print)
print("Missing values per column:\n", df.isnull().sum())

# 3. Exploratory Data Analysis (EDA)
print("\nSummary statistics:\n", df.describe())

# Plot Risk_Label distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Risk_Label', data=df)
plt.title('Risk Label Distribution')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# Boxplot of Revenue by Risk_Label
plt.figure(figsize=(10,6))
sns.boxplot(x='Risk_Label', y='Revenue', data=df)
plt.title('Revenue Distribution by Risk Label')
plt.show()

# 4. Prepare features and targets

# For tax filing prediction (regression)
X_filing = df.drop(['Tax_Liability', 'Risk_Label'], axis=1)
y_filing = df['Tax_Liability']

# For tax risk prediction (classification)
X_risk = df.drop(['Risk_Label'], axis=1)
y_risk = df['Risk_Label']

# 5. Split into train and test sets
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_filing, y_filing, test_size=0.2, random_state=42)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_risk, y_risk, test_size=0.2, stratify=y_risk, random_state=42)

# 6. Train and evaluate models

# 6a. Tax Risk Classification with Random Forest
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_r, y_train_r)
y_pred_r = clf.predict(X_test_r)
print("\nClassification Report for Tax Risk Prediction:")
print(classification_report(y_test_r, y_pred_r, target_names=le_risk.classes_))

# 6b. Tax Liability Regression with Random Forest Regressor
reg = RandomForestRegressor(random_state=42)
reg.fit(X_train_f, y_train_f)
y_pred_f = reg.predict(X_test_f)

mse = mean_squared_error(y_test_f, y_pred_f)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.2f}")

print(f"\nRegression Metrics for Tax Liability Prediction:\nRMSE: {rmse:.2f}\nR2")

# Optional: Feature importance plots
importances_clf = clf.feature_importances_
features_risk = X_risk.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances_clf, y=features_risk)
plt.title('Feature Importance - Tax Risk Model')
plt.show()

importances_reg = reg.feature_importances_
features_filing = X_filing.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances_reg, y=features_filing)
plt.title('Feature Importance - Tax Liability Model')
plt.show()
