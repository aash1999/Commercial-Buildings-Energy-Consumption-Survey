# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load the data
file_path = '/content/cbecs2018_final_public.csv'
df = pd.read_csv(file_path)

# Define the target variable
target_variable = 'ELCNS'

# Handle missing values in the target variable
df[target_variable] = pd.to_numeric(df[target_variable].fillna(df[target_variable].mean()))

# Remove unnecessary columns based on filtering criteria
df = df.loc[:, ~df.columns.str.startswith(('Z', 'FINAL', 'AIRHAND', 'BLR', 'DH', 'RFC', 'CW', 'M', 'LT'))]
df = df.loc[:, ~df.columns.str.contains('(OTH|FK|REN|HW|NG|WTR|RFG|PK|FA|TR|COOL|HEATP)', case=False, regex=True)]
df = df.loc[:, ~df.columns.str.endswith(('PC', 'SEAT', 'BED', 'HT', 'PR', 'WO', 'OT', 'USED', 'BTU', '1', '2'))]

# Remove columns with more than 20% missing values
missing_percentage = (df.isnull().sum() / len(df)) * 100
df = df.loc[:, missing_percentage <= 20]

# Use SimpleImputer to fill missing values for all features
imputer = SimpleImputer(strategy='mean')  # Or use 'median' if appropriate for specific columns
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Separate features and target
X = df_imputed.drop(columns=[target_variable])
y = df_imputed[target_variable]

# Fit a Random Forest model to calculate feature importance
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Extract feature importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Select columns with importance > 0.01
selected_features = feature_importances[feature_importances['Importance'] > 0.01]['Feature'].tolist()
X_selected = X[selected_features]

# Display basic information about the dataset
print("### Basic Information")
df.info()

# Display first 5 rows of the dataset
print("\n### First 5 Rows of the Dataset")
print(df.head())

# Handling missing values - After imputation
print("\n### Missing Value Analysis")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0].sort_values(ascending=False))

# Summary statistics for numerical columns
print("\n### Summary Statistics")
print(df.describe())

# Correlation matrix
print("\n### Correlation Matrix")
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Distribution plots for selected columns (adjust these as needed)
print("\n### Distribution Plots for Selected Columns")
for col in selected_features:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Count plots for categorical columns (if any selected columns are categorical)
print("\n### Count Plots for Categorical Columns")
categorical_columns = ['REGION', 'PBA']  # Adjust as needed based on the dataset
for col in categorical_columns:
    plt.figure()
    sns.countplot(data=df, x=col)
    plt.title(f'Count Plot for {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()

# Box plots for detecting outliers in numerical columns
print("\n### Box Plots for Outlier Detection")
for col in selected_features:
    plt.figure()
    sns.boxplot(data=df, x=col)
    plt.title(f'Box Plot for {col}')
    plt.xlabel(col)
    plt.show()

# Pairplot for exploring relationships between selected features
print("\n### Pairplot for Selected Columns")
sns.pairplot(df[selected_features])
plt.show()

# Groupby analysis for mean values of a column grouped by a categorical feature
print("\n### Groupby Analysis")
grouped_df = df.groupby('REGION')['ELCNS'].mean()  # Adjust as needed
print(grouped_df)

# Bar plot for the groupby analysis
grouped_df.plot(kind='bar')
plt.title('Average Annual Electricity Consumption by Region')
plt.xlabel('Region')
plt.ylabel('Average Electricity Consumption (ELCNS)')
plt.show()
