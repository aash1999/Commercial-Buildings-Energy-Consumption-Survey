Python 3.12.5 (v3.12.5:ff3bc82f7c9, Aug  7 2024, 05:32:06) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> # Import necessary libraries
... import pandas as pd
... import numpy as np
... import matplotlib.pyplot as plt
... import seaborn as sns
... 
... # Load the data
... file_path = '/content/cbecs2018_final_public.csv'
... df = pd.read_csv(file_path)
... 
... # Display basic information about the dataset
... print("### Basic Information")
... df.info()
... 
... print("\n### First 5 Rows of the Dataset")
... print(df.head())
... 
... # Handling missing values
... print("\n### Missing Value Analysis")
... missing_values = df.isnull().sum()
... print(missing_values[missing_values > 0].sort_values(ascending=False))
... 
... # Summary statistics for numerical columns
... print("\n### Summary Statistics")
... print(df.describe())
... 
... # Correlation matrix
... print("\n### Correlation Matrix")
... plt.figure(figsize=(15, 10))
... sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
... plt.title('Correlation Matrix of Numerical Features')
... plt.show()
... 
... # Distribution plots for selected columns (adjust these as needed)
... selected_columns = ['SQFT', 'ELCNS', 'ELEXP', 'NGBTU']  # Replace with relevant columns
... for col in selected_columns:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Count plots for categorical columns
print("\n### Count Plots for Categorical Columns")
categorical_columns = ['REGION', 'PBA']  # Replace with relevant columns
for col in categorical_columns:
    plt.figure()
    sns.countplot(data=df, x=col)
    plt.title(f'Count Plot for {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()

# Box plots for detecting outliers in numerical columns
print("\n### Box Plots for Outlier Detection")
for col in selected_columns:
    plt.figure()
    sns.boxplot(data=df, x=col)
    plt.title(f'Box Plot for {col}')
    plt.xlabel(col)
    plt.show()

# Pairplot for exploring relationships between features
print("\n### Pairplot for Selected Columns")
sns.pairplot(df[selected_columns])
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
