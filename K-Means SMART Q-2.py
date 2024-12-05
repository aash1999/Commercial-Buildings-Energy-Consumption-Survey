# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = "/content/cleaned_data (1).csv"
df = pd.read_csv(file_path)

# Define the target variable
target_variable = 'Electricity_Consumption'

# Ensure the target variable exists
if target_variable not in df.columns:
    raise ValueError(f"Target variable '{target_variable}' not found in the dataset.")

# Handle missing values in the target variable
df[target_variable] = pd.to_numeric(df[target_variable].fillna(df[target_variable].mean()))

# Remove unnecessary columns if any exist (adjust filtering based on your dataset needs)
# Example of removing columns with too many missing values or irrelevant features
missing_percentage = (df.isnull().sum() / len(df)) * 100
df = df.loc[:, missing_percentage <= 20]  # Retain columns with <= 20% missing data

# Use SimpleImputer to fill missing values for all features
imputer = SimpleImputer(strategy='mean')  # Use 'median' if suitable
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Separate features and target
X = df_imputed.drop(columns=[target_variable, "Unnamed: 0"])  # Drop irrelevant or index-like columns
y = df_imputed[target_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a Random Forest model to calculate feature importance
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Extract feature importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Select columns with importance > 0.01
selected_features = feature_importances[feature_importances['Importance'] > 0.01]['Feature'].tolist()
X_selected = X[selected_features]

# Scale the selected features for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Example with 3 clusters
df_imputed['Cluster'] = kmeans.fit_predict(X_scaled)

# Analyze characteristics of each cluster
cluster_summary = df_imputed.groupby('Cluster').mean()

# Print cluster characteristics
print("Cluster Characteristics:")
print(cluster_summary)

# Visualize clusters
sns.boxplot(data=df_imputed, x='Cluster', y=target_variable)
plt.title("Electricity Consumption by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Annual Electricity Consumption (2018)")
plt.show()

# Save clustered data for further analysis
df_imputed.to_csv('/content/cleaned_data (1).csv', index=False)
