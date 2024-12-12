# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# %%
#read data file
df = pd.read_csv("cbecs2018.csv")


#%%
#drop columns after FINALWT as not meaningful variables
df.drop(df.columns[df.columns.get_loc('FINALWT'):], axis=1, inplace=True)


# %%
#remove all other energy types and irrelevant columns
df_drop = df.drop("MFEXP", axis = 1)
df_drop = df_drop.drop("MFBTU", axis = 1)
df_drop.drop(df_drop.columns[df_drop.columns.get_loc('ELCNS'):], axis=1, inplace=True)


# %%
#make our new dependent variable EUI - we want fair comparison between big and small buildings
df_drop['EUI'] = df_drop['ELBTU']/df_drop['SQFT']
#Since we added new dependent variable, remove ELBTU
df_drop = df_drop.drop("ELBTU", axis = 1)


# %%
#We have column ELUSED, which indicates if buidling uses electricity. We want this as 1.
df_drop = df_drop[df_drop['ELUSED'] == 1]


# %%
#remove id column
df_drop = df_drop.drop("PUBID", axis = 1)


# %%
#Cleaning missing data
#calculate missing percentage for each column
missing_percentage = df_drop.isnull().mean() * 100

#we want to see how many columns have more than 20 percent of data missing
columns_missing_values = missing_percentage[missing_percentage > 20]
print(f"Number of columns with more than 20% missing values: {len(columns_missing_values)}")

#remove columns with over 20 percent of data missing
df_drop = df_drop.drop(columns = columns_missing_values.index)


# %%
#Fill in missing values with median
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
df_cleaned = pd.DataFrame(imputer.fit_transform(df_drop))
df_cleaned.columns = df_drop.columns
df_cleaned.index = df_drop.index


# %%
#make data for top 5 activities
top_activities_data = df_cleaned.copy()

# Mapping for PBA values to names
pba_mapping = {
    1: 'Vacant',
    2: 'Office',
    4: 'Laboratory',
    5: 'Nonrefrigerated warehouse',
    6: 'Food sales',
    7: 'Public order and safety',
    8: 'Outpatient health care',
    11: 'Refrigerated warehouse',
    12: 'Religious worship',
    13: 'Public assembly',
    14: 'Education',
    15: 'Food service',
    16: 'Inpatient health care',
    17: 'Nursing',
    18: 'Lodging',
    23: 'Strip shopping center',
    24: 'Enclosed mall',
    25: 'Retail other than mall',
    26: 'Service',
    91: 'Other'
}

# Apply mapping to the 'PBA' column
top_activities_data['PBA'] = top_activities_data['PBA'].map(pba_mapping)

# Top 10 most frequent building activities
top_activities = top_activities_data['PBA'].value_counts().head(5).index

# %%
# Create a figure with two side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Distribution Plot by Building Activity
sns.countplot(data=top_activities_data, 
              y='PBA', 
              order=top_activities,
              ax=axes[0])
axes[0].set_title('Distribution of Top 5 Building Activities')
axes[0].set_xlabel('Count')
axes[0].set_ylabel('Building Activity')

# Box Plot for EUI by Building Activity
sns.boxplot(data=top_activities_data, 
            y='PBA', 
            x='EUI', 
            order=top_activities, 
            ax=axes[1],
            showfliers=False)
axes[1].set_title('EUI Distribution by Building Activity')
axes[1].set_xlabel('Energy Use Intensity (EUI)')

# Remove the y-axis label
axes[1].set_ylabel('')  # Remove y-axis label
axes[1].tick_params(axis='y', labelleft=False)


plt.tight_layout()
plt.show()



# %%
# Define the mapping for YRCONC as a dictionary
yrconc_mapping = {
    2: 'Before 1946',
    3: '1946 to 1959',
    4: '1960 to 1969',
    5: '1970 to 1979',
    6: '1980 to 1989',
    7: '1990 to 1999',
    8: '2000 to 2012',
    9: '2013 to 2018'
}

# Map the YRCONC values 
yr_data = df_cleaned.copy()
yr_data['YRCONC'] = yr_data['YRCONC'].map(yrconc_mapping)
year_order = ['Before 1946', '1946 to 1959', '1960 to 1969', '1970 to 1979', 
              '1980 to 1989', '1990 to 1999', '2000 to 2012', '2013 to 2018']

#%%
# Create a figure with side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Distribution Plot by Year Constructed
sns.countplot(data=yr_data, 
              y='YRCONC', 
              ax=axes[0],
              order = year_order),

axes[0].set_title('Distribution of Buildings by Year Constructed')
axes[0].set_xlabel('Number of Buildings')
axes[0].set_ylabel('Year Constructed')


# Box Plot for EUI by Year Constructed
sns.boxplot(data=yr_data, 
            y='YRCONC', 
            x='EUI', 
            ax=axes[1],
            showfliers=False,
            order = year_order)
axes[1].set_title('EUI Distribution by Year Constructed')
axes[1].set_xlabel('Energy Use Intensity (EUI)')

# Remove the y-axis label
axes[1].set_ylabel('')  # Remove y-axis label
axes[1].tick_params(axis='y', labelleft=False)


plt.tight_layout()
plt.show()


#%%
# Perform the ANOVA 
# Fit the OLS model
anova_model = ols('EUI ~ C(YRCONC)', data=yr_data).fit()

# Perform the ANOVA (Type II sum of squares)
anova_table = sm.stats.anova_lm(anova_model, typ=2)

# Extract p-value and F-statistic
p_value = anova_table['PR(>F)'][0]
f_statistic = anova_table['F'][0]

print(f"P-value: {p_value}")
print(f"F-statistic: {f_statistic}\n")

#%%
# Tukey's HSD test
tukey = pairwise_tukeyhsd(endog=df_cleaned['EUI'], groups=df_cleaned['YRCONC'], alpha=0.05)
print(tukey.summary())


#%%
import scipy.stats as stats

# Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_cleaned, x='CDD65', y='EUI')

# Set plot titles and labels
plt.title('Cooling Degree Days vs EUI')
plt.xlabel('Cooling Degree Days')
plt.ylabel('EUI')

# Show plot
plt.show()

# %%
# Perform correlation test (Pearson correlation)
corr, p_value = stats.pearsonr(df_cleaned['CDD65'], df_cleaned['EUI'])

# Print the result
print(f'Pearson Correlation Coefficient: {corr}')
print(f'P-value: {p_value}')





# %%
df_rf = df_cleaned

# Assign X and y
X = df_rf.drop(columns=['EUI'])
y = df_rf['EUI']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# Random forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
rf_pred = rf_model.predict(X_test)

# Calculate Mean Squared Error for the Random Forest model
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)  # R-squared for Random Forest model
print(f'Random Forest Model MSE: {rf_mse}')
print(f'Random Forest Model R-squared: {rf_r2}')


# %%
# Feature importance function
feature_importances = rf_model.feature_importances_

features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the features by importance and get top 20
features_df = features_df.sort_values(by='Importance', ascending=False)
top_20_features = features_df.head(20)

# Plot the top 10 important features
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_20_features)
plt.title('Top 20 Important Features (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# %%
# List of features sorted from most to least important
sorted_importances = features_df['Importance'].values
sorted_features = features_df['Feature'].values

# Cumulative importances
cumulative_importances = np.cumsum(sorted_importances)

# Plot the cumulative importances
plt.figure(figsize=(6, 6))
plt.plot(cumulative_importances, 'g-')

# Axis labels and title
plt.xlabel('Feature Index')
plt.ylabel('Cumulative Importance')
plt.title('Cumulative Feature Importances for Random Forest Model')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

# Calculate the "elbow" point - where the change in cumulative importance becomes small
# This can be done by checking the second derivative of the cumulative importances
diff_cumulative_importances = np.diff(cumulative_importances)
diff_diff_cumulative_importances = np.diff(diff_cumulative_importances)


# %%
# Set the threshold for cumulative importance (60%)
threshold = 0.70
optimal_features_index = np.argmax(cumulative_importances >= threshold) + 1

# Select the top features based on the 70% threshold
top_features = sorted_features[:optimal_features_index]

# Create a new dataset with only the selected top features and the target variable
X_selected = X[top_features]
selected_df = pd.concat([X_selected, y], axis=1)

# Display the new dataset
print(f"New dataset with the top features (70% cumulative importance) and target variable:")
print(selected_df)



# %%
# Assuming `new_dataset` contains the top features and target variable
X_reduced = selected_df.drop(columns=['EUI', 'PBAPLUS', 'OTCLEQ'])
#X_reduced = selected_df.drop(columns=['EUI'])
y_reduced = selected_df['EUI']

# Split the new dataset into training and testing sets
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
    X_reduced, y_reduced, test_size=0.2, random_state=42
)

# Initialize the RandomForestRegressor
rf_model_reduced = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the Random Forest model on the reduced dataset
rf_model_reduced.fit(X_train_reduced, y_train_reduced)

# Make predictions on the test set
y_pred_reduced = rf_model_reduced.predict(X_test_reduced)

# Evaluate the model
mse_reduced = mean_squared_error(y_test_reduced, y_pred_reduced)
r2_reduced = r2_score(y_test_reduced, y_pred_reduced)

print(f"Mean Squared Error (Reduced Dataset): {mse_reduced}")
print(f"R-squared (Reduced Dataset): {r2_reduced}")


# %%
#cross-validation

# Define k-fold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

rf_model_cross = RandomForestRegressor(n_estimators=100, random_state=42)

mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

mse_scores = cross_val_score(rf_model_cross, X_reduced, y_reduced, scoring=mse_scorer, cv=kfold)

mse_scores = -mse_scores

r2_scores = cross_val_score(rf_model_cross, X_reduced, y_reduced, scoring='r2', cv=kfold)

# Print mean and standard deviation for both metrics
print(f"Cross-validated Mean Squared Error: {np.mean(mse_scores):.3f} ± {np.std(mse_scores):.3f}")
print(f"Cross-validated R-squared: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")



# %%
# Get feature importances
importances = rf_model_reduced.feature_importances_
features = X_reduced.columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot the top 10 important features
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Important Features')
plt.gca().invert_yaxis()  # Reverse the order to have the most important on top
plt.show()


# %%
# Get feature importances

# Create a DataFrame with feature names and their corresponding importance
feature_importance_df = pd.DataFrame({
    'Feature': X_reduced.columns,
    'Importance': importances
})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the sorted feature importance
print(feature_importance_df)








# %%
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_df)  # Scale only the selected columns

# List to store the average silhouette scores for different values of k
silhouette_scores = []

# Try clustering with k values from 2 to 10
k_values = range(2, 10)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(selected_df)
    
    # Calculate the silhouette score for the current k
    score = silhouette_score(selected_df, kmeans.labels_)
    silhouette_scores.append(score)

# Plot the silhouette scores for each k value
plt.figure(figsize=(8, 6))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different k Values')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(k_values)
plt.show()


#%%%

kmeans = KMeans(n_clusters=4, max_iter=300, random_state=42)
kmeans.fit(scaled_data) 

# Add the cluster labels to the original DataFrame
selected_df['kmeans_4'] = kmeans.labels_  

print(selected_df.head)
#%%


# Perform PCA to reduce the data to 2D
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)

# Add the PCA components to the dataframe for plotting
selected_df['pca1'] = pca_components[:, 0]
selected_df['pca2'] = pca_components[:, 1]

# Create a scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='pca1', y='pca2', hue='kmeans_4', palette='Set1', data=selected_df, s=100, alpha=0.7, edgecolor='k')
plt.title('KMeans Clusters Visualized with PCA')
plt.legend(title='Cluster')
plt.show()

#%%
plt.scatter(x=selected_df['WKHRS'], y=selected_df['EUI'], c=selected_df['kmeans_4'])
plt.xlabel('WKHRS')
plt.ylabel('EUI')

#%%


kmeans = KMeans(n_clusters=4, random_state=42)
selected_df['kmeans_4'] = kmeans.fit_predict(selected_df[['EUI', 'WKHRS']])

#%%
# Group by the cluster and calculate summary statistics
summary = selected_df.groupby('kmeans_4').agg({
    'EUI': ['mean', 'median', 'std'],
    'WKHRS': ['mean', 'median', 'std'],
    'SQFT': ['mean', 'median', 'std'],
    'RFGWI': ['mean', 'median', 'std'],
    'PBA': ['mean', 'median', 'std'],
    'NWKER': ['mean', 'median', 'std'],
    'OWNTYPE': ['mean', 'median', 'std'],
    'COOLP': ['mean', 'median', 'std'],
    'CDD65': ['mean', 'median', 'std'],
    'HDD65': ['mean', 'median', 'std'],
    'OTCLEQ_EL': ['mean', 'median', 'std'],
    'PCTERMN': ['mean', 'median', 'std'],
    'SCHED': ['mean', 'median', 'std'],
    'LTNHRP': ['mean', 'median', 'std'],
    'MAINHT': ['mean', 'median', 'std'],
    'NFLOOR': ['mean', 'median', 'std'],
    'FLCEILHT': ['mean', 'median', 'std'],
    'YRCONC': ['mean', 'median', 'std']
})

# Display the summary
print(summary)



#%%


custom_palette = {'0': 'purple', '1': 'skyblue', '2': 'lightgreen', '3': 'orange'}

plt.figure(figsize=(12, 6)) 
sns.boxplot(x='kmeans_4', y='EUI', data=selected_df, palette=custom_palette)
plt.title('Boxplot of WKHRS by Cluster')
plt.xlabel('Cluster')
plt.ylabel('WKHRS')
plt.grid(True)
plt.show()

# %%


custom_palette = {'0': 'purple', '1': 'skyblue', '2': 'lightgreen', '3': 'orange'}

plt.figure(figsize=(12, 6)) 
sns.boxplot(x='kmeans_4', y='EUI', data=selected_df, palette=custom_palette)
plt.title('Boxplot of EUI by Cluster')
plt.xlabel('Cluster')
plt.ylabel('EUI')
plt.grid(True)
plt.show()


# %%


custom_palette = {'0': 'purple', '1': 'skyblue', '2': 'lightgreen', '3': 'orange'}

plt.figure(figsize=(12, 6)) 
sns.boxplot(x='kmeans_4', y='WKHRS', data=selected_df, palette=custom_palette)
plt.title('Boxplot of WKHRS by Cluster')
plt.xlabel('Cluster')
plt.ylabel('WKHRS')
plt.grid(True)
plt.show()
#%%

custom_palette = {'0': 'purple', '1': 'skyblue', '2': 'lightgreen', '3': 'orange'}
plt.figure(figsize=(12, 6))  
sns.boxplot(x='kmeans_4', y='SQFT', data=selected_df,palette=custom_palette)
plt.title('Boxplot of SQFT by Cluster')
plt.xlabel('Cluster')
plt.ylabel('SQFT')
plt.grid(True)
plt.show()
# %%

grouped_data = [selected_df[selected_df['kmeans_4'] == cluster]['WKHRS'] for cluster in selected_df['kmeans_4'].unique()]
anova_result = stats.f_oneway(*grouped_data)
print("ANOVA Test:")
print(f"F-statistic: {anova_result.statistic}")
print(f"P-value: {anova_result.pvalue}")


# %%

grouped_data = [selected_df[selected_df['kmeans_4'] == cluster]['SQFT'] for cluster in selected_df['kmeans_4'].unique()]
anova_result = stats.f_oneway(*grouped_data)
print("ANOVA Test:")
print(f"F-statistic: {anova_result.statistic}")
print(f"P-value: {anova_result.pvalue}")


#%%

grouped_data = [selected_df[selected_df['kmeans_4'] == cluster]['EUI'] for cluster in selected_df['kmeans_4'].unique()]
anova_result = stats.f_oneway(*grouped_data)
print("ANOVA Test:")
print(f"F-statistic: {anova_result.statistic}")
print(f"P-value: {anova_result.pvalue}")