#%%
import pandas as pd
import numpy as np

#%%
#read data file
df = pd.read_csv("cbecs2018.csv")

#drop columns after FINALWT as not meaningful variables
df.drop(df.columns[df.columns.get_loc('FINALWT'):], axis=1, inplace=True)
# %%
#Cleaning missing data
#calculate missing percentage for each column
missing_percentage = df.isnull().mean() * 100

#we want to see how many columns have more than 20 percent of data missing
columns_missing_values = missing_percentage[missing_percentage > 20]
print(f"Number of columns with more than 20% missing values: {len(columns_missing_values)}")

#remove columns with over 20 percent of data missing
df_dropped = df.drop(columns = columns_missing_values.index)

# %%
#Fill in missing values with median
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
df_cleaned = pd.DataFrame(imputer.fit_transform(df_dropped))
df_cleaned.columns = df_dropped.columns
df_cleaned.index = df_dropped.index

#%%
#Remove other annual values
df_cleaned = df_cleaned.drop("MFEXP", axis = 1)
df_cleaned = df_cleaned.drop("MFBTU", axis = 1)
df_cleaned.drop(df_cleaned.columns[df_cleaned.columns.get_loc('ELCNS'):], axis=1, inplace=True)

#%%
#Create new dependent variable EUI - ELBTU/SQFT
df_cleaned['EUI'] = df_cleaned['ELBTU']/df_cleaned['SQFT']

#%%
#Since we added new dependent variable, remove ELBTU
df_cleaned = df_cleaned.drop("ELBTU", axis = 1)

#%%
#We have column ELUSED, which indicates if buidling uses electricity. We want his as 1.
df_cleaned = df_cleaned[df_cleaned['ELUSED'] == 1]

#%%
df_cleaned = df_cleaned.drop("PUBID", axis = 1)

#%%
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

def feature_selection_vif(df, target_variable, k_best=10):
    """
    Perform feature selection using SelectKBest and calculate VIF for selected features.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame with independent variables.
    target_variable (str): The name of the target variable.
    k_best (int): The number of top features to select (default is 10).
    
    Returns:
    selected_features_df (pd.DataFrame): DataFrame with selected features and their scores.
    vif_df (pd.DataFrame): DataFrame with selected features and their VIF values.
    """
    # Separate features (X) and target (y)
    X = df.drop(target_variable, axis=1)
    y = df[target_variable]

    # Apply SelectKBest with mutual_info_regression
    selector = SelectKBest(mutual_info_regression, k=k_best)
    X_new = selector.fit_transform(X, y)

    # Create DataFrame with selected features
    selected_features = X.columns[selector.get_support()]
    df_selected = pd.DataFrame(X_new, columns=selected_features)

    # Get the scores of the selected features
    selected_scores = selector.scores_[selector.get_support()]
    df_selected_scores = pd.DataFrame({
        'Feature': selected_features,
        'Score': selected_scores
    })

    # Sort the selected features by score
    df_selected_scores = df_selected_scores.sort_values(by='Score', ascending=False)

    # Calculate VIF for the selected features
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_selected.columns
    vif_data["VIF"] = [variance_inflation_factor(df_selected.values, i) for i in range(len(df_selected.columns))]

    return df_selected_scores, vif_data

#%%

df_selected_scores, vif_data = feature_selection_vif(df_cleaned, target_variable='EUI', k_best=10)

# Display the selected features and their VIF values
print("Selected Features and Scores:")
print(df_selected_scores)

print("\nVIF for Selected Features:")
print(vif_data)

#%%
#Remove repetitive columns
df_cleaned = df_cleaned.drop("PBAPLUS", axis = 1)
df_cleaned = df_cleaned.drop("WKHRSC", axis = 1)
df_cleaned = df_cleaned.drop("NWKERC", axis = 1)
df_cleaned = df_cleaned.drop("PCTRMC", axis = 1)


#%%

df_selected_scores, vif_data = feature_selection_vif(df_cleaned, target_variable='EUI', k_best=10)

# Display the selected features and their VIF values
print("Selected Features and Scores:")
print(df_selected_scores)

print("\nVIF for Selected Features:")
print(vif_data)


#%%
#Remove high vif
df_cleaned = df_cleaned.drop("LTOHRP", axis = 1)
df_cleaned = df_cleaned.drop("RFGWI", axis = 1)

#remove repetitive column
df_cleaned = df_cleaned.drop("LNHRPC", axis = 1)

#%%

df_selected_scores, vif_data = feature_selection_vif(df_cleaned, target_variable='EUI', k_best=10)

# Display the selected features and their VIF values
print("Selected Features and Scores:")
print(df_selected_scores)

print("\nVIF for Selected Features:")
print(vif_data)


#%%
#Remove high vif
df_cleaned = df_cleaned.drop("RFGICE", axis = 1)


#%%

df_selected_scores, vif_data = feature_selection_vif(df_cleaned, target_variable='EUI', k_best=10)

# Display the selected features and their VIF values
print("Selected Features and Scores:")
print(df_selected_scores)

print("\nVIF for Selected Features:")
print(vif_data)

#%%
#Remove high vif
df_cleaned = df_cleaned.drop("RFGCL", axis = 1)


#%%

df_selected_scores, vif_data = feature_selection_vif(df_cleaned, target_variable='EUI', k_best=10)

# Display the selected features and their VIF values
print("Selected Features and Scores:")
print(df_selected_scores)

print("\nVIF for Selected Features:")
print(vif_data)



# %%
df_selected_scores
# %%
import matplotlib.pyplot as plt

# Select the top 10 features
top_features = df_selected_scores.head(10)

# Plot a horizontal bar chart
plt.figure(figsize=(8, 6))
plt.barh(top_features['Feature'], top_features['Score'], color='darkkhaki')
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Top 10 Important Features")
plt.gca().invert_yaxis()  # Invert y-axis to show highest score at the top
plt.tight_layout()
plt.show()
# %%


# Plot the histogram for EUI
plt.figure(figsize=(8, 6))
plt.hist(df_cleaned['EUI'], bins=30, color='darkkhaki', edgecolor='black')
plt.xlabel("EUI (Energy Use Intensity)")
plt.ylabel("Frequency")
plt.title("Distribution of EUI")
plt.grid(axis='y', alpha=0.75)  # Add gridlines for better readability
plt.tight_layout()
plt.show()

#%%
# Create a new Data frame with the selected features and the dependent variables 'EUI'
# List of features to keep
features_to_keep = ['EUI','PBA', 'WKHRS', 'LTNHRP', 'NWKER','PCTERMN','COOLP','MAINHT','MAINCL','FLCEILHT','NOCC']

# Filter DataFrame by selecting only rows where 'Feature' is in the list
df_cleaned = df_cleaned[features_to_keep]

#df_cleaned.insert(0, 'EUI', df_cleaned['EUI'])
print(df_cleaned.head())
# %%
# Rename the features
df_cleaned = df_cleaned.rename(columns={'EUI': 'Energy_Use_Intensity'})
df_cleaned = df_cleaned.rename(columns={'PBA': 'Building_Activity'})
df_cleaned = df_cleaned.rename(columns={'WKHRS': 'Work_Hours'})
df_cleaned = df_cleaned.rename(columns={'LTNHRP': 'Percent_Lit_Off_Hours'})
df_cleaned = df_cleaned.rename(columns={'NWKER': 'Number_Workers'})
df_cleaned = df_cleaned.rename(columns={'PCTERMN': 'Number_Desktops'})
df_cleaned = df_cleaned.rename(columns={'COOLP': 'Percent_Cooled'})
df_cleaned = df_cleaned.rename(columns={'MAINHT': 'Main_Heat_Equip'})
df_cleaned = df_cleaned.rename(columns={'MAINCL': 'Main_Cool_Equp'})
df_cleaned = df_cleaned.rename(columns={'FLCEILHT': 'Floor_Ceiling_Height'})
df_cleaned = df_cleaned.rename(columns={'NOCC': 'Number_business'})

print(df_cleaned)

#%%
df_cleaned.to_csv('cleaned_data3.csv', index=False)


# %%
import seaborn as sns

codebook_df = pd.read_csv("PBA Codebook.csv")

# Merge the main dataset with the codebook on the encoded activity column
merged_df = df_cleaned.merge(codebook_df, left_on='Building_Activity', right_on='PBA ID', how='left')

# Now, use the decoded 'Activity_Name' column for plotting
top_10_activities = merged_df['Name'].value_counts().head(10)

# Plotting the distribution of the top 10 building activities by frequency as horizontal bars
plt.figure(figsize=(10, 6))
sns.barplot(y=top_10_activities.index, x=top_10_activities.values, color='darkkhaki', orient='h', edgecolor='black')

# Adding labels and title
plt.title('Top 10 Building Activities by Frequency')
plt.ylabel('Building Activity')
plt.xlabel('Number of Buildings')

# Display the plot
plt.show()




# %%
# Count the occurrences of each Main Cooling Equipment
top_cool_equipment = df_cleaned['Main_Cool_Equp'].value_counts().head(5)

codebook_df2 = pd.read_csv("MAINCL Codebook.csv")

merged_df2 = df_cleaned.merge(codebook_df2[['Code', 'Name']], left_on='Main_Cool_Equp', right_on='Code', how='left')

# Now, use the decoded 'Activity_Name' column for plotting
top_cool_equipment = merged_df2['Name'].value_counts().head(5)

# Plotting horizontal bar chart for top 10 Main Cooling Equipment
plt.figure(figsize=(10, 6))
top_cool_equipment.plot(kind='barh', color='paleturquoise', edgecolor='black')
plt.title('Top 5 Main Cooling Equipment')
plt.xlabel('Frequency')
plt.ylabel('Main Cooling Equipment')
plt.gca().invert_yaxis()  # To display the most frequent at the top
plt.show()
# %%
# Plotting histogram for Work_Hours
plt.figure(figsize=(8, 6))
plt.hist(df_cleaned['Work_Hours'], bins=20, color='paleturquoise', edgecolor='black')
plt.title('Distribution of Total hours open per week')
plt.xlabel('Total hours open per week')
plt.ylabel('Frequency')
plt.show()

# Plotting histogram for Number_Workers
plt.figure(figsize=(8, 6))
plt.hist(df_cleaned['Number_Workers'], bins=20, color='darkkhaki', edgecolor='black')
plt.title('Distribution of Number of Workers')
plt.xlabel('Number of Workers')
plt.ylabel('Frequency')
plt.show()

# %%
