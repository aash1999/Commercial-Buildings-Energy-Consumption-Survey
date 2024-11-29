#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
df_cleaned = df_cleaned.drop("ELBTU", axis = 1)
df_cleaned.drop(df_cleaned.columns[df_cleaned.columns.get_loc('ELEXP'):], axis=1, inplace=True)

print(df_cleaned)
# %%
from sklearn.ensemble import RandomForestRegressor

# Assuming df is your DataFrame and 'target_variable' is the column you want to predict
X = df_cleaned.drop("ELCNS", axis=1)
y = df_cleaned["ELCNS"]

# Applying RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Displaying feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
print(feature_importance.sort_values(by='Importance', ascending=False))
# %%
feature_importance.sort_values(by='Importance', ascending=False).head(10)

#%%
#We can see some features are repetitive, such as NWKERC, WKHRSC, PBAPLUS and SQFTC
#drop them and run again
df_cleaned = df_cleaned.drop("NWKERC", axis = 1)
df_cleaned = df_cleaned.drop("WKHRSC", axis = 1)
df_cleaned = df_cleaned.drop("PBAPLUS", axis = 1)
df_cleaned = df_cleaned.drop("SQFTC", axis = 1)
# %%
# Assuming df is your DataFrame and 'target_variable' is the column you want to predict
X = df_cleaned.drop("ELCNS", axis=1)
y = df_cleaned["ELCNS"]

# Applying RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Displaying feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
print(feature_importance.sort_values(by='Importance', ascending=False))
# %%
feature_importance.sort_values(by='Importance', ascending=False).head(10)
# %%
#Repetitive: Coolp, open24, and pubid
df_cleaned = df_cleaned.drop("COOLP", axis = 1)
df_cleaned = df_cleaned.drop("OPEN24", axis = 1)
df_cleaned = df_cleaned.drop("PUBID", axis = 1)
# %%
# Assuming df is your DataFrame and 'target_variable' is the column you want to predict
X = df_cleaned.drop("ELCNS", axis=1)
y = df_cleaned["ELCNS"]

# Applying RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Displaying feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
print(feature_importance.sort_values(by='Importance', ascending=False))
# %%
feature_importance.sort_values(by='Importance', ascending=False).head(10)
# %%
df_cleaned = df_cleaned.drop("LTNHRP", axis = 1)
# %%
X = df_cleaned.drop("ELCNS", axis=1)
y = df_cleaned["ELCNS"]

# Applying RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Displaying feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
print(feature_importance.sort_values(by='Importance', ascending=False))
# %%
feature_importance_sorted=feature_importance.sort_values(by='Importance', ascending=False).head(10)
top_10_feature_columns = feature_importance_sorted['Feature'].head(10).tolist()
df_cleaned = df_cleaned[["ELCNS"] + top_10_feature_columns]
print(df_cleaned)

#%%
#Updating the names of the variables.
df_cleaned.rename(columns={'ELCNS': 'Electricity_Consumption','NWKER':'Number_Workers','PBA':'Building_Activity','WKHRS':'Work_hours','CDD65':'Cooling_Days','PCTERMN':'Number_desktops','WLCNS':'Wall_Construction_Material','BOILER':'Boiler','OWNTYPE':'Own_Type' ,'PUBCLIM':'Climate_Zone'}, inplace=True)
print(df_cleaned)
#df_cleaned.to_csv('cleaned_data.csv')


# %%
# 1. Statistic Summary
print(df_cleaned.info())
print(df_cleaned.describe())

#%%
# 2. **Distribution of Features**
plt.figure(figsize=(12, 10))

# Histograms for continuous variables
plt.subplot(2, 3, 1)
sns.histplot(df_cleaned['Electricity_Consumption']/1000, kde=True, color='blue',bins=30)
plt.title('Distribution of Electricity Consumption')

plt.subplot(2, 3, 2)
sns.histplot(df_cleaned['SQFT']/1000, kde=True, color='skyblue',bins=30)
plt.title('Distribution of SQFT')

plt.subplot(2, 3, 3)
sns.histplot(df_cleaned['Number_Workers'], kde=True, color='orange',bins=30)
plt.title('Distribution of Number of Workers')

plt.subplot(2, 3, 4)
sns.histplot(df_cleaned['Work_hours'], kde=True, color='green',bins=30)
plt.title('Distribution of Work hours per week')

plt.subplot(2, 3, 5)
sns.histplot(df_cleaned['Cooling_Days'], kde=True, color='red',bins=30)
plt.title('Distribution of Cooling Degree Days')

plt.subplot(2, 3, 6)
sns.histplot(df_cleaned['Number_desktops'], kde=True, color='purple',bins=30)
plt.title('Distribution of Number of desktops')

plt.tight_layout()
plt.show()


#%%

# Correlation matrix heatmap
correlation_matrix = df_cleaned.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
# %%
