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
df_cleaned = df_cleaned.drop("ELBTU", axis = 1)
df_cleaned.drop(df_cleaned.columns[df_cleaned.columns.get_loc('ELEXP'):], axis=1, inplace=True)

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
feature_importance.sort_values(by='Importance', ascending=False).head(10)
# %%
