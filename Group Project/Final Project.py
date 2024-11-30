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
#Smart Q2
# Exploring some EDAs
# Creating a bubble chart Electricity consumption, SQFT, Number_workers and boiler
# Plot the bubble chart
sns.scatterplot(
    data=df_cleaned, 
    x='Number_Workers', 
    y='SQFT', 
    size= ('Electricity_Consumption'),
    sizes=(50, 1000), 
    palette='viridis', 
    hue='Boiler', 
    legend=True
)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# Show the plot
plt.show()

# %%
# Creating a bubble chart Electricity consumption, SQFT, Number_workers and boiler
# Plot the bubble chart
sns.scatterplot(
    data=df_cleaned, 
    x='Number_Workers', 
    y='Work_hours', 
    size= ('Electricity_Consumption'),
    sizes=(50, 1000), 
    palette='Spectral', 
    hue='Boiler', 
    legend=True
)


plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# Show the plot
plt.show()
# %%
# %%
# Creating a bubble chart Electricity consumption, SQFT, Number_workers and boiler
# Plot the bubble chart
df_cleaned['Building_Activity'] = df_cleaned['Building_Activity'].astype('category')
#Create a mapping for Building_Activity
building_activity_map = {
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

df_cleaned['Building_Activity'] = df_cleaned['Building_Activity'].replace(building_activity_map)


sns.scatterplot(
    data=df_cleaned, 
    x='Number_Workers', 
    y='Electricity_Consumption', 
    size= ('SQFT'),
    sizes=(50, 1000), 
    palette='Set2', 
    hue='Building_Activity', 
    legend=True
)


plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# Show the plot
plt.show()


# %%

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Function to format y-axis in millions
def millions(x, pos):
    return f'{x / 1_000_000:.1f}M'

# Create a pivot table (this part remains the same)
df_pivot = df_cleaned.pivot_table(index='Boiler', columns='Building_Activity', values='Electricity_Consumption', aggfunc='sum', fill_value=0)

# Calculate the sum of each 'Building_Activity' column and sort them in descending order
sorted_columns = df_pivot.sum(axis=0).sort_values(ascending=False).index

# Reorder the pivot table columns based on the sorted values
df_pivot = df_pivot[sorted_columns]
print(df_pivot)
# Plot the stacked bar plot with the sorted columns
df_pivot.plot(kind='bar', stacked=True, figsize=(12, 12))

# Apply the formatter for y-axis to display in millions
plt.gca().yaxis.set_major_formatter(FuncFormatter(millions))

# Set the title and labels
plt.title('Stacked Bar Plot of Annual Electricity Consumption according to Boiler and Building Activity')
plt.xlabel('Boiler')
plt.ylabel('Electricity Consumption (in Millions kW)')

# Annotate each segment with its value
for i in range(len(df_pivot)):
    cumulative_height = 0  
    for j in range(len(df_pivot.columns)):
        height = df_pivot.iloc[i, j]
        if height > 0: 
            # Calculate the position for the label
            plt.text(i, cumulative_height + height / 2, 
                     f'{height / 1_000_000:.1f}', 
                     ha='center', va='center', color='black')
            cumulative_height += height 

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Building Activity')

plt.tight_layout() 

# Show the plot
plt.show()


# %%
#Doing a Linear model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 2. Encode categorical variables if necessary (e.g., using one-hot encoding)
df_cleaned = pd.get_dummies(df_cleaned, drop_first=True)
#%%
# 3. Feature scaling (optional, depending on your data)
scaler = StandardScaler()
numerical_columns = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
df_cleaned[numerical_columns] = scaler.fit_transform(df_cleaned[numerical_columns])

print(df_cleaned)
# %%
# Define the features (independent variables) and target (dependent variable)
X = df_cleaned.drop('Electricity_Consumption', axis=1)  # Replace 'target_column' with the name of your target variable
y = df_cleaned['Electricity_Consumption']

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#%%
# Initialize the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

intercept = model.intercept_
coefficients = model.coef_
feature_names = X.columns

#Getting equation
equation = f"y = {intercept:.2f}"
for feature, coef in zip(feature_names, coefficients):
    equation += f" + ({coef:.2f} * {feature})"

print(equation)
#%%
# Predict the target values for the test set
y_pred = model.predict(X_test)
# Calculate R2 and Mean Squared Error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² Score: {r2}")
print(f"Mean Squared Error: {mse}")
# %%
print(df_cleaned)

# %%
#Test the model
input_data = {
    'SQFT': 28000, 
    'Number_Workers': 12,
    'Building_Activity': 2,
    'Work_hours':45,
    'Cooling_Days': 189,
    'Number_desktops':20,
    'Wall_Construction_Material':1,
    'Boiler':2,
    'Own_Type':2,
    'Climate_Zone':4,
    }
#%%
# Convert the input data into the same format used for training the model
# For simplicity, let's assume that you trained the model using a DataFrame.
input_df = pd.DataFrame([input_data])

# Predict using the trained model
predicted_value = model.predict(input_df)

print(predicted_value)
# %%
