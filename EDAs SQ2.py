#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df_cleaned=pd.read_csv('cleaned_data3.csv')

print(df_cleaned.head())
#%%
df_continuous_var = df_cleaned.drop(['Building_Activity', 'Main_Heat_Equip', 'Main_Cool_Equp'], axis=1)

corr_matrix = df_continuous_var.corr()

# Create the heatmap
plt.figure(figsize=(10, 8))  # Set the figure size
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Add title and labels
plt.title('Correlation Matrix')

# Show the plot
plt.show()


#%%
#VIF Values
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["Feature"] = df_continuous_var.columns
vif_data["VIF"] = [variance_inflation_factor(df_continuous_var.values, i) for i in range(len(df_continuous_var.columns))]

print(vif_data.round(2))

##########
#%%
#Smart Q2
#How do occupancy metrics (Number of Workers, % Light off and Working hours) affect EUI?

# Exploring some EDAs
# Creating a bubble chart Electricity consumption, SQFT, Number_workers and boiler
# Plot the bubble chart


df_EDA2 = df_cleaned[df_cleaned['Floor_Ceiling_Height'] != 995]
df_EDA2 = df_cleaned[df_cleaned['Main_Cool_Equp'].isin([1, 2, 3, 4, 5, 6])]
df_EDA2['Main_Cool_Equp'] = df_EDA2['Main_Cool_Equp'].replace({1: 'Split-System', 2: 'Pack-Unit', 3: 'Elec-chiller', 4:'Heat-pump', 5: 'Individual-A/C', 6:'Swamp-Cooler'})

sns.scatterplot(
    data=df_EDA2, 
    x='Percent_Cooled', 
    y='Energy_Use_Intensity', 
    #size= ('Floor_Ceiling_Height'),
    sizes=(50, 1000), 
    palette='Pastel2', 
    hue='Main_Cool_Equp', 
    legend=True
)
#plt.ylim(0,600)
plt.xlabel('Percent of Cooled')
plt.title('Percent of Cooled vs EUI by Main Cooling System')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# Show the plot
plt.show()
#%%
df_EDA2 = df_cleaned[df_cleaned['Main_Cool_Equp'].isin([1, 2, 3, 4, 5, 6])]
df_EDA2['Main_Cool_Equp'] = df_EDA2['Main_Cool_Equp'].replace({1: 'Split-System', 2: 'Pack-Unit', 3: 'Elec-chiller', 4:'Heat-pump', 5: 'Individual-A/C', 6:'Swamp-Cooler'})

sns.scatterplot(
    data=df_EDA2, 
    x='Percent_Cooled', 
    y='Energy_Use_Intensity', 
    #size= ('Floor_Ceiling_Height'),
    sizes=(50, 1000), 
    palette='Pastel2', 
    hue='Main_Cool_Equp', 
    legend=True
)
#plt.ylim(0,400)
plt.xlabel('Percent of Cooled')
plt.title('Percent of Cooled vs EUI by Main Cooling System')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# Show the plot
plt.show()

#%%

#Violin plot 
# Create the violin plot
x_order = ['Elec-chiller','Pack-Unit','Split-System','Individual-A/C','Heat-pump', 'Swamp-Cooler']
plt.figure(figsize=(8, 6))
sns.violinplot(x='Main_Cool_Equp', y='Energy_Use_Intensity', data=df_EDA2,palette='Pastel2',order=x_order)

# Customize plot labels and title|
plt.title('EUI vs Cooling System')
plt.xlabel('Main Cooling System')
plt.ylabel('Energy Use Intensity')
plt.tight_layout()
# Show plot
plt.show()

#%%
#Anova test
import scipy.stats as stats

# Example: 'dependent_var' is the variable you're testing, 'group_var' is the group/categorical variable
groups = [df_EDA2[df_EDA2['Main_Cool_Equp'] == group]['Energy_Use_Intensity'] for group in df_EDA2['Main_Cool_Equp'].unique()]

# Running the ANOVA
f_statistic, p_value = stats.f_oneway(*groups)

print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")


#%%
#%%

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter



# Create the pivot table
building_activity_map = {

    2: 'Office',
    13: 'Public assembly',
    14: 'Education',
    15: 'Food service',
    16: 'Healthcare'
}


df_EDA2['Building_Activity'] = df_EDA2['Building_Activity'].replace(building_activity_map)



df_pivot = df_EDA2.pivot_table(
    index='Main_Cool_Equp',
    columns='Building_Activity',
    values='Energy_Use_Intensity',
    aggfunc='sum',
    fill_value=0
)

# Calculate the sum of each 'Building_Activity' column and sort them in descending order
sorted_columns = df_pivot.sum(axis=0).sort_values(ascending=False).head(5).index  # Use .head(10) to get top 10

# Reorder the pivot table columns to keep only the top 10 building activities
df_pivot = df_pivot[sorted_columns]

# Print the updated pivot table
print(df_pivot)


# Sort the rows (Main_Cool_Equp) based on the total sum of EUI in descending order
df_pivot = df_pivot.loc[df_pivot.sum(axis=1).sort_values(ascending=False).index]

# Plot the stacked bar plot with the sorted columns
df_pivot.plot(kind='bar', stacked=True, figsize=(12, 12))

#x_order=['Elec-chiller','Heat-pump','Pack-Unit','Individual-A/C','Split-System','Swamp-Cooler']

#plt.xticks(ticks=range(len(x_order)), labels=x_order)


# Set the title and labels
plt.title('Stacked Bar Plot of Annual EUI according to Main Cooling System and Building Activity')
plt.xlabel('Main Cooling System')
plt.ylabel('Energy Use Intensity (Thousands)')

# Annotate each segment with its value
for i in range(len(df_pivot)):
    cumulative_height = 0  
    for j in range(len(df_pivot.columns)):
        height = df_pivot.iloc[i, j]
        if height > 0: 
            # Calculate the position for the label
            plt.text(i, cumulative_height + height / 2, 
                     f'{height/10000:.2f}', 
                     ha='center', va='center', color='black')
            cumulative_height += height 

# Apply the formatter to the y-axis to divide by 1000
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:.0f}'))

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Building Activity')

plt.tight_layout() 

# Show the plot
plt.show()
#%%

#Violin plot 
# Create the violin plot
#x_order = ['Elec-chiller','Pack-Unit','Split-System','Individual-A/C','Heat-pump', 'Swamp-Cooler']
plt.figure(figsize=(8, 6))
sns.violinplot(x='Main_Cool_Equp', y='Percent_Cooled', data=df_EDA2,palette='Pastel2')

# Customize plot labels and title|
plt.title('% Cooled vs Cooling System')
plt.xlabel('Main Cooling System')
plt.ylabel('Percent Cooled')
plt.tight_layout()
# Show plot
plt.show()


# %%

#%%
#Anova test
import scipy.stats as stats

# Example: 'dependent_var' is the variable you're testing, 'group_var' is the group/categorical variable
groups = [df_EDA2[df_EDA2['Main_Cool_Equp'] == group]['Percent_Cooled'] for group in df_EDA2['Main_Cool_Equp'].unique()]

# Running the ANOVA
f_statistic, p_value = stats.f_oneway(*groups)

print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

##############
#%%
#%%

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter



# Create the pivot table
building_activity_map = {

    2: 'Office',
    4: 'Government office',
    6: 'Mixed-use office',
    12: 'Convenience store',
    21: 'Religious Worship',
    32: 'Fast food',
    33: 'Restaurant',
    35: 'Hospital',
    38: 'Hotel',
    42: 'Retail',
    50: 'Mall'
}
df_EDA2['Building_Activity'] = df_EDA2['Building_Activity'].replace(building_activity_map)



df_pivot = df_EDA2.pivot_table(
    index='Main_Cool_Equp',
    columns='Building_Activity',
    values='Percent_Cooled',
    aggfunc='sum',
    fill_value=0
)

# Calculate the sum of each 'Building_Activity' column and sort them in descending order
sorted_columns = df_pivot.sum(axis=0).sort_values(ascending=False).head(5).index  # Use .head(10) to get top 10

# Reorder the pivot table columns to keep only the top 10 building activities
df_pivot = df_pivot[sorted_columns]

# Print the updated pivot table
print(df_pivot)


# Sort the rows (Main_Cool_Equp) based on the total sum of EUI in descending order
df_pivot = df_pivot.loc[df_pivot.sum(axis=1).sort_values(ascending=False).index]

# Plot the stacked bar plot with the sorted columns
df_pivot.plot(kind='bar', stacked=True, figsize=(12, 12))

#x_order=['Elec-chiller','Heat-pump','Pack-Unit','Individual-A/C','Split-System','Swamp-Cooler']

#plt.xticks(ticks=range(len(x_order)), labels=x_order)


# Set the title and labels
plt.title('Stacked Bar Plot of Percent_Cooled, Main Cooling System and Building Activity')
plt.xlabel('Percent Cooled')
plt.ylabel('Energy Use Intensity')

# Annotate each segment with its value
for i in range(len(df_pivot)):
    cumulative_height = 0  
    for j in range(len(df_pivot.columns)):
        height = df_pivot.iloc[i, j]
        if height > 0: 
            # Calculate the position for the label
            plt.text(i, cumulative_height + height / 2, 
                     f'{height/1000:.2f}', 
                     ha='center', va='center', color='black')
            cumulative_height += height 

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Building Activity')

plt.tight_layout() 

# Show the plot
plt.show()


#############