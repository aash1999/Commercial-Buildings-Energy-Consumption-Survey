#%%
# Q2 How do building characteristics, such as square footage, number of rooms, operating hours, and lit hours,
#  influence annual energy consumption across electricity in 2018?

import pandas as pd

df=pd.read_csv('Electricity_consumption.csv')

df.head()

#%%
#1. Eliminate all the variables not relevant for our analysis.

df = df.loc[:, ~df.columns.str.startswith('Z')]
df = df.loc[:, ~df.columns.str.startswith('FINAL')]
df = df.loc[:, ~df.columns.str.contains('OTH') ]
df = df.loc[:, ~df.columns.str.contains('FK') ]
df = df.loc[:, ~df.columns.str.startswith('AIRHAND') ]
df = df.loc[:, ~(df.columns.str.endswith('PC') & (df.columns != 'LAPTPC'))]
df = df.loc[:, ~df.columns.str.startswith('BLR')]
df = df.loc[:, ~df.columns.str.startswith('DH')]
df = df.loc[:, ~(df.columns.str.startswith('REN')& (df.columns != 'RENELC'))]
df = df.loc[:, ~df.columns.str.startswith('HW')]
df = df.loc[:, ~df.columns.str.startswith('EM')]
df = df.loc[:, ~(df.columns.str.startswith('CH') & (df.columns != 'CHILLR_EL') & (df.columns != 'CHILP_EL') & (df.columns != 'CHILP_EL'))]
df = df.loc[:, ~df.columns.str.startswith('RFC')]
df = df.loc[:, ~(df.columns.str.endswith('CO') & (df.columns != 'CO2'))]
df = df.loc[:, ~df.columns.str.endswith('SEAT')]
df = df.loc[:, ~df.columns.str.endswith('BED')]
df = df.loc[:, ~df.columns.str.endswith('HT')]
df = df.loc[:, ~df.columns.str.endswith('NG')]
df = df.loc[:, ~df.columns.str.endswith('FK')]
df = df.loc[:, ~df.columns.str.endswith('PR')]
df = df.loc[:, ~df.columns.str.endswith('WO')]
df = df.loc[:, ~df.columns.str.endswith('OT')]
df = df.loc[:, ~(df.columns.str.endswith('USED')& (df.columns != 'ELUSED'))]
df = df.loc[:, ~df.columns.str.startswith('W')]
df = df.loc[:, ~df.columns.str.startswith('X')]
df = df.loc[:, ~df.columns.str.startswith('Y')]
df = df.loc[:, ~df.columns.str.startswith('ST')]
df = df.loc[:, ~(df.columns.str.contains('FA')& (df.columns != 'FACACT'))]
df = df.loc[:, ~df.columns.str.startswith('HP')]
df = df.loc[:, ~df.columns.str.endswith('BTU')]
df = df.loc[:, ~df.columns.str.contains('WTR')]
df = df.loc[:, ~df.columns.str.startswith('RFG')]
df = df.loc[:, ~df.columns.str.startswith('PK')]
df = df.loc[:, ~df.columns.str.endswith('1')]
df = df.loc[:, ~df.columns.str.endswith('2')]
df = df.loc[:, ~df.columns.str.contains('NG')]
df = df.loc[:, ~df.columns.str.contains('HW')]
df = df.loc[:, ~((df.columns.str.contains('AC', case=False, na=False)) & (df.columns != 'VACANT')) & ~df.columns.str.endswith('EL')]
df = df.loc[:, ~((df.columns.str.contains('TR', case=False, na=False)) & (df.columns != 'ELEVTR')&(df.columns != 'NELEVTR')&(df.columns != 'ESCLTR') &(df.columns != 'NESCLTR'))]
df = df.loc[:, ~((df.columns.str.contains('COOL', case=False, na=False)) & (df.columns != 'ELCOOL'))]
df = df.loc[:, ~df.columns.str.startswith('CW')]
df = df.loc[:, ~df.columns.str.startswith('M')]
df = df.loc[:, ~df.columns.str.startswith('LT')]
print(df)
#df.to_csv('cleaned_data4.csv', index=False)

# %%
#Check the missing values.
missing_values = df.isnull().sum()

print(missing_values)

# Filter out columns with zero missing values
missing_values_non_zero = missing_values[missing_values > 0]

print(missing_values_non_zero)

# %%
# Get percentage of missing values
missing_percentage = (missing_values_non_zero / len(df)) * 100

# Combine both the counts and percentages for better insight
missing_info_non_zero = pd.DataFrame({
    'Missing Values': missing_values_non_zero,
    'Percentage': missing_percentage
})
print(missing_info_non_zero)
# %%
#2. Remove columns that have more than 20% missing values

missing_percentage = (missing_values / len(df)) * 100

# Identify columns with more than 20% missing values
columns_to_drop = missing_percentage[missing_percentage > 20].index

# Drop those columns from the DataFrame
df_cleaned = df.drop(columns=columns_to_drop)

print(f"Columns dropped: {columns_to_drop}")
print("Cleaned DataFrame:")
print(df_cleaned)
#df_cleaned.to_csv('cleaned_data1.csv', index=False)

#%%

df=df_cleaned
df = df.loc[:, ~(df.columns.str.startswith('C')& (df.columns != 'CENDIV'))]
df = df.loc[:, ~df.columns.str.startswith('D')]
df = df.loc[:, ~(df.columns.str.startswith('H')& (df.columns != 'HALO'))]
df = df.loc[:, ~df.columns.str.startswith('PR')]
df = df.loc[:, ~df.columns.str.contains('HEATP')]
df = df.loc[:, ~df.columns.str.startswith('AWN')]
df = df.loc[:, ~df.columns.str.startswith('LIT')]
df = df.loc[:, ~df.columns.str.startswith('OT')]
df = df.loc[:, ~df.columns.str.startswith('REFL')]
df = df.loc[:, ~(df.columns.str.startswith('T')&(df.columns != 'TVVIDEO'))]
df = df.loc[:, ~df.columns.str.startswith('V')]
print(df)

# %%
#Check the missing values
missing_values = df.isnull().sum()
print(missing_values)
missing_values_non_zero = missing_values[missing_values > 0]
print(missing_values_non_zero)

# Get percentage of missing values for non-zero missing columns
missing_percentage_non_zero = (missing_values_non_zero / len(df)) * 100

# Combine both the counts and percentages for better insight
missing_info_non_zero = pd.DataFrame({
    'Missing Values': missing_values_non_zero,
    'Percentage': missing_percentage_non_zero
})
print(missing_info_non_zero)

#%%
#Fill NAs with mode or mean depending on the variables:

df['OWNOCC'] = pd.to_numeric(df['OWNOCC'].fillna(df['OWNOCC'].mode().iloc[0]))
df['SLFCON'] = pd.to_numeric(df['SLFCON'].fillna(df['SLFCON'].mode().iloc[0]))
df['BOILER'] = pd.to_numeric(df['BOILER'].fillna(df['BOILER'].mode().iloc[0]))
df['REHEAT'] = pd.to_numeric(df['REHEAT'].fillna(df['REHEAT'].mode().iloc[0]))
df['FIREPLC'] = pd.to_numeric(df['FIREPLC'].fillna(df['FIREPLC'].mode().iloc[0]))
df['EVAPCL'] = pd.to_numeric(df['EVAPCL'].fillna(df['EVAPCL'].mode().iloc[0]))
df['ECN'] = pd.to_numeric(df['ECN'].fillna(df['ECN'].mode().iloc[0]))
df['ENRGYPLN'] = pd.to_numeric(df['ENRGYPLN'].fillna(df['ENRGYPLN'].mode().iloc[0]))
df['PCTERM'] = pd.to_numeric(df['PCTERM'].fillna(df['PCTERM'].mode().iloc[0]))
df['PCTERMN'] = pd.to_numeric(df['PCTERMN'].fillna(df['PCTERMN'].mean()))
df['LAPTOP'] = pd.to_numeric(df['LAPTOP'].fillna(df['LAPTOP'].mode().iloc[0]))
df['SERVER'] = pd.to_numeric(df['SERVER'].fillna(df['SERVER'].mode().iloc[0]))
df['SRVRCLST'] = pd.to_numeric(df['SRVRCLST'].fillna(df['SRVRCLST'].mode().iloc[0]))
df['LGOFFDEV'] = pd.to_numeric(df['LGOFFDEV'].fillna(df['LGOFFDEV'].mode().iloc[0]))
df['SMOFFDEV'] = pd.to_numeric(df['SMOFFDEV'].fillna(df['SMOFFDEV'].mode().iloc[0]))
df['TVVIDEO'] = pd.to_numeric(df['TVVIDEO'].fillna(df['TVVIDEO'].mode().iloc[0]))
df['FLUOR'] = pd.to_numeric(df['FLUOR'].fillna(df['FLUOR'].mode().iloc[0]))
df['BULB'] = pd.to_numeric(df['BULB'].fillna(df['BULB'].mode().iloc[0]))
df['HALO'] = pd.to_numeric(df['HALO'].fillna(df['HALO'].mode().iloc[0]))
df['LED'] = pd.to_numeric(df['LED'].fillna(df['LED'].mode().iloc[0]))
df['SCHED'] = pd.to_numeric(df['SCHED'].fillna(df['SCHED'].mode().iloc[0]))
df['OCSN'] = pd.to_numeric(df['OCSN'].fillna(df['OCSN'].mode().iloc[0]))
df['SKYLT'] = pd.to_numeric(df['SKYLT'].fillna(df['SKYLT'].mode().iloc[0]))
df['ELCNS'] = pd.to_numeric(df['ELCNS'].fillna(df['ELCNS'].mean()))
df['ELEXP'] = pd.to_numeric(df['ELEXP'].fillna(df['ELEXP'].mean()))
print(df)

#Reorder considering ELEXP and ELCNS at the beginning of DF.
new_order = ['ELEXP', 'ELCNS'] + [col for col in df.columns if col not in ['ELEXP', 'ELCNS']]

# Reorder the DataFrame
df = df[new_order]

# Print and save cleaned df to CSV
print(df)
df.to_csv('Energy_consumption_cleaned_data.csv', index=False)


# %%
