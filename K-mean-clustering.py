#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#%%
df = pd.read_csv("Selected Data.csv")
#%%

print(df.head())

#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)  # Scale only the selected columns

# List to store the average silhouette scores for different values of k
silhouette_scores = []

# Try clustering with k values from 2 to 10
k_values = range(2, 10)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df)
    
    # Calculate the silhouette score for the current k
    score = silhouette_score(df, kmeans.labels_)
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
df['kmeans_4'] = kmeans.labels_  

print(df.head)
#%%
from sklearn.decomposition import PCA

# Perform PCA to reduce the data to 2D
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)

# Add the PCA components to the dataframe for plotting
df['pca1'] = pca_components[:, 0]
df['pca2'] = pca_components[:, 1]

# Create a scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='pca1', y='pca2', hue='kmeans_4', palette='Set1', data=df, s=100, alpha=0.7, edgecolor='k')
plt.title('KMeans Clusters Visualized with PCA')
plt.legend(title='Cluster')
plt.show()

#%%
plt.scatter(x=df['WKHRS'], y=df['EUI'], c=df['kmeans_4'])
plt.xlabel('WKHRS')
plt.ylabel('EUI')

#%%
import pandas as pd
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42)
df['kmeans_4'] = kmeans.fit_predict(df[['EUI', 'WKHRS']])

#%%
# Group by the cluster and calculate summary statistics
summary = df.groupby('kmeans_4').agg({
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
summary.to_csv ('summary1.csv', index=False)


#%%
import seaborn as sns
import matplotlib.pyplot as plt

custom_palette = {'0': 'purple', '1': 'skyblue', '2': 'lightgreen', '3': 'orange'}

plt.figure(figsize=(12, 6)) 
sns.boxplot(x='kmeans_4', y='EUI', data=df, palette=custom_palette)
plt.title('Boxplot of WKHRS by Cluster')
plt.xlabel('Cluster')
plt.ylabel('WKHRS')
plt.grid(True)
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

custom_palette = {'0': 'purple', '1': 'skyblue', '2': 'lightgreen', '3': 'orange'}

plt.figure(figsize=(12, 6)) 
sns.boxplot(x='kmeans_4', y='EUI', data=df, palette=custom_palette)
plt.title('Boxplot of EUI by Cluster')
plt.xlabel('Cluster')
plt.ylabel('EUI')
plt.grid(True)
plt.show()


# %%
import seaborn as sns
import matplotlib.pyplot as plt

custom_palette = {'0': 'purple', '1': 'skyblue', '2': 'lightgreen', '3': 'orange'}

plt.figure(figsize=(12, 6)) 
sns.boxplot(x='kmeans_4', y='WKHRS', data=df, palette=custom_palette)
plt.title('Boxplot of WKHRS by Cluster')
plt.xlabel('Cluster')
plt.ylabel('WKHRS')
plt.grid(True)
plt.show()
#%%

custom_palette = {'0': 'purple', '1': 'skyblue', '2': 'lightgreen', '3': 'orange'}
plt.figure(figsize=(12, 6))  
sns.boxplot(x='kmeans_4', y='SQFT', data=df,palette=custom_palette)
plt.title('Boxplot of SQFT by Cluster')
plt.xlabel('Cluster')
plt.ylabel('SQFT')
plt.grid(True)
plt.show()
# %%

import scipy.stats as stats
import pandas as pd
grouped_data = [df[df['kmeans_4'] == cluster]['WKHRS'] for cluster in df['kmeans_4'].unique()]
anova_result = stats.f_oneway(*grouped_data)
print("ANOVA Test:")
print(f"F-statistic: {anova_result.statistic}")
print(f"P-value: {anova_result.pvalue}")


# %%
import scipy.stats as stats
import pandas as pd
grouped_data = [df[df['kmeans_4'] == cluster]['SQFT'] for cluster in df['kmeans_4'].unique()]
anova_result = stats.f_oneway(*grouped_data)
print("ANOVA Test:")
print(f"F-statistic: {anova_result.statistic}")
print(f"P-value: {anova_result.pvalue}")


#%%

import scipy.stats as stats
import pandas as pd
grouped_data = [df[df['kmeans_4'] == cluster]['EUI'] for cluster in df['kmeans_4'].unique()]
anova_result = stats.f_oneway(*grouped_data)
print("ANOVA Test:")
print(f"F-statistic: {anova_result.statistic}")
print(f"P-value: {anova_result.pvalue}")

