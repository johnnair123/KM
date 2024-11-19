import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics importsilhouette_score
from sklearn.cluster import KMeans

df=pd.read_csv('https://raw.githubusercontent.com/safal/DS-
ML/refs/heads/main/car.csv')

print(df.head())
print(df.shape)
new_df=df[['Volume','Weight','CO2']]
print(new_df.head())
sil_scores = []
clusters = range(2, 11)
for i in clusters:
kmeans = KMeans(n_clusters=i, random_state=1)
labels = kmeans.fit_predict(new_df)
sil_scores.append(silhouette_score(new_df, labels))
print(sil_scores)
max_sil_score = max(sil_scores)
best_cluster = clusters[sil_scores.index(max_sil_score)]
print(f"Maximum Silhouette Score: {max_sil_score}")
print(f"Optimal Number of Clusters: {best_cluster}")
kmeans = KMeans(n_clusters=3, random_state=1)
kmeans.fit(new_df)
plt.plot(clusters,sil_scores, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.show()
cluster_df=pd.DataFrame({'Clusters':labels})
dataframe=pd.concat([new_df,cluster_df],axis=1)
print(dataframe.head())
