import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)

df = pd.DataFrame(X, columns=["feature1", "feature2"])

kmeans = KMeans(n_clusters=4, random_state=42)
df["kmeans_cluster"] = kmeans.fit_predict(df)

agg = AgglomerativeClustering(n_clusters=4)
df["agg_cluster"] = agg.fit_predict(df)

dbscan = DBSCAN(eps=0.6, min_samples=5)
df["dbscan_cluster"] = dbscan.fit_predict(df)

centroids = kmeans.cluster_centers_

plt.scatter(df["feature1"], df["feature2"], c=df["kmeans_cluster"])
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c="red", marker="X")
plt.title("K-Means Clustering")
plt.show()

plt.scatter(df["feature1"], df["feature2"], c=df["agg_cluster"])
plt.title("Agglomerative Clustering")
plt.show()

plt.scatter(df["feature1"], df["feature2"], c=df["dbscan_cluster"])
plt.title("DBSCAN Clustering")
plt.show()

print(df.head())
