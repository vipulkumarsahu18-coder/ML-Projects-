import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=200, centers=3, n_features=2, random_state=42)

df = pd.DataFrame(X, columns=["feature1", "feature2"])

model = KMeans(n_clusters=3, random_state=42)
df["cluster"] = model.fit_predict(df[["feature1", "feature2"]])

print(df.head())

centroids = model.cluster_centers_
print("Centroids:\n", centroids)

plt.scatter(df["feature1"], df["feature2"], c=df["cluster"])
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c="red", marker="X")
plt.title("K-Means Clustering")
plt.show()
