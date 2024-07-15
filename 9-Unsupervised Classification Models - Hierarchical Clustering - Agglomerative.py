import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate random data
np.random.seed(0)
X, _ = make_blobs(n_samples=100, centers=3, random_state=0, cluster_std=0.60)

# Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
labels = agg_clustering.fit_predict(X)

# Calculate centroids
centroids = np.array([X[labels == i].mean(axis=0) for i in range(3)])

# Annotate centroid coordinates
for i, centroid in enumerate(centroids):
    plt.text(centroid[0], centroid[1], f'({centroid[0]:.2f}, {centroid[1]:.2f})', 
                fontsize=12, ha='right', color='red', fontweight="bold")

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

plt.title("Agglomerative Hierarchical Clustering")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show(block=False)

# Dendrogram
Z = linkage(X, 'ward')
plt.figure()
dendrogram(Z)
plt.title("Dendrogram for Agglomerative Clustering")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.show()
