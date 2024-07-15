import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def kmeans(X, k, max_iters=100):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids

    return centroids, labels

# Example usage
if __name__ == "__main__":
    # Generate random data
    np.random.seed(0)
    X = np.random.rand(100, 2)

    # Run k-means
    k = 3
    centroids, labels = kmeans(X, k)

    # Create a DataFrame to include coordinates and their corresponding cluster
    data = pd.DataFrame(X, columns=['X1', 'X2'])
    data['Cluster'] = labels

    # Create a DataFrame for the centroids
    centroid_df = pd.DataFrame(centroids, columns=['Centroid_X1', 'Centroid_X2'])
    centroid_df['Cluster'] = range(k)

    # Plot the results
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', label='Data Points')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

    # Add centroid labels to the plot
    for i, centroid in enumerate(centroids):
        plt.text(centroid[0], centroid[1], f'({centroid[0]:.2f}, {centroid[1]:.2f})', 
                 fontsize=12, ha='right', color='red', fontweight="bold")

    plt.title("K-Means Clustering")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()
