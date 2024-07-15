import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

# Step 1: Generate synthetic data with three clusters
np.random.seed(0)

# Cluster 1
mean1 = [0, 0]
cov1 = [[1, 0.3], [0.3, 1]]
data1 = np.random.multivariate_normal(mean1, cov1, 100)

# Cluster 2
mean2 = [5, 5]
cov2 = [[1, -0.6], [-0.6, 1]]
data2 = np.random.multivariate_normal(mean2, cov2, 100)

# Cluster 3
mean3 = [9, 1]
cov3 = [[1, 0.5], [0.5, 1]]
data3 = np.random.multivariate_normal(mean3, cov3, 100)

# Combine the data
data = np.vstack((data1, data2, data3))

# Step 2: Fit a Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(data)

# Predict the cluster for each data point
labels = gmm.predict(data)

# Step 3: Plot the data and the resulting GMM clusters
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=30)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Gaussian Mixture Model Clusters')

# Define a function to plot the GMM components as ellipses
def plot_gmm_ellipse(gmm, ax, n_std=2.0):
    """
    Plots the Gaussian components of a GMM as ellipses.

    Parameters:
        gmm: The fitted GaussianMixture model.
        ax: The matplotlib axis to plot on.
        n_std: The number of standard deviations for the ellipse radii (default is 2.0 for ~95% confidence).
    """
    for mean, cov in zip(gmm.means_, gmm.covariances_):
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle = np.arctan2(*eigenvectors[:, 0][::-1])
        angle = np.degrees(angle)
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='k', facecolor='none')

        ax.add_patch(ell)

    # Plot the means of the clusters
    ax.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', s=100, marker='X', label='Cluster Means')
    
    
    # Annotate the coordinates of the means
    for i, mean in enumerate(gmm.means_):
        ax.annotate(f'({mean[0]:.2f}, {mean[1]:.2f})', 
                     xy=mean, 
                     textcoords='offset points', 
                     xytext=(5, 5), 
                     ha='center', 
                     fontsize=14, 
                     color='black',
                     fontweight = 'bold')

# Plot the GMM ellipses with the desired confidence level
fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=30)
plot_gmm_ellipse(gmm, ax, n_std=2.0)  # 2 standard deviations for ~95% confidence interval
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_title('Gaussian Mixture Model Clusters')

plt.show()
