import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Generate sample data. # You can change the parameters of make_moons to vary the dataset
X, _ = make_moons(n_samples=300, noise=0.1, random_state=40)  # Change `n_samples` and `noise`

# Plot the data
plt.figure() # create new figure
plt.scatter(X[:, 0], X[:, 1])
plt.title('Sample Data')
plt.show(block=False)


# You can vary the following parameters:
# - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
# - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
# Apply DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=300)
labels = dbscan.fit_predict(X)

# Plot the clustered data
plt.figure() # Create new figure
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma')
plt.title('DBSCAN Clustering')
plt.show()
