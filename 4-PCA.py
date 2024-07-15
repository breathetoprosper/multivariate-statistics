import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Define the matrix X
X = np.array([[1, 5, 3, 1],
              [4, 2, 6, 3],
              [1, 4, 3, 2],
              [4, 4, 1, 1],
             [5, 5, 2, 3]])


# Create a PCA object
pca = PCA()
X_pca = pca.fit_transform(X)

# Calculate cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Print cumulative explained variance ratio
print("Cumulative explained variance ratio:")
for i, ratio in enumerate(cumulative_variance_ratio, 1):
    print(f"Component {i}: {ratio:.4f}")

# Determine the number of components that explain at least 80% of variance
defined_variance = 0.80
n_components = np.argmax(cumulative_variance_ratio >= defined_variance) + 1

# Plot explained variance ratio
plt.figure(figsize=(8, 6))
plt.plot(cumulative_variance_ratio, marker='o', linestyle='--')
plt.axhline(y=defined_variance, color='r', linestyle='-', label=f'{defined_variance * 100:.1f}% Variance Explained')
plt.axvline(x=n_components, color='g', linestyle='-', label='{} Components'.format(n_components))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance Ratio vs. Number of Components')
plt.legend()
plt.grid()
plt.show()

# Print the chosen number of components
print("Number of components to retain {} variance: {}".format(defined_variance * 100, n_components))

# Perform PCA with the chosen number of components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Print the explained variance ratio for these components
print("Explained variance ratio for {} components:".format(n_components), pca.explained_variance_ratio_)

# Print the reduced transformed data
print("Reduced transformed data with {} components:".format(n_components))
print(X_pca)
