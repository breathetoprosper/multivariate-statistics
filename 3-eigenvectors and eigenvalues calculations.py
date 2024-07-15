# Multivariate Operations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, f

# Given data
X = np.array([[2, 12],
              [8, 9],
              [6, 9],
              [8, 10]])

mu = np.mean(X, axis=0)  # Calculate the mean
covariance_matrix = np.cov(X.T)  # Calculate the covariance matrix
correlation_matrix = np.corrcoef(X.T)  # Calculate the correlation matrix


# Calculate eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# prints
print("Mean:", mu)
print("\nCovariance Matrix:\n", covariance_matrix)
print("\nCorrelation Matrix:\n", correlation_matrix)
print("\neigenvalues:\n", eigenvalues)
print("\neigenvectors:\n", eigenvectors)
