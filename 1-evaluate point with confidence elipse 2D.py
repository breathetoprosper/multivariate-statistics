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
alpha = 0.05  # Significance level

# Point to check
mu_1 = np.array([7, 11])

# Dynamically determine n and df
n = X.shape[0]  # Number of observations
p = X.shape[1]  # Number of dimensions
df1 = p
df2 = n - p

# Calculate the scaling factor
scaling_factor = ((n - 1) * p / (n - p)) * f.ppf(1 - alpha, df1, df2)

# Calculate eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Sort eigenvalues and eigenvectors
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Calculate the angle of rotation
angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

# Calculate the axes lengths
axes_lengths = np.sqrt(scaling_factor * eigenvalues)

# Generate points for the ellipse
theta = np.linspace(0, 2 * np.pi, 100)
ellipse_x = axes_lengths[0] * np.cos(theta)
ellipse_y = axes_lengths[1] * np.sin(theta)

# Rotate the ellipse
R = np.array([[np.cos(angle), -np.sin(angle)],
              [np.sin(angle), np.cos(angle)]])
ellipse_points = np.dot(R, np.array([ellipse_x, ellipse_y]))

# Translate the ellipse to the center
ellipse_points[0, :] += mu[0]
ellipse_points[1, :] += mu[1]

# Create the plot
plt.figure(figsize=(10, 8))
plt.plot(ellipse_points[0, :], ellipse_points[1, :], 'b-')
plt.plot(mu[0], mu[1], 'ko', markersize=10)
plt.annotate(f'μ ({mu[0]:.2f}, {mu[1]:.2f})', (mu[0], mu[1]), xytext=(5, 5),
             textcoords='offset points', fontweight='bold')
plt.plot(mu_1[0], mu_1[1], 'go', markersize=10)  # Plot mu_1 in green
plt.annotate(f'μ₁ ({mu_1[0]}, {mu_1[1]})', (mu_1[0], mu_1[1]), xytext=(5, 5),
             textcoords='offset points', color='green', fontweight='bold')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'{(1-alpha)*100}% Confidence Ellipse')
plt.axis('equal')
plt.grid(True)
plt.show()

# Print eigenvalues and eigenvectors
print("Eigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)

# Calculate and print the F-value
F_value = f.ppf(1 - alpha, df1, df2)
print(f"\nF-value (F_{df1},{df2},{alpha}) = {F_value:.4f}")

# Do the hypothesis test:


# Check if mu_1 belongs to the confidence region
diff = mu_1 - mu
maha_dist = diff @ np.linalg.inv(covariance_matrix) @ diff.T
critical_value = f.ppf(1 - alpha, df1, df2) * (p * (n - 1) / (n - p))

print(f"Mahalanobis distance of μ₁: {maha_dist:.4f}")
print(f"Critical value: {critical_value:.4f}")

# Decision based on the hypothesis test
if maha_dist <= critical_value:
    conclusion = f"We do not reject the H₀ at {(1-alpha)*100}% significance level. μ₁ ({mu_1[0]}, {mu_1[1]}) belongs to the confidence region."
else:
    conclusion = f"We reject the H₀ at {(1-alpha)*100}% significance level. μ₁ ({mu_1[0]}, {mu_1[1]}) does not belong to the confidence region."

print(f"\nConclusion: {conclusion}")