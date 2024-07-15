import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chi2, f


import sys
sys.stdout.reconfigure(encoding='utf-8')

# Given data
X_bar = np.array([38.541, 25.854, 122.358])  # Mean vector
S = np.array([[13.8195, 15.8284, 24.7250],
              [15.8284, 34.8769, 63.0215],
              [24.7250, 63.0215, 540.1553]])  # Covariance matrix
mu_1 = np.array([16.89, 8.76, 109.23])  # Point to check

alpha = 0.05  # Significance level
n = 15  # Number of observations for each variable
p = 3  # Number of variables (a, b, c)

# Degrees of freedom
df1 = p
df2 = n - 1  # Using n-1 as we're dealing with sample statistics

# Calculate the scaling factor
scaling_factor = ((n - 1) * p / (n - p)) * f.ppf(1 - alpha, df1, df2)

# Calculate eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(S)

# Sort eigenvalues and eigenvectors
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Calculate the axes lengths
axes_lengths = np.sqrt(scaling_factor * eigenvalues)

# Generate points for the ellipsoid
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = axes_lengths[0] * np.outer(np.cos(u), np.sin(v))
y = axes_lengths[1] * np.outer(np.sin(u), np.sin(v))
z = axes_lengths[2] * np.outer(np.ones_like(u), np.cos(v))

# Rotate the ellipsoid
for i in range(len(x)):
    for j in range(len(x)):
        [x[i,j], y[i,j], z[i,j]] = np.dot(eigenvectors, [x[i,j], y[i,j], z[i,j]])

# Translate the ellipsoid to the center
x += X_bar[0]
y += X_bar[1]
z += X_bar[2]

# Create the 3D plot
plt.ioff()  # Turn on interactive mode
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b', alpha=0.2)
ax.scatter(X_bar[0], X_bar[1], X_bar[2], color='black', s=100, label='μ')
ax.scatter(mu_1[0], mu_1[1], mu_1[2], color='green', s=100, label='μ₁')

# Annotate X_bar
ax.text(X_bar[0], X_bar[1], X_bar[2],
        f'μ ({X_bar[0]:.2f}, {X_bar[1]:.2f}, {X_bar[2]:.2f})',
        color='black')

# Annotate μ₁
ax.text(mu_1[0], mu_1[1], mu_1[2],
        f'μ₁ ({mu_1[0]:.2f}, {mu_1[1]:.2f}, {mu_1[2]:.2f})',
        color='green')

ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('c')
ax.legend()
plt.title(f'{(1-alpha)*100}% Confidence Ellipsoid')

# Check if mu_1 belongs to the confidence region
diff = mu_1 - X_bar
maha_dist = diff @ np.linalg.inv(S) @ diff.T
critical_value = scaling_factor

print(f"Mahalanobis distance of μ1: {maha_dist:.4f}")
print(f"Critical value: {critical_value:.4f}")

if maha_dist <= critical_value:
    conclusion = f"We do not reject the H0 at {(1-alpha)*100}% significance level. μ1 ({mu_1[0]}, {mu_1[1]}, {mu_1[2]}) belongs to the confidence region."
else:
    conclusion = f"We reject the H0 at {(1-alpha)*100}% significance level. μ1 ({mu_1[0]}, {mu_1[1]}, {mu_1[2]}) does not belong to the confidence region."

print(f"\nConclusion: {conclusion}")

plt.show(block=True)
