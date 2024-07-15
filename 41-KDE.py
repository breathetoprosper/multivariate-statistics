import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm

# Define the original density function
def original_density(x):
    # Mixture of two normal distributions
    return 0.5 * norm.pdf(x, loc=-2, scale=1) + 0.5 * norm.pdf(x, loc=2, scale=1)

# Generate sample data from the original density function
np.random.seed(0)
data = np.concatenate([np.random.normal(loc=-2, scale=1, size=500),
                        np.random.normal(loc=2, scale=1, size=500)])

# Define the range of x values for plotting
x = np.linspace(min(data) - 1, max(data) + 1, 1000)

# Calculate KDE using Scott's and Silverman's methods
scott_bandwidth = np.std(data) * (4 / (3 * len(data))) ** (1 / 5)
silverman_bandwidth = np.std(data) * (4 / (3 * len(data))) ** (1 / 5) * (1 / (2.0 ** 0.2))

kde_scott = gaussian_kde(data, bw_method=scott_bandwidth)
kde_silverman = gaussian_kde(data, bw_method=silverman_bandwidth)

kde_scott_values = kde_scott.evaluate(x)
kde_silverman_values = kde_silverman.evaluate(x)
original_values = original_density(x)

# Plot the results
plt.figure(figsize=(14, 6))

# Plot original density function
plt.subplot(1, 2, 1)
plt.plot(x, original_values, label="Original Density Function", color='black')
plt.fill_between(x, original_values, alpha=0.2, color='black')
plt.title("Original Density Function")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()

# Plot KDE using Scott's and Silverman's Bandwidths
plt.subplot(1, 2, 2)
plt.plot(x, kde_scott_values, label=f'Scott\'s Bandwidth ({scott_bandwidth:.2f})', color='blue')
plt.plot(x, kde_silverman_values, label=f'Silverman\'s Bandwidth ({silverman_bandwidth:.2f})', linestyle='--', color='red')
plt.fill_between(x, kde_scott_values, alpha=0.3, color='blue')
plt.fill_between(x, kde_silverman_values, alpha=0.3, color='red')
plt.title("KDE with Scott's and Silverman's Bandwidths")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()
plt.show()
