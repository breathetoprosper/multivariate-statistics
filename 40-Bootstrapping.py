import numpy as np
import matplotlib.pyplot as plt

# Sample data
data = np.array([2.3, 2.9, 3.1, 3.6, 4.0, 4.2, 4.5, 5.1, 5.6, 6.0])

# Bootstrapping function
def bootstrap_mean_confidence_interval(data, num_resamples=1000, confidence=0.95):
    means = []
    n = len(data)
    
    # Resampling
    for _ in range(num_resamples):
        resample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(resample))
    
    # Compute the lower and upper percentiles
    lower_bound = np.percentile(means, (1-confidence)/2 * 100)
    upper_bound = np.percentile(means, (1+confidence)/2 * 100)
    
    return means, lower_bound, upper_bound

# Compute the 95% confidence interval for the mean
bootstrapped_means, lower, upper = bootstrap_mean_confidence_interval(data)

# Plot the initial data
plt.figure(figsize=(12, 6))

# Initial plot: Histogram of the original data
plt.subplot(1, 2, 1)
plt.hist(data, bins=5, edgecolor='black', alpha=0.7)
plt.title('Histogram of Original Data')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Bootstrapped plot: Histogram of bootstrapped means
plt.subplot(1, 2, 2)
plt.hist(bootstrapped_means, bins=30, edgecolor='black', alpha=0.7)

# Explicitly cast to float
original_mean = float(np.mean(data))
lower_bound = float(lower)
upper_bound = float(upper)

plt.axvline(x=original_mean, color='r', linestyle='--', label='Original Mean')
plt.axvline(x=lower_bound, color='g', linestyle='--', label='95% CI Lower Bound')
plt.axvline(x=upper_bound, color='b', linestyle='--', label='95% CI Upper Bound')

plt.title('Histogram of Bootstrapped Means')
plt.xlabel('Mean Value')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()
