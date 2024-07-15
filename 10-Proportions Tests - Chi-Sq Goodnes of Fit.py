import numpy as np
import scipy.stats as stats

# Hardcoded observed frequencies
observed = np.array([50, 30, 20])  # Sum = 100

# Expected frequencies must sum to the same total as observed
expected = np.array([40, 40, 20])  # Sum = 100

# Perform Chi-squared test
chi2_stat, p_value = stats.chisquare(observed, expected)

# Output the results
alpha = 0.05
print(f"Chi-squared Statistic: {chi2_stat}")
print(f"P-value: {p_value}")

# Conclusion
if p_value < alpha:
    print("Reject the null hypothesis: There is a difference between observed and expected frequencies.")
else:
    print("Fail to reject the null hypothesis: There is no difference between observed and expected frequencies.")
