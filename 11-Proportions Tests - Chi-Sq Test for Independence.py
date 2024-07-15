import numpy as np
import scipy.stats as stats

# Hardcoded contingency table (e.g., preferences for two groups)
data = np.array([[20, 30], 
                 [30, 40]])  # Example: Group A and Group B for two categories

# Perform Chi-squared test
chi2_stat, p_value, _, _ = stats.chi2_contingency(data)

# Output the results
alpha = 0.05
print(f"Chi-squared Statistic: {chi2_stat}")
print(f"P-value: {p_value}")

# Conclusion
if p_value < alpha:
    print("Reject the null hypothesis: there is a significant association between the variables.")
else:
    print("Fail to reject the null hypothesis: there is no significant association between the variables.")
