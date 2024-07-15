import numpy as np
import pandas as pd
from scipy.stats import chi2

def box_m_test(data):
    # Data should be a dictionary of DataFrames for different groups
    n = sum([df.shape[0] for df in data.values()])  # Total sample size
    k = len(data)  # Number of groups
    pooled_cov = sum([(df.shape[0] - 1) * df.cov() for df in data.values()]) / (n - k)
    
    # Calculate Box's M statistic
    m_stat = (n - k) * np.log(np.linalg.det(pooled_cov))
    
    # Calculate degrees of freedom
    df1 = k * (k - 1) / 2
    df2 = n - k
    
    # Calculate p-value
    p_value = 1 - chi2.cdf(m_stat, df1)
    
    return m_stat, p_value

# Example data
data = {
    'Group1': pd.DataFrame(np.random.rand(10, 3)),
    'Group2': pd.DataFrame(np.random.rand(10, 3)),
    'Group3': pd.DataFrame(np.random.rand(10, 3)),
}

m_stat, p_value = box_m_test(data)
print(f"Box's M Statistic: {m_stat}, p-value: {p_value}")

# Hypothesis test
alpha = 0.05
if p_value < 0.05:
    print(f"p-value: {p_value} < Box's M Statistic: {m_stat}. Reject the null hypothesis. There are significant differences between covariance matrices.")
else:
    print(f"p-value: {p_value} > Box's M Statistic: {m_stat}. Fail to reject the null hypothesis. There are no significant differences between covariance matrices.")


