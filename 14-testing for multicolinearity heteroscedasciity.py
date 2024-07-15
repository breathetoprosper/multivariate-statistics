'''
a) Gauss-Markov Tests:

Residual plots: This visualizes the relationship between fitted values and residuals. It helps detect non-linearity, heteroscedasticity, and outliers.
VIF (Variance Inflation Factor): Measures multicollinearity. VIF > 5-10 indicates problematic multicollinearity.
Durbin-Watson: Tests for autocorrelation in residuals. Values around 2 indicate no autocorrelation.

b) Multicollinearity Tests:

Correlation Matrix: Shows correlations between predictors. High correlations (>0.8) suggest multicollinearity.
VIF: Already explained above.

c) Heteroscedasticity Tests:

Residual Plots: Already explained above.
Breusch-Pagan test: Tests for heteroscedasticity. Low p-values (<0.05) indicate heteroscedasticity.
White's test: Another test for heteroscedasticity. Interpretation is similar to Breusch-Pagan.

d) Normality Tests:

Lilliefors test: Tests if data comes from a normally distributed population.
Shapiro-Wilk test: Tests the null hypothesis that a sample came from a normally distributed population.
Kolmogorov-Smirnov test: Compares the sample distribution to a normal distribution.
Anderson-Darling test: Tests if a sample comes from a specific distribution (in this case, normal).
Jarque-Bera test: Tests the null hypothesis that sample data have the skewness and kurtosis matching a normal distribution.
Q-Q plot: Graphical method to compare two probability distributions.

For all these normality tests, p-values < 0.05 generally indicate non-normality.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Generate dummy data
np.random.seed(42)
n = 100
X1 = np.random.normal(0, 1, n)
X2 = np.random.normal(0, 1, n)
X3 = 0.5 * X1 + 0.5 * X2 + np.random.normal(0, 0.5, n)
Y = 2 * X1 + 3 * X2 + 4 * X3 + np.random.normal(0, 1, n)

# Create a DataFrame
df = pd.DataFrame({'Y': Y, 'X1': X1, 'X2': X2, 'X3': X3})

# Fit OLS model
model = ols('Y ~ X1 + X2 + X3', data=df).fit()

# Create a figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(20, 20))
fig.suptitle('Statistical Test Plots', fontsize=16)

# 1. Gauss-Markov Tests
print("Gauss-Markov Tests:")

# Residual plots
residuals = model.resid
fitted_values = model.fittedvalues

axs[0, 0].scatter(fitted_values, residuals)
axs[0, 0].set_xlabel('Fitted values')
axs[0, 0].set_ylabel('Residuals')
axs[0, 0].set_title('Residual Plot')

# VIF
X = sm.add_constant(df[['X1', 'X2', 'X3']])
vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("Variance Inflation Factors:")
print(vif)
print("\nConclusion: VIF values less than 5 indicate no problematic multicollinearity.")

# Durbin-Watson
dw_statistic = durbin_watson(model.resid)
print(f"\nDurbin-Watson statistic: {dw_statistic}")
print("Conclusion: A value close to 2 indicates no significant autocorrelation in the residuals.")

# 2. Multicollinearity Tests
print("\nMulticollinearity Tests:")

# Correlation Matrix
correlation_matrix = df[['X1', 'X2', 'X3']].corr()
print("Correlation Matrix:")
print(correlation_matrix)
print("\nConclusion: Correlation coefficients close to 1 or -1 indicate high correlation between variables.")

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axs[0, 1])
axs[0, 1].set_title('Correlation Matrix Heatmap')

# 3. Heteroscedasticity Tests
print("\nHeteroscedasticity Tests:")

# Breusch-Pagan test
bp_test = het_breuschpagan(model.resid, model.model.exog)
print(f"Breusch-Pagan test p-value: {bp_test[1]}")
print(f"Conclusion: {'Reject' if bp_test[1] < 0.05 else 'Fail to reject'} the null hypothesis of homoscedasticity.")

# White's test
white_test = het_white(model.resid, model.model.exog)
print(f"\nWhite's test p-value: {white_test[1]}")
print(f"Conclusion: {'Reject' if white_test[1] < 0.05 else 'Fail to reject'} the null hypothesis of homoscedasticity.")

# 4. Normality Tests
print("\nNormality Tests:")

# Lilliefors test
_, lilliefors_pvalue = lilliefors(model.resid)
print(f"Lilliefors test p-value: {lilliefors_pvalue}")
print(f"Conclusion: {'Reject' if lilliefors_pvalue < 0.05 else 'Fail to reject'} the null hypothesis of normality.")

# Shapiro-Wilk test
_, shapiro_pvalue = stats.shapiro(model.resid)
print(f"\nShapiro-Wilk test p-value: {shapiro_pvalue}")
print(f"Conclusion: {'Reject' if shapiro_pvalue < 0.05 else 'Fail to reject'} the null hypothesis of normality.")

# Kolmogorov-Smirnov test
_, ks_pvalue = stats.kstest(model.resid, 'norm')
print(f"\nKolmogorov-Smirnov test p-value: {ks_pvalue}")
print(f"Conclusion: {'Reject' if ks_pvalue < 0.05 else 'Fail to reject'} the null hypothesis of normality.")

# Anderson-Darling test
result = stats.anderson(model.resid, dist='norm')
print(f"\nAnderson-Darling test statistic: {result.statistic}")
print("Critical values:")
for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    print(f"    {sl}%: {cv}")
print(f"Conclusion: {'Reject' if result.statistic > result.critical_values[2] else 'Fail to reject'} the null hypothesis of normality at the 5% significance level.")

# Jarque-Bera test
jb_statistic, jb_pvalue = stats.jarque_bera(model.resid)
print(f"\nJarque-Bera test statistic: {jb_statistic}")
print(f"Jarque-Bera test p-value: {jb_pvalue}")
print(f"Conclusion: {'Reject' if jb_pvalue < 0.05 else 'Fail to reject'} the null hypothesis of normality.")

# Q-Q plot
sm.qqplot(model.resid, ax=axs[1, 0], line='45')
axs[1, 0].set_title('Q-Q Plot')
axs[1, 0].text(0.05, 0.95, "If points follow the line closely,\nit suggests normality.", transform=axs[1, 0].transAxes, verticalalignment='top')

# Histogram of residuals
axs[1, 1].hist(model.resid, bins=20, edgecolor='black')
axs[1, 1].set_title('Histogram of Residuals')
axs[1, 1].set_xlabel('Residuals')
axs[1, 1].set_ylabel('Frequency')
axs[1, 1].text(0.05, 0.95, "A bell-shaped histogram\nsuggests normality.", transform=axs[1, 1].transAxes, verticalalignment='top')

plt.tight_layout()
plt.show()

print("\nOverall Conclusion:")
print("1. Gauss-Markov Assumptions: Check residual plot for patterns. VIF and Durbin-Watson suggest no major issues.")
print("2. Multicollinearity: Examine correlation matrix for high correlations between predictors.")
print("3. Heteroscedasticity: Breusch-Pagan and White's tests results indicate whether there's evidence of heteroscedasticity.")
print("4. Normality: Multiple tests provide evidence about the normality of residuals. Consider results collectively.")
print("\nRemember, these tests should be interpreted together with domain knowledge and the specific context of your analysis.")