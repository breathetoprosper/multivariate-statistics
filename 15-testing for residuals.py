'''
Residual = Observed - Predicted
After conducting a chi-q test for independence:
You can visualize residuals to check these characteristics:
1. Independence: Expect a scatterplot of residuals centered around zero without any discernible patterns. 
    Good fit; no systematic patterns
2. Dependence: Look for patterns or clusters in the residual plot, indicating specific relationships or associations.
    Evidence of association; model may need refinement


'''
# This code is just to see the pictures
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Set up the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('Comparison of Homoscedastic vs Heteroscedastic Residual Plots', fontsize=16)

# Generate homoscedastic data
np.random.seed(42)
x_homo = np.linspace(0, 10, 100)
y_homo = 2 * x_homo + np.random.normal(0, 1, 100)  # Constant variance

# Fit homoscedastic model
model_homo = LinearRegression()
model_homo.fit(x_homo.reshape(-1, 1), y_homo)

# Calculate homoscedastic residuals and fitted values
fitted_values_homo = model_homo.predict(x_homo.reshape(-1, 1))
residuals_homo = y_homo - fitted_values_homo

# Plot homoscedastic residuals
ax1.scatter(fitted_values_homo, residuals_homo)
ax1.set_xlabel('Fitted values')
ax1.set_ylabel('Residuals')
ax1.set_title('Homoscedastic Residual Plot')
ax1.axhline(y=0, color='r', linestyle='--')

# Generate heteroscedastic data
x_hetero = np.linspace(0, 10, 100)
y_hetero = 2 * x_hetero + np.random.normal(0, 0.5 * x_hetero)  # Increasing variance

# Fit heteroscedastic model
model_hetero = LinearRegression()
model_hetero.fit(x_hetero.reshape(-1, 1), y_hetero)

# Calculate heteroscedastic residuals and fitted values
fitted_values_hetero = model_hetero.predict(x_hetero.reshape(-1, 1))
residuals_hetero = y_hetero - fitted_values_hetero

# Plot heteroscedastic residuals
ax2.scatter(fitted_values_hetero, residuals_hetero)
ax2.set_xlabel('Fitted values')
ax2.set_ylabel('Residuals')
ax2.set_title('Heteroscedastic Residual Plot')
ax2.axhline(y=0, color='r', linestyle='--')

# Adjust layout and display
plt.tight_layout()
plt.show()

# ------------------------
# This code is an example for residuals check tests
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

# Sample data
x = np.array([1, 2, 3, 4, 5])
# y = np.array([2, 2.5, 3.5, 4.5, 5])  # Homoscedastic example
y = np.array([2, 2.5, 3.5, 4.5, 10])  # Heteroscedastic example

# Fit a linear model
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()

# Get residuals
residuals = model.resid

# Breusch-Pagan test
bp_test = het_breuschpagan(residuals, model.model.exog)
bp_p_value = bp_test[1]

# Shapiro-Wilk test
shapiro_test = stats.shapiro(residuals)
shapiro_p_value = shapiro_test.pvalue

# Conclusions
alpha = 0.05

# Breusch-Pagan Test Conclusion
if bp_p_value < alpha:
    bp_conclusion = "Reject the null hypothesis: evidence of heteroscedasticity."
else:
    bp_conclusion = "Fail to reject the null hypothesis: no evidence of heteroscedasticity."

# Shapiro-Wilk Test Conclusion
if shapiro_p_value < alpha:
    sw_conclusion = "Reject the null hypothesis: residuals are not normally distributed."
else:
    sw_conclusion = "Fail to reject the null hypothesis: residuals are normally distributed."

# Output results
print("### Example ###")
print(f"Breusch-Pagan p-value: {bp_p_value}")
print(bp_conclusion)
print(f"Shapiro-Wilk p-value: {shapiro_p_value}")
print(sw_conclusion)

