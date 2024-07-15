'''
Analyzing OLS Results for Heteroscedasticity

To determine if your model is now homoscedastic (constant variance) instead of heteroscedastic, compare the following key metrics before and after applying transformations or corrections:
1. R-squared and Adjusted R-squared

    Before: OLS Results
        R-squared: 0.576
        Adjusted R-squared: 0.572

    After (Log Transformation):
        R-squared: 0.646
        Adjusted R-squared: 0.643

Interpretation: An increase in R-squared suggests that the model explains more variance in YY, which can indicate better fitting and potential improvement in handling heteroscedasticity.
2. Standard Errors

    Before: OLS Results
        Standard errors are larger, indicating less precision.

    After (WLS and Log):
        Check if standard errors decrease in magnitude for coefficients.

Interpretation: Smaller standard errors in WLS and log transformation results indicate more stable estimates, suggesting reduced heteroscedasticity.
3. Durbin-Watson Statistic

    Before: OLS Results: 2.312
    After: WLS Results: 2.278

Interpretation: Values close to 2 suggest no autocorrelation in residuals. Slight changes can occur, but values around 2 are preferable.
4. Omnibus and Jarque-Bera Tests

    Before: Omnibus: 3.626, JB: 3.088
    After (WLS): Omnibus: 1.176, JB: 1.191

Interpretation: Lower values after corrections suggest a more normal distribution of residuals, which is consistent with homoscedasticity.
5. Residuals Patterns

While not in the summary tables, plot residuals against fitted values or XX:

    Before: Look for patterns (e.g., fan shape).
    After: A random scatter suggests improved homoscedasticity.
    
AIC, BIC - we use these to compare tests. lower is better.
'''
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt
from scipy import stats

# Set a random seed for reproducibility
np.random.seed(42)

# Step 1: Create a heteroscedastic dataset
X = np.linspace(1, 10, 100)
y = 2 * X + np.random.normal(0, X, 100)  # Increasing variance with X

# Create a DataFrame
data = pd.DataFrame({'X': X, 'y': y})

# Plotting the heteroscedasticity
plt.scatter(data['X'], data['y'])
plt.title('Heteroscedasticity: Original Data')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# Step 2: Fit a linear regression model
model = sm.OLS(data['y'], sm.add_constant(data['X'])).fit()

# Perform Breusch-Pagan test
bp_test = het_breuschpagan(model.resid, model.model.exog)
bp_results = dict(zip(['LM Statistic', 'LM p-value', 'F Statistic', 'F p-value'], bp_test))
print("Breusch-Pagan test results before transformation:")
print(bp_results)

print()
# Hypothesis test before transformation
alpha = 0.05
if bp_results['LM p-value'] < alpha:
    print("Reject the null hypothesis: Heteroscedasticity is present.")
else:
    print("Fail to reject the null hypothesis: No evidence of heteroscedasticity.")

print()
# Step 3: Check minimum value of y and transform the data safely
#print("Minimum value of y:", data['y'].min())

# Ensure all values are suitable for log transformation
if data['y'].min() <= 0:
    #print("Adjusting values to ensure all are positive.")
    data['y'] = data['y'] - data['y'].min() + 1  # Shift values to make them positive

# Log Transformation
data['y_transformed'] = np.log(data['y'])
model_transformed = sm.OLS(data['y_transformed'], sm.add_constant(data['X'])).fit()

# Breusch-Pagan test after log transformation
bp_test_transformed = het_breuschpagan(model_transformed.resid, model_transformed.model.exog)
bp_results_transformed = dict(zip(['LM Statistic', 'LM p-value', 'F Statistic', 'F p-value'], bp_test_transformed))
print("Breusch-Pagan test results after log transformation:")
print(bp_results_transformed)

# Hypothesis test for log transformation
if bp_results_transformed['LM p-value'] < alpha:
    print("\nReject the null hypothesis: Heteroscedasticity is present.")
else:
    print("\nFail to reject the null hypothesis: No evidence of heteroscedasticity.")

# Residual plot for transformed model
plt.scatter(data['X'], model_transformed.resid)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals of Log Transformed Model')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.show()

'''
# Square Root Transformation
data['y_sqrt'] = np.sqrt(data['y'])
model_sqrt = sm.OLS(data['y_sqrt'], sm.add_constant(data['X'])).fit()

# Breusch-Pagan test for square root transformation
bp_test_sqrt = het_breuschpagan(model_sqrt.resid, model_sqrt.model.exog)
bp_results_sqrt = dict(zip(['LM Statistic', 'LM p-value', 'F Statistic', 'F p-value'], bp_test_sqrt))
print("Breusch-Pagan test results after square root transformation:")
print(bp_results_sqrt)

# Hypothesis test for square root transformation
if bp_results_sqrt['LM p-value'] < alpha:
    print("Reject the null hypothesis: Heteroscedasticity is present.")
else:
    print("Fail to reject the null hypothesis: No evidence of heteroscedasticity.")

print()
# Box-Cox Transformation
data['y_boxcox'], lambda_param = stats.boxcox(data['y'] - data['y'].min() + 1)
model_boxcox = sm.OLS(data['y_boxcox'], sm.add_constant(data['X'])).fit()

# Breusch-Pagan test after Box-Cox transformation
bp_test_boxcox = het_breuschpagan(model_boxcox.resid, model_boxcox.model.exog)
bp_results_boxcox = dict(zip(['LM Statistic', 'LM p-value', 'F Statistic', 'F p-value'], bp_test_boxcox))
print("Breusch-Pagan test results after Box-Cox transformation:")
print(bp_results_boxcox)

# Hypothesis test for Box-Cox transformation
if bp_results_boxcox['LM p-value'] < alpha:
    print("Reject the null hypothesis: Heteroscedasticity is present.")
else:
    print("Fail to reject the null hypothesis: No evidence of heteroscedasticity.")

print()
# Weighted Least Squares (WLS)
weights = 1 / (data['X']**2)  # Example of using X to define weights
model_wls = sm.WLS(data['y'], sm.add_constant(data['X']), weights=weights).fit()
bp_test_wls = het_breuschpagan(model_wls.resid, model_wls.model.exog)
bp_results_wls = dict(zip(['LM Statistic', 'LM p-value', 'F Statistic', 'F p-value'], bp_test_wls))
print("Breusch-Pagan test results after WLS:")
print(bp_results_wls)

# Hypothesis test for WLS
if bp_results_wls['LM p-value'] < alpha:
    print("Reject the null hypothesis: Heteroscedasticity is present.")
else:
    print("Fail to reject the null hypothesis: No evidence of heteroscedasticity.")

print()
# Polynomial Regression
data['X_squared'] = data['X']**2
model_poly = sm.OLS(data['y'], sm.add_constant(data[['X', 'X_squared']])).fit()
bp_test_poly = het_breuschpagan(model_poly.resid, model_poly.model.exog)
bp_results_poly = dict(zip(['LM Statistic', 'LM p-value', 'F Statistic', 'F p-value'], bp_test_poly))
print("Breusch-Pagan test results after polynomial regression:")
print(bp_results_poly)

# Hypothesis test for polynomial regression
if bp_results_poly['LM p-value'] < alpha:
    print("Reject the null hypothesis: Heteroscedasticity is present.")
else:
    print("Fail to reject the null hypothesis: No evidence of heteroscedasticity.")


'''
