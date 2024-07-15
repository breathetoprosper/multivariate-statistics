'''
The Gauss-Markov Theorem states that, in a linear regression model, the ordinary least squares (OLS) estimator is the best linear unbiased estimator (BLUE), provided certain conditions are met. These conditions are:

    Linearity: The model is linear in the parameters.
    Exogeneity: The errors have a mean of zero conditional on the independent variables.
    Homoscedasticity: The errors have constant variance.
    No autocorrelation: The errors are uncorrelated with each other.
    No perfect multicollinearity: The independent variables are not perfectly correlated.
    //Multicollinearity occurs when two or more independent variables in a regression model are highly correlated, 
    leading to unreliable coefficient estimates and increased standard errors. 
    This can make it difficult to determine the individual effect of each variable.

If these conditions are satisfied, the OLS estimator has the following statistical properties in finite samples:

    Unbiased: The mean of the OLS estimators equals the true value of the parameters.
    Best: The OLS has the smallest variance among all linear unbiased estimators.

These properties ensure that the OLS estimator is efficient and reliable under the specified conditions.

In this specific dataset(don't change it so yo uca see this conclusion):
Thank you for providing the image and the conclusions. Let's analyze what we're seeing and interpret the results.

1. The image:
This image shows three scatter plots, each representing the relationship between one of the independent variables (X1, X2, X3) and the dependent variable (y).

- X1 vs y: There appears to be a positive linear relationship. As X1 increases, y tends to increase as well. This suggests that X1 might be a good predictor for y.

- X2 vs y: There's no clear pattern visible. The points seem to be scattered randomly, which might indicate that X2 has little to no linear relationship with y.

- X3 vs y: Similar to X2, there's no clear pattern. This suggests that X3 might also have little to no linear relationship with y.

These plots help us assess the linearity assumption of the Gauss-Markov theorem. While X1 shows a linear relationship, X2 and X3 don't appear to have strong linear relationships with y.

2. Interpreting the conclusions:

a) Durbin-Watson statistic: 2.3827
This is close to 2, which is good. It suggests that there's no significant autocorrelation in the residuals. The independence assumption is likely met.

b) Breusch-Pagan test p-value: 0.7570
This p-value is well above 0.05, indicating that we fail to reject the null hypothesis of homoscedasticity. The homoscedasticity assumption is likely met.

c) Shapiro-Wilk test p-value: 0.5021
This p-value is above 0.05, suggesting that we fail to reject the null hypothesis that the residuals are normally distributed. The normality assumption is likely met.

d) Variance Inflation Factors (VIF):
- X1, X2, and X3 have VIF values close to 1, which is excellent. This indicates no multicollinearity among these predictors.
- However, the Intercept has a VIF of 9.910762, which is above 5. This is unusual and might be causing the multicollinearity warning.

The conclusion states that multicollinearity may be present due to a VIF > 5 for one or more variables. This is likely referring to the Intercept's VIF. However, in practice, we usually don't consider the Intercept when checking for multicollinearity. The predictor variables themselves don't show signs of multicollinearity.

Overall, most Gauss-Markov assumptions seem to be met (independence, homoscedasticity, normality of residuals). The linearity assumption might be questionable for X2 and X3 based on the scatter plots. The multicollinearity warning is likely a false positive due to the Intercept's VIF.

Given these results, the model might be appropriate for use, but with some caveats:
1. The relationships of X2 and X3 with y should be further investigated, possibly considering non-linear relationships.
2. The high VIF for the Intercept should be examined, although it may not necessarily indicate a problem with the model's predictors.

It would be advisable to reassess the model, possibly considering transformations of variables or removal of non-significant predictors, to improve its overall fit and adherence to assumptions.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_breuschpagan

# Generate sample data (replace this with your actual data)
np.random.seed(42)
n = 100
X = np.random.rand(n, 3)
y = 2 + 3*X[:, 0] + 1.5*X[:, 1] + 0.5*X[:, 2] + np.random.normal(0, 1, n)

# Create a DataFrame
df = pd.DataFrame(X, columns=['X1', 'X2', 'X3'])
df['y'] = y

# Fit the OLS model
model = ols('y ~ X1 + X2 + X3', data=df).fit()

# 1. Linearity
plt.figure(figsize=(12, 4))
for i, col in enumerate(['X1', 'X2', 'X3']):
    plt.subplot(1, 3, i+1)
    plt.scatter(df[col], df['y'])
    plt.xlabel(col)
    plt.ylabel('y')
plt.tight_layout()
plt.show()

# 2. Independence (Durbin-Watson test)
dw_statistic = durbin_watson(model.resid)
print(f"Durbin-Watson statistic: {dw_statistic:.4f}")

# 3. Homoscedasticity (Breusch-Pagan test)
_, p_value, _, _ = het_breuschpagan(model.resid, model.model.exog)
print(f"Breusch-Pagan test p-value: {p_value:.4f}")

# 4. Normality of residuals (Shapiro-Wilk test)
_, normality_p_value = stats.shapiro(model.resid)
print(f"Shapiro-Wilk test p-value: {normality_p_value:.4f}")

# 5. No multicollinearity (VIF)
vif_data = pd.DataFrame()
vif_data["Variable"] = model.model.exog_names
vif_data["VIF"] = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
print("Variance Inflation Factors:")
print(vif_data)

# Conclusion
print("\nConclusion:")
assumptions_met = True

if dw_statistic < 1.5 or dw_statistic > 2.5:
    print("- Independence assumption may be violated (Durbin-Watson statistic not close to 2)")
    assumptions_met = False

if p_value < 0.05:
    print("- Homoscedasticity assumption may be violated (Breusch-Pagan test p-value < 0.05)")
    assumptions_met = False

if normality_p_value < 0.05:
    print("- Normality of residuals assumption may be violated (Shapiro-Wilk test p-value < 0.05)")
    assumptions_met = False

if vif_data["VIF"].max() > 5:
    print("- Multicollinearity may be present (VIF > 5 for one or more variables)")
    assumptions_met = False

if assumptions_met:
    print("All Gauss-Markov assumptions appear to be met at the 5% significance level.")
    print("The model is appropriate for use.")
else:
    print("One or more Gauss-Markov assumptions may be violated at the 5% significance level.")
    print("The model may not be appropriate for use without further investigation or adjustments.")