'''
Explanation and Hypotheses:

    Wald Test:
    The Wald test is a statistical test used to determine whether the parameters of a model are significantly different from a specified value, typically zero. It's commonly used in the context of regression models.

    Example:
        Suppose you have a simple linear regression model that predicts a person's weight based on their height:
        weight=β0+β1⋅height

        Null Hypothesis (H0): β1=0 (height has no effect on weight).
        Alternative Hypothesis (H1): β1≠0 (height has an effect on weight).

        After fitting the model to data, the Wald test evaluates the estimated coefficient β^1 If the test's p-value is less than the significance level (e.g., 0.05), you reject the null hypothesis, concluding that height significantly affects weight. If the p-value is greater than the significance level, you fail to reject the null hypothesis, concluding there is no significant evidence that height affects weight.
    
        So the interpretation is:
        Null Hypothesis (H0): The second coefficient is zero.
        Alternative Hypothesis (H1): The second coefficient is not zero.
        Conclusion: Based on the p-value, we decide whether to reject H0.

    Likelihood Ratio Test:
    The Likelihood Ratio Test (LRT) is used to compare the goodness of fit between two nested models: one is a special case of the other (the simpler model is nested within the more complex model).

        Example:

        Suppose you have two models for predicting a student's exam score:

            Simpler Model (Restricted Model):
            score=β0+β1⋅study_hours

            More Complex Model (Unrestricted Model):
            score=β0+β1⋅study_hours+β2⋅attendance

            Null Hypothesis (H0): The simpler model is sufficient; the additional variable (attendance) in the complex model does not significantly improve the model.
            Alternative Hypothesis (H1): The more complex model provides a significantly better fit.

        The LRT compares the log-likelihoods of the two models:

        LR Statistic= -2(log(L_restricted) - log(L_unrestricted))

        Where log(L_restricted) is the log-likelihood of the simpler model and log(L_unrestricted)) is the log-likelihood of the more complex model.

        If the LR statistic is large enough (based on a chi-squared distribution), you reject the null hypothesis, concluding that the more complex model provides a significantly better fit. 
        If the LR statistic is not large, you fail to reject the null hypothesis, concluding that the simpler model is sufficient.
    
        So the interpretation is:
        Null Hypothesis (H0): The restricted model is sufficient (no significant difference between the restricted and unrestricted models).
        Alternative Hypothesis (H1): The unrestricted model provides a better fit.
        Conclusion: Based on the p-value, we decide whether to reject H0.

    Lagrange Multiplier Test:
        Null Hypothesis (H0): There is no heteroskedasticity (the variance of the errors is constant).
        Alternative Hypothesis (H1): There is heteroskedasticity (the variance of the errors is not constant).
        Conclusion: Based on the p-value, we decide whether to reject H0.
'''


import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.stats.diagnostic import het_breuschpagan

# Generate some example data
np.random.seed(0)
n = 100
X = np.random.rand(n, 2)
beta = np.array([1, 2])
y = X @ beta + np.random.normal(size=n)

# Fit the model
X = sm.add_constant(X)  # Adds a constant term to the predictor
model = sm.OLS(y, X).fit()

# Wald test
R = np.array([[0, 1, 0]])  # Hypothesis matrix
wald_test = model.wald_test(R, scalar=True)
wald_statistic = wald_test.statistic
wald_pvalue = wald_test.pvalue

# Fit the restricted model
X_restricted = X[:, [0, 1]]  # Removing the second predictor
model_restricted = sm.OLS(y, X_restricted).fit()

# Likelihood Ratio Test
lr_test_stat = -2 * (model_restricted.llf - model.llf)
lr_test_pvalue = stats.chi2.sf(lr_test_stat, df=1)

# Lagrange Multiplier Test (Breusch-Pagan test for heteroskedasticity)
lm_test = het_breuschpagan(model.resid, model.model.exog)
lm_test_stat = lm_test[0]
lm_test_pvalue = lm_test[1]

# Conclusions at 5% significance level
alpha = 0.05

# Wald Test Conclusion
if wald_pvalue < alpha:
    wald_conclusion = "Reject the null hypothesis: The second coefficient is significantly different from zero."
else:
    wald_conclusion = "Fail to reject the null hypothesis: There is no evidence that the second coefficient is different from zero."

# Likelihood Ratio Test Conclusion
if lr_test_pvalue < alpha:
    lr_conclusion = "Reject the null hypothesis: The unrestricted model provides a significantly better fit than the restricted model."
else:
    lr_conclusion = "Fail to reject the null hypothesis: There is no significant difference in fit between the restricted and unrestricted models."

# Lagrange Multiplier Test Conclusion
if lm_test_pvalue < alpha:
    lm_conclusion = "Reject the null hypothesis: There is evidence of heteroskedasticity in the model."
else:
    lm_conclusion = "Fail to reject the null hypothesis: There is no evidence of heteroskedasticity in the model."

# Print Results
print("Wald Test")
print(f"Test Statistic: {wald_statistic}")
print(f"p-value: {wald_pvalue}")
print(wald_conclusion)

print("\nLikelihood Ratio Test")
print(f"Likelihood Ratio Test Statistic: {lr_test_stat}")
print(f"p-value: {lr_test_pvalue}")
print(lr_conclusion)

print("\nLagrange Multiplier Test")
print(f"Lagrange Multiplier Test Statistic: {lm_test_stat}")
print(f"p-value: {lm_test_pvalue}")
print(lm_conclusion)
