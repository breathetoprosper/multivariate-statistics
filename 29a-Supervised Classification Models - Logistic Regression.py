import numpy as np
import pandas as pd
import statsmodels.api as sm

# Sample data
np.random.seed(0)
n = 100
X = np.random.rand(n, 2)  # Independent variables
X = sm.add_constant(X)  # Add constant term (intercept)
y = (X[:, 1] + X[:, 2] > 1).astype(int)  # Dependent variable

# Create a DataFrame for convenience
data = pd.DataFrame(X, columns=['const', 'X1', 'X2'])
data['y'] = y

# Fit logistic regression model
logit_model = sm.Logit(data['y'], data[['const', 'X1', 'X2']])
logit_result = logit_model.fit()

# Print the summary of the logistic regression model
print("Logistic Regression Model Summary:")
print(logit_result.summary())
