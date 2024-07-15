from sklearn.linear_model import BayesianRidge
import numpy as np

# Define data
X = np.array([[1, 4], [2, 7], [3, 9], [4, 5], [5, 17]])
y = np.array([4, 7, 4, 5, 6])

# Define and fit the model
model = BayesianRidge()
model.fit(X, y)

print(f"Bayesian Ridge coefficients: {model.intercept_}, {model.coef_}")
