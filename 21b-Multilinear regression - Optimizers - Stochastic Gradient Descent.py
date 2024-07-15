from sklearn.linear_model import SGDRegressor
import numpy as np

# Define data
X = np.array([[1, 4], [2, 7], [3, 9], [4, 5], [5, 17]])
y = np.array([4, 7, 4, 5, 6])

# Define and fit the model
model = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01)
model.fit(X, y)

print(f"SGD coefficients: {model.intercept_}, {model.coef_}")
