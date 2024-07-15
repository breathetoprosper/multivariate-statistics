from sklearn.linear_model import Ridge
import numpy as np

# Define data
X = np.array([[1, 4], [2, 7], [3, 9], [4, 5], [5, 17]])
y = np.array([4, 7, 4, 5, 6])

# Define and fit the model
model = Ridge(alpha=1.0)  # alpha is the regularization strength
model.fit(X, y)

print(f"Ridge coefficients: {model.intercept_}, {model.coef_}")


