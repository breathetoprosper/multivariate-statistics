from sklearn.linear_model import Lasso
import numpy as np

# Define data
X = np.array([[1, 4], [2, 7], [3, 9], [4, 5], [5, 17]])
y = np.array([4, 7, 4, 5, 6])

# Define and fit the model
model = Lasso(alpha=1.0)  # alpha is the regularization strength
model.fit(X, y)

print(f"Lasso coefficients: {model.intercept_}, {model.coef_}")
