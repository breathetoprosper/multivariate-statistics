from sklearn.linear_model import ElasticNet
import numpy as np

# Define data
X = np.array([[1, 4], [2, 7], [3, 9], [4, 5], [5, 17]])
y = np.array([4, 7, 4, 5, 6])

# Define and fit the model
model = ElasticNet(alpha=1.0, l1_ratio=0.5)  # alpha is the regularization strength, l1_ratio is the mix ratio
model.fit(X, y)

print(f"Elastic Net coefficients: {model.intercept_}, {model.coef_}")
