import numpy as np

# Define data
X = np.array([[1, 4], [2, 7], [3, 9], [4, 5], [5, 17]])
y = np.array([4, 7, 4, 5, 6])

# Add intercept term
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Gradient Descent Parameters
learning_rate = 0.01
iterations = 1000
m = len(y)

# Initialize theta
theta = np.random.randn(X_b.shape[1], 1)

# Gradient Descent
for _ in range(iterations):
    gradients = 2/m * X_b.T @ (X_b @ theta - y.reshape(-1, 1))
    theta -= learning_rate * gradients

print(f"Gradient Descent coefficients: {theta.flatten()}")
