import numpy as np

# Step 1: Define the data
X = np.array([
    [1, 4],
    [2, 7],
    [3, 9],
    [4, 5],
    [5, 17]
])
y = np.array([4, 7, 4, 5, 6])

# Step 2: Add a column of ones for the intercept term
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add x0 = 1 to each instance

# Step 3: Calculate the coefficients using the Normal Equation
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# Print the coefficients
print(f"y = {theta_best[0]} + ({theta_best[1]}) * x1 + ({theta_best[2]}) * x2")