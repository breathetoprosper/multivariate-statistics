# 1 Scikit Way

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Create/Load Dataset
# you have a CSV file:
# data = pd.read_csv('your_data.csv')

# For this example, let's create some sample data
np.random.seed(0)
X = np.random.rand(100, 3)  # 100 samples, 3 features
y = 2 + 3*X[:, 0] + 1.5*X[:, 1] + 0.5*X[:, 2] + np.random.randn(100) * 0.1

# 2. Split the data intro training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred = model.predict(X_test)

# 5. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean squared error: {mse:.4f}")
print(f"R-squared score: {r2:.4f}")


# 6. Print the coeffs and intercept
print("Coefficients:")
for i, coef in enumerate(model.coef_):
    print(f"X{i+1}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")