from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np

# Define data
X = np.array([[1, 4], [2, 7], [3, 9], [4, 5], [5, 17]])
y = np.array([4, 7, 4, 5, 6])

# Define PCA and linear regression
pca = PCA(n_components=2)
lr = LinearRegression()
model = make_pipeline(pca, lr)

# Fit the model
model.fit(X, y)

print(f"PCR coefficients: {model.named_steps['linearregression'].intercept_}, {model.named_steps['linearregression'].coef_}")
