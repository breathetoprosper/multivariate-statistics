from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load data
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train model
model = SVC()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Support Vector Machine Accuracy:", accuracy_score(y_test, y_pred))
