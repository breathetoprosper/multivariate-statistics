import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from StatsTests import COLUMNS_OF_INTEREST, TARGET, df
import statsmodels.api as sm
import statsmodels.formula.api as smf


# Prepare data
X = df[COLUMNS_OF_INTEREST]
y = df[TARGET]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'LDA': LDA(),
    'QDA': QDA(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': xgb.XGBClassifier(),
    'Neural Networks': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42,
                                     learning_rate_init=0.001, early_stopping=True, validation_fraction=0.2,
                                     n_iter_no_change=10),
    'Random Forests': RandomForestClassifier()
}

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {model.__class__.__name__}")
    print(classification_report(y_test, y_pred))
    return cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

print("\nModel Evaluation and Cross-Validation Results:\n")
for name, model in models.items():
    print(f"Evaluating {name}...")
    cv_scores = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f} +/- {cv_scores.std():.2f}\n")

