# We use this file to classify new data based on the previously saved model
# trained in StatsTests.py
# The file was saved as random_forest_model.pkl.
# This file is in the same directory
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the scaler, PCA, and trained model
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
model = joblib.load('random_forest_model.pkl')

# Function to prepare new data and apply the model
def predict_new_data(new_data):
    # Ensure new data has the same columns
    columns_of_interest = ['Length', 'Left', 'Right', 'Bottom', 'Top', 'Diagonal']
    new_data = new_data[columns_of_interest]

    # Scale and transform new data using the saved scaler and PCA
    X_scaled = scaler.transform(new_data)
    X_pca = pca.transform(X_scaled)

    # Make predictions
    predictions = model.predict(X_pca)
    
    return predictions

# Example usage with new data
new_data = pd.DataFrame({
    'Length': [10.2, 15.4, 13.3],
    'Left': [5.6, 7.1, 6.2],
    'Right': [5.5, 7.0, 6.1],
    'Bottom': [3.2, 4.0, 3.8],
    'Top': [4.5, 5.3, 4.9],
    'Diagonal': [11.0, 16.2, 14.5]
})

predictions = predict_new_data(new_data)
print(predictions)
