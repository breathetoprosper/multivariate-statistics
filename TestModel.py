# We use this file to classify new data based on the previously saved model
# trained in StatsTests.py
# The file was saved as random_forest_model2.pkl.
# This file is in the same directory
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from StatsTests import COLUMNS_OF_INTEREST, df_val, df

# Load the scaler, PCA, and trained model
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
model = joblib.load('random_forest_model.pkl')

# Function to prepare new data and apply the model
def predict_new_data(new_data):
    # Ensure new data has the same columns
    # columns_of_interest = ['Length', 'Left', 'Right', 'Bottom', 'Top', 'Diagonal']
    new_data = new_data[COLUMNS_OF_INTEREST]

    # Scale and transform new data using the saved scaler and PCA
    X_scaled = scaler.transform(new_data)
    X_pca = pca.transform(X_scaled)

    # Make predictions
    predictions = model.predict(X_pca)
    
    return predictions

# predict with dv_tt

# predict with new data df_val
predictions = predict_new_data(df_val)
print(predictions)

# print(len(df))
# print(len(df_val))
