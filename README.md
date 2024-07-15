This is a full Classification model project done in Python/Xampp

Files: 

#Access.py
1. It imports an SQL dataset from PHPMyAdmin and gets it into a DataFrame
2. Does preliminary EDA
   
#StatsTests.py
1. It imports the DataFrame from Access.py
2. Does EDA
3. Does Multicollinearity Test
4. Does PCA
5. Does Classification Model
6. Prints Accuracy and Classification Report
7. saves the PCA, Scaler, and Model

#TestModel.py
1. loads the Model
2. Makes predictions on new data
