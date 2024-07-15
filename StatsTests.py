# The principal components (PCs) are the weights or coefficients.
# PC's are linear combinations of the features.
# that describe how each original feature contributes to the new principal components. 
# so after the pca you get the weights
'''
so say you have: 
Principal component loadings:
       Length      Left     Right    Bottom       Top  Diagonal
PC1  0.060736 -0.451692 -0.479899 -0.413376 -0.374314  0.500990
PC2  0.799232  0.376376  0.295015 -0.210502 -0.092474  0.282263
PC3  0.025409 -0.087226 -0.077899 -0.602411  0.786412 -0.065836
PC4  0.581893 -0.456453 -0.344579  0.463670  0.230048 -0.257692 

Let's denote the original features as:

    F1 = Length
    F2 = Left
    F3 = Right
    F4 = Bottom
    F5 = Top
    F6 = Diagonal

The equations for the principal components can be written as follows:
Principal Component 1 (PC1)
PC1=0.060736⋅F1-0.451692⋅F2-0.479899⋅F3-0.413376⋅F4-0.374314⋅F5+0.500990⋅F6

Principal Component 2 (PC2)
PC2=0.799232⋅F1+0.376376⋅F2+0.295015⋅F3-0.210502⋅F4-0.092474⋅F5+0.282263⋅F6

Principal Component 3 (PC3)
PC3=0.025409⋅F1-0.087226⋅F2-0.077899⋅F3-0.602411⋅F4+0.786412⋅F5-0.065836⋅F6

Principal Component 4 (PC4)
PC4=0.581893⋅F1-0.456453⋅F2-0.344579⋅F3+0.463670⋅F4+0.230048⋅F5-0.257692⋅F6


In these equations:

    Each term represents the contribution of an original feature Fi to a principal component.
    The coefficients (or loadings) are the weights that reflect how strongly each original feature contributes to the principal component.

When you project your data into the space of these principal components, you use these equations to compute the values of PC1, PC2, PC3, and PC4 for each data point in your original feature space.

'''
from Access import get_dataframe  # Import the function from Access.py
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bartlett
from factor_analyzer.factor_analyzer import calculate_kmo

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Retrieve the DataFrame from Access.py
df = get_dataframe()

def plot_data(df):
    if df is not None and not df.empty:
        # Now you can perform your tests or analysis on the DataFrame
        print(df.head())  # Display the first few rows of the DataFrame
        print(df.info())  # Display DataFrame info
        print(df.describe(include='all'))  # Display descriptive statistics for all columns

        # Example test: Check if any column has missing values
        missing_values = df.isnull().sum()
        print("Missing values per column:")
        print(missing_values)

        # Determine the number of columns and create a grid layout
        num_columns = len(df.columns)
        num_rows = (num_columns // 3) + (num_columns % 3 > 0)  # Number of rows needed (3 columns per row)

        # Set the size of the figure
        plt.figure(figsize=(15, num_rows * 4))  # Adjust height based on number of rows

        # Loop through each column in the DataFrame and create a subplot for each
        for i, column in enumerate(df.columns):
            plt.subplot(num_rows, 3, i + 1)  # Create a subplot in a grid (3 columns per row)
            df[column].hist(bins=30, edgecolor='black')  # Plot histogram for each column
            plt.title(column)  # Set the title of the subplot to the column name
            plt.xlabel('Value')  # Set the x-axis label
            plt.ylabel('Frequency')  # Set the y-axis label

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Show the plot
        plt.show()

    else:
        print("Failed to retrieve data or DataFrame is empty.")


def multicolinearity_tests(df):
    # Specify the columns of interest
    columns_of_interest = ['Length', 'Left', 'Right', 'Bottom', 'Top', 'Diagonal']
    
    # Select only the columns of interest
    df_selected = df[columns_of_interest]
    
    # Ensure that all selected columns are numeric
    df_selected = df_selected.apply(pd.to_numeric, errors='coerce')
    
    # Drop rows with NaN values in any of the selected columns
    df_selected = df_selected.dropna()
    
    # Check if there are enough rows remaining after dropping NaNs
    if df_selected.shape[0] < 2:
        print("Not enough data points for KMO test after dropping NaNs.")
        return

    print("\nMulticolinaerity Tests: Bartlett and KMO:\n")
    # Perform Bartlett's test
    # Bartlett's test expects the data in separate arrays; here, we'll transpose the DataFrame
    stat, p_value = bartlett(*[df_selected[col] for col in columns_of_interest])

    # Print results for Bartlett's test
    print("Bartlett's test statistic:", stat)
    print("p-value:", p_value)

    # Interpret the Bartlett's test results
    alpha = 0.05
    if p_value < alpha:
        print("Decision: Reject the null hypothesis: The variances are significantly different.\n")
    else:
        print("Decision: Fail to reject the null hypothesis: No significant difference in variances.\n")
        
    # Perform KMO Test
    kmo_all, kmo_model = calculate_kmo(df_selected)
        
    # Create a DataFrame to display KMO statistics with column names
    kmo_df = pd.DataFrame({
        'Variable': columns_of_interest,
        'KMO Statistic': kmo_all
    })
    
    #KMO decision rule:
    print("KMO decision rule:")
    print("0.8 - 1.0: Excellent")
    print("0.7 - 0.79: Good")
    print("0.6 - 0.69: Mediocre")
    print("0.5 - 0.59: Poor")
    print("Below 0.5: Unacceptable\n")
    # Print KMO statistics for each variable
    print("KMO statistic for each variable:")
    print(kmo_df)

    # Print overall KMO statistic for the model
    print("\nKMO statistic (overall):", kmo_model)

#conduct PCA test
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca_test(X_train, X_test, n_components=0.90):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # Principal components (each row represents a principal component)
    print("PCA components (loadings):")
    print(pca.components_, "\n")
    print("Explained variance ratio:")
    print(pca.explained_variance_ratio_, "\n")
    print("Cumulative explained variance ratio:")
    print(cumulative_explained_variance, "\n")

    # Identify which original features are important for each principal component
    # Higher absolute values in `pca.components_` indicate greater contribution
    components_df = pd.DataFrame(pca.components_, columns=columns_of_interest, index=[f'PC{i+1}' for i in range(pca.n_components_)])
    print("Principal component loadings:")
    print(components_df, "\n")
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, 'bo-', markersize=10)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
    plt.grid(True)
    plt.show()

    return X_train_pca, X_test_pca, pca

def train_and_evaluate_model(X_train_pca, X_test_pca, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train_pca, y_train)
    
    y_pred = model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    classification = classification_report(y_test, y_pred)
    
    return accuracy, classification

def pca_biplot(df):
    # Specify the columns of interest
    columns_of_interest = ['Length', 'Left', 'Right', 'Bottom', 'Top', 'Diagonal']
    
    # Select only the columns of interest
    X = df[columns_of_interest]
    
    # Optional: Standardize features before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA(n_components=2)  # We use 2 components for visualization
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot the scores (samples in the new PCA space)
    plt.figure(figsize=(10, 7))
    
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, edgecolors='k')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Biplot')
    
    # Plot the loadings (features)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    for i, feature in enumerate(columns_of_interest):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                  head_width=0.05, head_length=0.1, 
                  fc='k', ec='k')
        plt.text(loadings[i, 0] * 1.2, loadings[i, 1] * 1.2, 
                 feature, color='k', ha='center', va='center')
    
    plt.grid(True)
    plt.show()

#############################
# Example usage

columns_of_interest = ['Length', 'Left', 'Right', 'Bottom', 'Top', 'Diagonal']

X = df[columns_of_interest]
y = df['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_pca, X_test_pca, pca = pca_test(X_train, X_test)

accuracy, classification = train_and_evaluate_model(X_train_pca, X_test_pca, y_train, y_test)
print("Model accuracy:", accuracy)
print("Classification Report:\n", classification)

pca_biplot(df)