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
from Access import get_dataframe
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from scipy.stats import bartlett
from factor_analyzer.factor_analyzer import calculate_kmo
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# Retrieve the DataFrame from Access.py
df = get_dataframe()

# Print DataFrame info to check columns
print()
print(df.info(), "\n")

# Define columns of interest
COLUMNS_OF_INTEREST = ['variance', 'skewness', 'curtosis', 'entropy']
TARGET = 'class'

# Ensure that required columns are in the DataFrame
missing_cols = [col for col in COLUMNS_OF_INTEREST + [TARGET] if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in the DataFrame: {missing_cols}")

# Shuffle the DataFrame
df_shuffled = df.sample(frac=1, random_state=37).reset_index(drop=True)

# Calculate the split index
split_index = int(0.2 * len(df_shuffled))

# Split the DataFrame into 20% validation and 80% training
df_val = df_shuffled.iloc[:split_index].reset_index(drop=True)  # 20% for validation
df_tt = df_shuffled.iloc[split_index:].reset_index(drop=True)  # 80% for training

def plot_data(df):
    # Function to plot histograms for each column
    if df is not None and not df.empty:
        print(df.head())
        print(df.info())
        print(df.describe(include='all'))
        missing_values = df.isnull().sum()
        print("Missing values per column:")
        print(missing_values)
        num_columns = len(df.columns)
        num_rows = (num_columns // 3) + (num_columns % 3 > 0)
        plt.figure(figsize=(15, num_rows * 4))
        for i, column in enumerate(df.columns):
            plt.subplot(num_rows, 3, i + 1)
            df[column].hist(bins=30, edgecolor='black')
            plt.title(column)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
    else:
        print("Failed to retrieve data or DataFrame is empty.")

def multicolinearity_tests(df):
    # Function to perform multicollinearity tests
    df_selected = df[COLUMNS_OF_INTEREST]
    df_selected = df_selected.apply(pd.to_numeric, errors='coerce').dropna()
    if df_selected.shape[0] < 2:
        print("Not enough data points for KMO test after dropping NaNs.")
        return
    print("Multicolinearity Tests: Bartlett and KMO:\n")
    stat, p_value = bartlett(*[df_selected[col] for col in COLUMNS_OF_INTEREST])
    print("Bartlett's test statistic:", stat)
    print("p-value:", p_value)
    alpha = 0.05
    if p_value < alpha:
        print("Decision: Reject the null hypothesis: The variances are significantly different.\n")
    else:
        print("Decision: Fail to reject the null hypothesis: No significant difference in variances.\n")
    kmo_all, kmo_model = calculate_kmo(df_selected)
    kmo_df = pd.DataFrame({
        'Variable': COLUMNS_OF_INTEREST,
        'KMO Statistic': kmo_all
    })
    print("KMO decision rule:")
    print("0.8 - 1.0: Excellent")
    print("0.7 - 0.79: Good")
    print("0.6 - 0.69: Mediocre")
    print("0.5 - 0.59: Poor")
    print("Below 0.5: Unacceptable\n")
    print("KMO statistic for each variable:")
    print(kmo_df)
    print("\nKMO statistic (overall):", kmo_model, "\n")

def pca_biplot(df):
    # Function to plot PCA biplot
    X = df[COLUMNS_OF_INTEREST]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, edgecolors='k', color = '#ccffff')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Biplot')
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    for i, feature in enumerate(COLUMNS_OF_INTEREST):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                  head_width=0.05, head_length=0.1, 
                  fc='red', ec='red', 
                  linewidth = 4, color = 'red')
        plt.text(loadings[i, 0] * 1.2, loadings[i, 1] * 1.2, 
                 feature, color='black', 
                 ha='center', va='center',
                 fontsize=16, fontweight='bold')
    plt.grid(True)
    plt.show()

def pca_test(X_train, X_test, n_components=0.90):
    # Function to perform PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    print("PCA components (loadings):")
    print(pca.components_, "\n")
    print("Explained variance ratio:")
    print(pca.explained_variance_ratio_, "\n")
    print("Cumulative explained variance ratio:")
    print(cumulative_explained_variance, "\n")
    components_df = pd.DataFrame(pca.components_, columns=COLUMNS_OF_INTEREST, index=[f'PC{i+1}' for i in range(pca.n_components_)])
    print("Principal component loadings:")
    print(components_df, "\n")
    
    # Plot Scree Plot
    plt.figure(figsize=(8, 5))
    x_values = range(1, len(pca.explained_variance_ratio_) + 1)
    y_values = pca.explained_variance_ratio_
    plt.plot(x_values, y_values, 'bo-', markersize=10)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(x_values)
    plt.grid(True)
    
    # Add y-tick values on top of each node
    offset = 0.02 # to adjust the tick above the node
    for x, y in zip(x_values, y_values):
        plt.text(x, y + offset, f'{y:.2f}', fontsize=9, fontweight='bold', ha='center', va='bottom')
    
    plt.show()
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(pca, 'pca.pkl')
    
    # Plot PCA biplot (assuming df is defined elsewhere)
    pca_biplot(df)
    return X_train_pca, X_test_pca, pca

def train_and_evaluate_model(X_train_pca, X_test_pca, y_train, y_test):
    # Function to train and evaluate model
    model = RandomForestClassifier()
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    classification = classification_report(y_test, y_pred)
    joblib.dump(model, 'random_forest_model.pkl')
    
    print(f"Model accuracy: {accuracy:.2f}")
    print("Classification report:\n", classification)
    
    return accuracy, classification

def class_distribution(df, name):
    # Assuming TARGET is a column in the DataFrame df
    class_distribution = df[TARGET].value_counts()
    print(f"Class distribution in {name} dataset:\n", class_distribution, "\n")
    
    # Plot class distribution
    plt.figure(figsize=(8, 5))
    bars = class_distribution.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title(f"Class Distribution in {name} dataset")
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(ticks=[0, 1], labels=['Class 0', 'Class 1'], rotation=0)
    plt.grid(axis='y')
    
    # Add y-tick values on top of each bar
    for bar in bars.patches:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, 
                 f'{height}', ha='center', va='bottom')
    
    plt.show()
    
    return df

def balance_data(df):
    # Function to balance the dataset
    df_majority = df[df[TARGET] == 0]
    df_minority = df[df[TARGET] == 1]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    return df_balanced

# Perform multicolinearity test:
multicolinearity_tests(df_tt)

# Check class distribution in the validation set
check_class_distribution = class_distribution(df_tt, "original")

# Balance the dataset
df_balanced = balance_data(df_tt)

# Check the new class distribution
balanced_class_distribution = class_distribution(df_balanced, "balanced")

# Split into features and target variable
X = df_balanced[COLUMNS_OF_INTEREST]
y = df_balanced[TARGET]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Perform PCA and get transformed data
X_train_pca, X_test_pca, pca = pca_test(X_train, X_test)

# Train and evaluate the model
accuracy, classification = train_and_evaluate_model(X_train_pca, X_test_pca, y_train, y_test)

