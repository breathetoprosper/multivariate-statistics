import pandas as pd
import statsmodels.api as sm

# Sample data
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Class_Attendance': [0, 1, 1, 1, 1, 0, 1, 0, 1, 1],
    'Pass': [0, 0, 1, 1, 1, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# Define independent variable
X = df[['Hours_Studied']]
X = sm.add_constant(X)

# Define dependent variable
y = df['Pass']

# Fit the Probit regression model (Pro-bability un-It). it represents the Z-score associated with a given probability in a normal distribution.
try:
    probit_model = sm.Probit(y, X)
    probit_result = probit_model.fit()
    
    # Output the summary of the model
    print(probit_result.summary())
except Exception as e:
    print("Error:", e)

# Define significance level
alpha = 0.05

# Extract p-value for Hours_Studied
p_value_hours_studied = probit_result.pvalues['Hours_Studied']

print("\nConclusion:")
if p_value_hours_studied < alpha:
    print("\nReject the null hypothesis: There is a significant effect of hours studied on passing the exam.")
else:
    print("\nFail to reject the null hypothesis: There is no significant effect of hours studied on passing the exam.")
