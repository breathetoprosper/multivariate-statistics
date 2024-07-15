import pandas as pd
import statsmodels.api as sm

# (Lo-garithm un-It, here it suggests odds). Meaning the log-odds of an event happening. 
# logit(p) = log(p/(1-p))
# Sample data.
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Class_Attendance': [0, 1, 1, 1, 1, 0, 1, 0, 1, 1],
    'Pass': [0, 0, 1, 1, 1, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# Check for multicollinearity
print(df.corr())

# Using only Hours_Studied
X = df[['Hours_Studied']]
X = sm.add_constant(X)

# Define dependent variable
y = df['Pass']

# Fit the logistic regression model
try:
    model = sm.Logit(y, X)
    result = model.fit()
    
    # Output the summary of the model
    print(result.summary())
except Exception as e:
    print("Error:", e)

# Define significance level
alpha = 0.05

# Extract p-value for Hours_Studied
p_value_hours_studied = result.pvalues['Hours_Studied']

print("\nConclusion:")
if p_value_hours_studied < 0.05:
    print("\nReject the null hypothesis: There is a significant effect of hours studied on passing the exam.")
else:
    print("\nFail to reject the null hypothesis: There is no significant effect of hours studied on passing the exam.")
