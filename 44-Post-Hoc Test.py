'''
we do anova to determine  if 
there are any statistically significant differences between the means 
of three or more independent (unrelated) groups.
More precisely,  if:
Result: Significant (p < 0.05),means at least one group differs from the others.

However, it does not tell us which specific groups 
differ from each other—only that there is some difference somewhere among the groups.

Post-hoc tests are performed after finding a significant result from ANOVA.
They help to determine exactly which pairs of groups differ from each other.
The test is Tukey's HSD.

We Control for Type I Errors: 
Post-hoc tests adjust for the increased risk of Type I errors (false positives) 
that occur when multiple comparisons are made.

In summary, ANOVA tells you whether there's a significant effect overall, 
while post-hoc tests help you pinpoint where the differences are between specific groups.

'''

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.stats.multicomp import MultiComparison

# Sample data
# Let's assume we have data from 3 groups
np.random.seed(0)
group_1 = np.random.normal(loc=5, scale=1, size=30)
group_2 = np.random.normal(loc=6, scale=1, size=30)
group_3 = np.random.normal(loc=7, scale=1, size=30)

# Combine the data into a DataFrame
data = pd.DataFrame({
    'Value': np.concatenate([group_1, group_2, group_3]),
    'Group': ['Group 1']*30 + ['Group 2']*30 + ['Group 3']*30
})

# Perform one-way ANOVA
model = ols('Value ~ C(Group)', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("ANOVA Results:")
print(anova_table)

# Check if ANOVA is significant
if anova_table['PR(>F)'][0] < 0.05:
    print("\nANOVA is significant, proceeding with Tukey's HSD test...")
    
    # Perform Tukey's HSD test
    mc = MultiComparison(data['Value'], data['Group'])
    tukey_result = mc.tukeyhsd()
    
    print("\nTukey's HSD Test Results:")
    print(tukey_result)
else:
    print("\nANOVA is not significant, no need for post-hoc test.")
    
    '''
    intepretation.
    Tukey's HSD Test Results:
 Multiple Comparison of Means - Tukey HSD, FWER=0.05 
=====================================================
 group1  group2 meandiff p-adj   lower  upper  reject
-----------------------------------------------------
Group 1 Group 2   0.2676 0.5536 -0.3458  0.881  False
Group 1 Group 3   1.4234    0.0    0.81 2.0368   True
Group 2 Group 3   1.1558 0.0001  0.5424 1.7692   True
-----------------------------------------------------

Group 1 Group 2 - p-value> 0.05, so we don't reject the null. 
meaning: there is no significant difference between Group 1 and Group 2.
The other combos we reject:
Group 1 is significantly different from Group 3.
Group 2 is significantly different from Group 3.

now we know which grops are significantly different.

    '''
