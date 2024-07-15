'''
Testes de Multicolinearidade

Bartlett – Testa a hipótese da matriz de correlações entre as variáveis ser uma
matriz identidade – não existe correlação entre nenhum par de variáveis.
H0: A matriz de correlações é uma matriz de identidade
Ha: A matriz de correlações não é uma matriz de identidade
Interessa rejeitar a H0
'''

import numpy as np
from scipy.stats import bartlett
from factor_analyzer.factor_analyzer import calculate_kmo

# Example data for variables (e.g., 5 variables and 10 observations)
data = np.array([
    [10, 12, 11, 14, 13],
    [15, 16, 14, 17, 15],
    [20, 22, 21, 19, 23],
    [14, 15, 13, 14, 16],
    [11, 12, 10, 12, 11],
    [16, 18, 17, 20, 18],
    [21, 22, 20, 23, 21],
    [13, 14, 12, 14, 13],
    [17, 18, 16, 19, 18],
    [22, 24, 23, 25, 22]
])

# Example data for Bartlett's test
group1 = np.array([10, 12, 11, 14])
group2 = np.array([15, 16, 14, 17])
group3 = np.array([20, 22, 21, 19])

# Perform Bartlett's test
stat, p_value = bartlett(group1, group2, group3)

# Perform KMO test
kmo_all, kmo_model = calculate_kmo(data)

# Print Bartlett's test results
print()
print("Bartlett's test statistic:", stat)
print("p-value:", p_value)
if p_value < 0.05:
    print("Reject the null hypothesis: The variances are significantly different.")
else:
    print("Fail to reject the null hypothesis: No significant difference in variances.")

print()
# Print KMO test results
print("KMO Test: The KMO statistic ranges from 0 to 1, \nwith values closer to 1 indicating that your data is suitable for factor analysis.")
print("0.8 - 1.0: Excellent")
print("0.7 - 0.79: Good")
print("0.6 - 0.69: Mediocre")
print("0.5 - 0.59: Poor")
print("Below 0.5: Unacceptable")
print()
print("KMO statistic (overall):", kmo_model)

print()
print("note: If Bartlett's test and the KMO measure yield different results, it suggests that there might be issues with the data's suitability for PCA, so proceeding with caution and carefully interpreting the results is advised.")
