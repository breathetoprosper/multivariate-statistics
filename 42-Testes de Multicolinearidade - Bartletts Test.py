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

# Example data for three groups
group1 = np.array([10, 12, 11, 14])
group2 = np.array([15, 16, 14, 17])
group3 = np.array([20, 22, 21, 19])

# Perform Bartlett's test
stat, p_value = bartlett(group1, group2, group3)

# Print results
print("Bartlett's test statistic:", stat)
print("p-value:", p_value)

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The variances are significantly different.")
else:
    print("Fail to reject the null hypothesis: No significant difference in variances.")
