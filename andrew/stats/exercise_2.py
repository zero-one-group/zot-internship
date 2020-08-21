'''
Let X and Y be independent N(0, 1) random variables. Let Z = X^2 + Y. What’s the expected sample correlation between X and Y? What about X and Z? Elaborate on your findings.
'''

from scipy import stats
import matplotlib.pyplot as plt

X = stats.norm.rvs(size = 10000)
Y = stats.norm.rvs(size = 10000)
Z = X*X + Y

pearson_corr = stats.pearsonr(X, Y)
print("Pearson correlation coefficient (X, Y) =", pearson_corr[0])

pearson_corr = stats.pearsonr(X, Z)
print("Pearson correlation coefficient (X, Z) =", pearson_corr[0])


# As both X and Y are generated randomly, we would expect them to have zero correlation. This is shown by calculating the correlation coefficient. Z is calculated from X and Y which have zero correlation, so the correlation coefficient between X and Z should be close to zero too.

