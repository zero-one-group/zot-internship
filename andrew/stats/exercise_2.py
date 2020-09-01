'''
Let X and Y be independent N(0, 1) random variables. Let Z = X^2 + Y. What’s the expected sample correlation between X and Y? What about X and Z? Elaborate on your findings.
'''

import matplotlib.pyplot as plt
from scipy import stats

xs = stats.norm.rvs(size=int(1e5))
ys = stats.norm.rvs(size=int(1e5))
zs = xs*xs + ys

pearson_corr = stats.pearsonr(xs, ys)
print("Pearson correlation coefficient (X, Y) =", pearson_corr[0])
plt.figure(0)
plt.scatter(xs, zs)
plt.show()

pearson_corr = stats.pearsonr(xs, zs)
print("Pearson correlation coefficient (X, Z) =", pearson_corr[0])


# As both X and Y are generated randomly, we would expect them to have zero correlation. This is shown by calculating the correlation coefficient. Z is calculated from X and Y which have zero correlation, so the correlation coefficient between X and Z should be close to zero too.

