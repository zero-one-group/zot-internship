'''
An antiquated generator for the normal distribution is:
Generate U1, U2, ... U12 which are iid uniform distribution between -0.5 and 0.5
Set Z as sum of Ui where i = 1, .., 12.
the argument being that the CLT normality is sufficiently accurate with 12 terms.

a. Show that E[Z] = 0 and var(Z) = 1.
b. Using histograms, compare this CLT-normal generator with the Box–Muller
algorithm. Pay particular attention to tail probabilities.
c. Compare both of the generators in part a. with rnorm.
Note that this exercise does not suggest using the CLT for normal generations!
This is a very poor approximation indeed.
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

uniform_distribution = [stats.uniform.rvs(loc=-0.5, scale=1, size=int(1e5)) for i in range(1, 13)]
Z = sum(uniform_distribution)
mean = np.mean(Z)
variance = stats.tstd(Z) ** 2
print("Mean =", mean)
print("Variance =", variance)

uniform_distribution_1 = stats.uniform.rvs(size=int(1e5))
uniform_distribution_2 = stats.uniform.rvs(size=int(1e5))
X = np.sqrt(-2 * np.log(uniform_distribution_1)) * np.cos(2 * np.pi * uniform_distribution_2)
plt.figure(0)
plt.hist(X, bins=100, alpha=0.5, label='Box-muller')
plt.hist(Z, bins=100, alpha=0.5 ,label='Central Limit Theorem')
plt.legend(loc = 'upper right')
plt.savefig('generator.png')

normal_distribution = stats.norm.rvs(size=int(1e5))
plt.figure(1)
plt.hist(normal_distribution, bins=100, alpha=0.5, label='Normal distribution')
plt.hist(Z, bins=100, alpha=0.5, label='Central Limit Theorem')
plt.legend(loc = 'upper right')
plt.savefig('generator_2.png')
