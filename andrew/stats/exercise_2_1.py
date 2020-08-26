'''Show the inverse transformation by using N(5, 9) and Exp(12)'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Inverse Transform Sampling applied to exponential distribution
random_numbers = np.random.rand(int(1e5))
exponential_dist = list(np.random.exponential(scale=1/12, size=int(1e5)))
exponential_inverse = (-1/12) * np.log(1 - random_numbers)

plt.figure(0)
plt.hist(exponential_dist, bins=100, alpha=0.5, label='Exponential Distribution')
plt.hist(exponential_inverse, bins=100, alpha=0.5, label='Inverse transformation')
plt.xlim(0, 1)
plt.legend(loc = 'upper right')
plt.savefig('Exponential_dist.png')


# The inverse transform sampling method cannot be used to generate normally distributed random numbers as the cdf for the normal distribution is not available analytically.Use Box-muller method instead.
uniform_distribution_1 = stats.uniform.rvs(size=int(1e5))
uniform_distribution_2 = stats.uniform.rvs(size=int(1e5))
X = np.sqrt(-2 * np.log(uniform_distribution_1)) * np.cos(2 * np.pi * uniform_distribution_2)
# X is normally distributed random variable with mean 0 and variance 1. The transformation Z = σX + µ will give Z ∼ Normal(µ, σ^2).
Z = (3 * X) + 5
normal_random_variates = stats.norm.rvs(5, 3, size=int(1e5))

plt.figure(1)
plt.hist(Z, bins=100, alpha=0.5, label='Box-muller method')
plt.hist(normal_random_variates, bins=100, alpha=0.5, label='N(5, 9)')
plt.legend(loc = 'upper right')
plt.savefig('Normal_dist.png')
