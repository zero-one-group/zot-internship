'''
Two distributions that have explicit forms of the cdf are the logistic and Cauchy distributions. Thus, they are well-suited to the inverse transform method. For each of the following, verify the form of the cdf and then generate 10,000 random variables using the inverse transform.
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

random_numbers = np.random.rand(int(1e5))
logistic_dist = stats.logistic.rvs(size=int(1e5))
logistic_inverse = np.log(random_numbers/(1-random_numbers))
plt.figure(0)
plt.hist(logistic_dist, bins=100, alpha=0.5, label='Logistic distribution')
plt.hist(logistic_inverse, bins=100, alpha=0.5, label='Inverse transformation')
plt.legend(loc = 'upper right')
plt.savefig('logistic.png')



cauchy_dist = stats.cauchy.rvs(size=int(1e5))
cauchy_inverse = np.tan(np.pi * (random_numbers - 0.5))
plt.figure(1)
plt.hist(cauchy_inverse, bins=1000, alpha=0.5, label='Inverse transformation')
plt.hist(cauchy_dist, bins=1000, alpha=0.5, label='Cauchy distribution')
plt.legend(loc = 'upper right')
plt.xlim(-5000, 5000)
plt.savefig('cauchy.png')
