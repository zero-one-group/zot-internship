'''
A Cauchy distribution has an infinite mean. Show that the sampling distribution of the sample mean does not converge. What about the median?
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

n_sims = 1000
n_rows = 10000

cauchies = stats.cauchy.rvs(size=n_rows*n_sims).reshape(n_sims, n_rows)
means = np.mean(cauchies, axis=1)
medians = np.median(cauchies, axis=1)

plt.figure(0)
plt.hist(means, bins=50)
plt.savefig('cauchy_mean.png')

plt.figure(1)
plt.hist(medians, bins=50)
plt.savefig('cauchy_median.png')
