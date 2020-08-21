'''
A Cauchy distribution has an infinite mean. Show that the sampling distribution of the sample mean does not converge. What about the median?
'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

number_of_iteration = 10000

mean = []
median = []
for iteration in range(number_of_iteration):
    cauchy_dist = stats.cauchy.rvs(size = number_of_iteration)
    median.append(np.median(cauchy_dist))
    mean.append(np.mean(cauchy_dist))

plt.figure(0)
plt.hist(mean, bins = 50)
plt.savefig('cauchy_mean.png')

plt.figure(1)
plt.hist(median, bins = 50)
plt.savefig('cauchy_median.png')
