'''
In each of the following cases, construct an Accept–Reject algorithm,
generate a sample of the corresponding random variables, and draw the density function
on top of the histogram.
a. Generate normal random variables using a Cauchy candidate in Accept–Reject.
b. Generate gamma G(4.3, 6.2) random variables using a gamma G(4, 7) candidate.
'''

from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

random_uniform_numbers_y = stats.uniform.rvs(0, 0.4, size = 10000)
random_uniform_numbers_x = stats.uniform.rvs(-8, 16, size = 10000)
cauchy_pdf = 1/(np.pi * (1 + (random_uniform_numbers_x ** 2)))

plt.figure(0)
for idx in range(len(random_uniform_numbers_x)):
    if random_uniform_numbers_y[idx] < cauchy_pdf[idx]:
        plt.plot(random_uniform_numbers_x[idx], random_uniform_numbers_y[idx], 'b+')
    else:
        plt.plot(random_uniform_numbers_x[idx], random_uniform_numbers_y[idx], 'r+')
plt.savefig('accept_reject_cauchy')


gamma_candidate = stats.gamma.rvs(4, 7, size = 10000)
gamma_distribution = stats.gamma.rvs(4.3, 6.2, size = 10000)
plt.figure(1)
plt.hist(gamma_candidate, bins = 100, alpha = 0.5)
plt.hist(gamma_distribution, bins = 100, alpha = 0.5)
plt.savefig('hist')

