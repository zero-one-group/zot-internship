'''
a. Generate gamma and beta random variables
b. Show that if U ∼ U[0,1], then X = − log U/λ ∼ Exp(λ).
c. Show that if U ∼ U[0,1], then X = log u/(1−u) is a Logistic(0, 1) random variable.
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

gamma_dist = stats.gamma.rvs(a=1, size=int(1e5))
beta_dist = stats.beta.rvs(a=3, b=0.5, size=int(1e5))
plt.figure(0)
plt.hist(gamma_dist, bins=100, alpha=0.5, label = 'Gamma distribution')
plt.hist(beta_dist, bins=100, alpha=0.5, label = 'Beta distribution')
plt.legend(loc = 'upper right')
plt.savefig('gamma_and_beta_distribution.png')


param = 5 #lambda value
uniform_dist = stats.uniform.rvs(size=int(1e5))
X_expon = -np.log(uniform_dist) / param
exponential_dist = stats.expon.rvs(scale=1/param, size=int(1e5)) 
plt.figure(1)
plt.hist(exponential_dist, bins=100, alpha=0.5, label='Exponential distribution')
plt.hist(X_expon, bins=100, alpha=0.5, label='X = -ln(U)/λ')
plt.legend(loc='upper right')
plt.savefig('exponential_dist')


logistic_dist = stats.logistic.rvs(size=int(1e5))
X_logistic = np.log(uniform_dist / (1 - uniform_dist))
plt.figure(2)
plt.hist(logistic_dist, bins=100, alpha=0.5, label='Logistic distribution')
plt.hist(X_logistic, bins=100, alpha=0.5, label='X = -ln(U/(1-U))')
plt.legend(loc = 'upper right')
plt.savefig('logistic_dist')

