"""Suppose we have 100 i.i.d. draws from U[0, θ].Consider the max of the 100 draws as an
estimator for θ.Approximate the estimator’s sampling distribution by simulating the events 1000 times
What is the sampling distribution of the estimated RMSE of the estimator? Is it biased?
Is it consistent?
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def uniform(theta, draws):
    return np.random.uniform(0, theta, size=draws)

def max_uniform_estimator(theta, draws, nsim):
    return list(max(uniform(theta,draws)) for sim in range(nsim))


def rmse(array1, array2):
    return np.sqrt(np.square(np.subtract(array1, array2)).mean())

estimator1 = max_uniform_estimator(10, 100, 1000)
    
plt.hist(estimator1, bins=50)
plt.savefig('test.png')

rmse(estimator1, 10)
