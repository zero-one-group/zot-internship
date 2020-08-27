"""A Cauchy distribution has an infinite mean. Show that the sampling
distribution of the sample mean does not converge. What about the
median?"""

import numpy as np
import matplotlib.pyplot as plt

def cauchy():
    nsims = 100000
    return np.random.standard_cauchy(nsims)

def take_mean(sample):
    return np.mean(sample) 


def take_median(sample):
    return np.median(sample) 


means = [take_mean(cauchy()) for _ in range(1000)]
medians = [take_median(cauchy()) for _ in range(1000)]

np.percentile(means, [1,99])

np.percentile(medians, [1,99])

