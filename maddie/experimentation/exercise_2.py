#!/usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.model_selection as model_selection

def mixture(size):
    distribution_1 = np.random.poisson(1, size=size)
    distribution_2 = np.random.poisson(5, size=size)
    uniform_dist = np.random.uniform(1, size=size)
    return np.where(uniform_dist < 0.99, distribution_1, distribution_2)

def ab_test(data):
    sample_a, sample_b = model_selection.train_test_split(mixture(10000), test_size=0.5)
    rate_a, rate_b = sum(sample_a) / len(sample_a), sum(sample_b) / len(sample_b)
    probability = rate_a / (rate_a + rate_b)

    bookings_a = sum(sample_a)
    bookings_b = stats.nbinom(bookings_a, probability)
    
    a = np.arange(stats.nbinom.ppf(0.01, bookings_a, probability), stats.nbinom.ppf(0.99, bookings_a, probability))
    ab = stats.nbinom.pmf(a, bookings_a, probability)
    
    # H0
    x = np.arange(stats.nbinom.ppf(0.01, bookings_a, 0.5), stats.nbinom.ppf(0.99, bookings_a, 0.5))
    xy = stats.nbinom.pmf(x, bookings_a, 0.5)
    return ab

total_users = 10000
n_sims = 1000
test_simulation = [ab_test(mixture(total_users)) for _ in range(n_sims)]
plt.hist(test_simulation, bins=25)
plt.show()

# negative binomial dist is the prob distribution of the no. successes before the rth failure in a bernoulli process. it has the same probability of p successes in each trial.





