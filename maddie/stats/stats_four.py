#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

n_sims = 1000

def generate_mean():
    cauchy_distribution = np.random.standard_cauchy(100)
    return np.mean(cauchy_distribution)

means = [generate_mean() for _ in range(n_sims)]
np.percentile(means, [1, 99])
plt.hist(means, bins=25); plt.show()

def generate_median():
    return np.median(cauchy_distribution)

medians = [generate_median() for _ in range(n_sims)]
np.percentile(medians, [1, 99])
plt.hist(medians, bins=25); plt.show()






    

