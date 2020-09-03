#!/usr/bin/env python

import numpy as np
from scipy import stats
import statsmodels.stats.proportion as sm

n_points = 100
n_sims = 1000

def simulate_one_realisation():
    return np.random.beta(1, 2, size=n_points)

def generate_mean(sample):
    return np.mean(sample)


# Because the 95% confidence interval of the mean was taken and in total 100,000 points were taken, 95,000 of the 100,000 intervals will contain the true mean value

confidence_level = 0.95
degrees_freedom = 100000 - 1
sample_standard_error = [stats.sem(simulate_one_realisation()) for _ in range(n_sims)]
sample_mean = [generate_mean(simulate_one_realisation()) for _ in range(n_sims)]
ninety_five_confidence_interval = stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
print(ninety_five_confidence_interval)

# Contruct 90% confidence interval for probability that 95% confidence interval contains the true mean

ninety_confidence_interval = sm.proportion_confint(0.95, 100000, alpha=0.1, method='beta')
print(ninety_confidence_interval)

# The 95% confidence interval is the range of values where one can be 95% certain that it contains the true mean of the sample. The 95% is thus a probability that the true value will lie between the upper and lower bound of a probability distribution. 
