'''
Create a power-calculator application in Python.
■   Given the traffic, variance of the KPI variable, significance level and power level, calculate a table of MDE and expected duration of runs.
'''

import numpy as np
from statsmodels.stats.power import TTestIndPower
from scipy import stats

def pooled_std(sample_a, sample_b):
    n1 = len(sample_a)
    n2 = len(sample_b)
    pooled_var = (((n1-1) * np.var(sample_a)**2) +((n2-1) * np.var(sample_b)**2))/(n1+n2-2)
    return np.sqrt(pooled_var)

def effect_size(sample_a, sample_b):
    return np.abs(np.mean(sample_a) - np.mean(sample_b)) / pooled_std(sample_a, sample_b)


# Criteria
np.random.seed(201)
sample_size = 1000
sample_a = np.random.normal(size=sample_size)
sample_b = np.random.normal(size=sample_size)
effect_size = effect_size(sample_a, sample_b)
alpha = 0.05
power = 0.80
daily_rate = 2000


num_of_users = TTestIndPower().solve_power(
        effect_size = effect_size, nobs1 = None, ratio = 1.0,
        power = power, alpha = alpha, alternative='two-sided'
)
duration = num_of_users * 24 / daily_rate


print('Effect size: %.2f' % effect_size)
print('Sample size: %.2f' % num_of_users)
print('Duration of runs: %.2f' % duration, 'hours')
