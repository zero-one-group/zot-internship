'''
Cox and Lewis (1966) reported 799 time intervals between pulses on a nerve fibre. The dataset can be downloaded here. Use the bootstrap to get confidence intervals for the median and skewness of these data. In particular, present basic bootstrap, bootstrap-t and percentile confidence intervals.
'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import re

def read_data(filename):
    with open(filename, "r") as file:
        data = file.read()
        data = re.split('\t|\n', data)
    return list(map(float, data))

def bootstrap(data, num_of_simulation):
    bootstrap_sample = []
    for idx in range(num_of_simulation):
        randomly_chosen_sample = [data[np.random.randint(len(data))]
                                  for idx in range(len(data))]
        bootstrap_sample.append(randomly_chosen_sample)
    return np.array(bootstrap_sample)

def confidence_interval(data, confidence_level):
    parameter = np.mean(data)
    alpha = 1 - confidence_level
    upper_bound = (2 * parameter) + stats.t.ppf(1-(alpha/2), len(data) - 1)
    lower_bound = (2 * parameter) - stats.t.ppf(1-(alpha/2), len(data) - 1)
    return lower_bound, upper_bound

def bootstrap_t(data, confidence_level):
    standard_error = stats.sem(data)
    upper_bound = np.mean(data) + (np.percentile(data, confidence_level) * standard_error)
    lower_bound = np.mean(data) - (np.percentile(data, confidence_level) * standard_error)
    return lower_bound, upper_bound

data = read_data('nerve.txt')
bootstrap_sample = bootstrap(data, num_of_simulation=10000)

median = np.median(bootstrap_sample, axis=1)
plt.figure(0)
plt.hist(median, bins=10, label='Median (bootstrap)')
plt.savefig('median.png')

skewness = stats.skew(bootstrap_sample, axis=1)
plt.figure(1)
plt.hist(skewness, bins=50, label='Skewness (bootstrap)')
plt.savefig('skewness.png')

# Basic bootstrap confidence interval
print("Median CI (Basic) =", confidence_interval(median, confidence_level=0.95))
print("Skewness CI (Basic) =", confidence_interval(skewness, confidence_level=0.95))

# Bootstrap-t confidence interval
print("Median CI (Bootstrap-t) =", bootstrap_t(median, confidence_level=0.95))
print("Skewness CI (Bootstrap-t) =", bootstrap_t(skewness, confidence_level=0.95))

# Percentile confidence interval
print("Median CI (Percentile)=", np.percentile(median, [2.5, 97.5]))
print("Skewness CI (Percentile)=", np.percentile(skewness, [2.5, 97.5]))


