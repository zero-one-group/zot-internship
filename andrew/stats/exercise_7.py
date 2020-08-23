'''
Suppose we have 100 i.i.d draws of Beta(1, 2) distribution. Simulate the events 1000 times, and each time saving the 95% confidence interval of the mean. 
■   How many of your intervals contain the true mean?
■   Construct the 90% confidence interval for the probability that the 95% confidence interval contains the true mean.
■   Explain, in your own words, what does 95% mean in the 95% confidence interval?
'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def mean_confidence_interval(data, confidence):
    data_array = 1.0 * np.array(data)
    num = len(data_array)
    mean, standard_err = np.mean(data_array), stats.sem(data_array)
    space = standard_err * stats.t.ppf((1 + confidence) / 2., num - 1)
    return mean, mean-space, mean+space

def double_sided_z_score(probability):
    return stats.norm.ppf(probability + ((1-probability)/2))

def CI_for_population_proportion(proportion, CI, sample_size):
    space = double_sided_z_score(CI) * np.sqrt(proportion*(1-proportion)/sample_size)
    return proportion - space, proportion + space


intervals = []
for simulation in range(1000):
    beta_dist = np.random.beta(1, 2, size = 100)
    mean, lower_bound, upper_bound = mean_confidence_interval(beta_dist, 0.95)
    intervals.insert(simulation, [lower_bound, upper_bound])

true_mean = 1/3
CI_95 = [intervals[idx] for idx in range(len(intervals))
        if intervals[idx][0] < true_mean < intervals[idx][1]]
print("Number of intervals containing true mean =", len(CI_95))


proportion = len(CI_95) / len(intervals)
lower_end, upper_end = CI_for_population_proportion(proportion, 0.9, len(intervals))
print("90% CI for the probability that 95% CI contains the true mean =", lower_end, "to", upper_end)

# 95% in 95% confidence interval is how sure you can be that the true mean lies within the intervals. The 95% confidence interval defines a range of values that you can be 95% certain contains the true (population) mean.
