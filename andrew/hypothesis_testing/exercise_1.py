'''
Let X ~ Beta(1.8, 1). An unsuspecting researcher has 100 i.i.d samples of X, and would like to conduct the following test at 10% significance - H0: E(X) = 2, H1: E(X) < 1.
■   What’s the probability that the researcher rejects the null hypothesis?
■   What does 10% in 10% significance level mean? Show your argument using a simulation.
■   Explain, in your own words, what Type I and Type II errors are.

'''

import numpy as np
from scipy import stats

num_of_samples = 100
beta_dist = np.random.beta(1.8, 1, size=num_of_samples)
sample_mean = np.mean(beta_dist)
sample_std = np.std(beta_dist)
h_0 = 0.65
alpha = 0.1

# Test statistic is not normal and population standard deviation is unknown, use T-test
t_value = (sample_mean-h_0) / (sample_std/np.sqrt(num_of_samples))
critical_value = stats.t.ppf(1-alpha, df=num_of_samples-1) #one-sided test
p_value = stats.t.sf(np.abs(t_value), num_of_samples-1) #one-sided test
print("T-value =", t_value)
print("Critical value =", critical_value)
print("Probability of rejecting null hypothesis =", 1-p_value)

# 10% significance level means that there is a 10% probability of rejecting the null hypothesis when it is true
# A type I error refers to the situation where the null hypothesis has been rejected even though it is true. A type II error occurs when a null hypothesis is not rejected even though it is false.
