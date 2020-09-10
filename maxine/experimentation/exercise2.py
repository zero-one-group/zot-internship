"""Suppose the number of bookings of a user is i.i.d Poisson mixture of Pois(1)
and Pois(5) with weights 99% and 1% respectively. Think of the mixture
distribution as a combination of regular travellers and travel agents.
Moreover, we only observe a user only if they make at least one booking. Think
of this as a mixture of legitimate users that may book and illegitimate bots
that would never book. The legitimate non-bookers are indistinguishable from
the illegitimate bots.
- Consider the negative binomial test (part 2). Assume there are no booking-rate
differences in A and B (i.e. assuming the null hypothesis). Simulate A/B tests
with 10k users, and show that applying the negative binomial test on bookings
yields an inverted-U p-value sampling distribution.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm


user_a = np.random.poisson(lam=1.0, size=9900)
user_b = np.random.poisson(lam=5.0, size=100)
user_simulate = np.concatenate((user_a, user_b))

total_users = 10000
test_a = user_simulate[:5000]
test_b = user_simulate[5000:]
cr_a = sum(test_a)/len(test_a)
cr_b = sum(test_b)/len(test_b)
p = cr_a / (cr_a + cr_b)

actions_in_a = sum(test_a)
actions_in_b = stats.nbinom(actions_in_a, p) 

x = np.arange(stats.nbinom.ppf(0.01, actions_in_a, p), 
stats.nbinom.ppf(0.99, actions_in_a, p))
ac_x = stats.nbinom.pmf(x, actions_in_a, p)
y = np.arange(stats.nbinom.ppf(0.01, actions_in_a, 0.5), 
stats.nbinom.ppf(0.99, actions_in_a, 0.5))
ac_y = stats.nbinom.pmf(x, actions_in_a, 0.5)
ttest, pval = stats.ttest_ind(ac_x, ac_y)
pval

"""
accepted_x = [x[idx] for idx in range(len(x)) if decision_boundary[0] 
              <= x[idx] <= decision_boundary[1]]
#H0 = p = 0.5
#non bookers = expeceted 0
stats.binom_test(1, len(test_b),p) 
decision_boundary = stats.nbinom.interval(alpha, actions_in_a, 0.5)
"""
