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

def simulate_users(total_user, pois_a, pois_b, weight_a):
    user_a = np.random.poisson(lam=pois_a, size=total_user)
    user_b = np.random.poisson(lam=pois_b, size=total_user)
    user_uniform = np.random.uniform(size=total_user)
    return np.where(user_uniform < weight_a, user_a, user_b)


def split_traffic(users):
    variants = np.random.choice(['a', 'b'], size=len(users))
    test_a = users * (variants == 'a')
    test_b = users * (variants == 'b')
    return test_a, test_b


def pval_booking(users):
    test_a, test_b = split_traffic(users)
    actions_in_a = sum(test_a)
    actions_in_b = sum(test_b)
    p_val = stats.nbinom.cdf(actions_in_b, actions_in_a, 0.5)
    return p_val


def pval_bookers(users):
    """if we observe on bookers"""
    test_a, test_b = split_traffic(users)
    actions_in_a = sum(test_a>0)
    actions_in_b = sum(test_b>0)
    p_val = stats.nbinom.cdf(actions_in_b, actions_in_a, 0.5)
    return p_val

def power(pvals, alpha):
    return np.mean([pval > (1 - alpha) for pval in pvals])


def trim(data, top_percent):
    return data < np.percentile(data, top_percent)


def pval_welch_booker(users):
    """if we can observe non-bookers"""
    test_a, test_b = split_traffic(users)
    t_stat, p_val = stats.ttest_ind(test_a, test_b, equal_var=False)
    return p_val

def effect_size(test_a, test_b):
    return np.mean(test_a) - np.mean(test_b) / np.sqrt((np.std(test_a)**2 + np.std(test_b)**2)/2)
    

def pval_welch_two_users(user_a, user_b):
    test_a, test_b = split_traffic(user_a)
    another_test_a, another_test_b = split_traffic(user_b)
    t_stat, p_val = stats.ttest_ind(test_a, another_test_b, equal_var=False)
    return p_val

nsims = 1000

#i
pvals = [pval_booking(simulate_users(10000, 1, 5, 0.99)) for _ in range(nsims)]
plt.figure()
plt.hist(pvals, bins=100)
plt.savefig('exp2.png')

#ii
pvals_bookers = [pval_bookers(simulate_users(10000, 1, 5, 0.99)) 
                 for _ in range(nsims)]
plt.figure()
plt.hist(pvals_bookers, bins=100)
plt.savefig('exp2_ii.png')

#iii
pvals_welch = [pval_welch_booker(simulate_users(10000, 1, 5, 0.99)) 
               for _ in range(nsims)]
plt.figure()
plt.hist(pvals_welch, bins=100)
plt.savefig('exp2_iii.png')

#iv
pvals_two_users = [pval_welch_two_users(simulate_users(10000, 1, 5, 0.99),
                                        simulate_users(10000, 1.03, 5, 0.99))
                   for _ in range(nsims)]
print("Power of test is ", power(pvals_two_users, 0.05))

#v
pvals_trimmed = [pval_welch_two_users(trim(simulate_users(10000, 1, 5, 0.99),
                                           98),
                                      trim(simulate_users(10000, 1.03, 5,
                                                          0.99), 98)) for _ in range(nsims)]
print("Power of test after trimmed is ", power(pvals_trimmed, 0.05))

plt.figure()
plt.hist(pvals_trimmed, bins=100)
plt.savefig('exp2_v_1.png')

