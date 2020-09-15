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

def simulate_users(total_user, pois_a, pois_b):
    user_a = np.random.poisson(lam=pois_a, size=total_user)
    user_b = np.random.poisson(lam=pois_b, size=total_user)
    user_uniform = np.random.uniform(size=total_user)
    return np.where(user_uniform < 0.99, user_a, user_b)


def split_traffic(users, probability):
    traffic_a = int(probability*len(users))
    traffic_b = int((1-probability)*len(users))
    test_a = np.random.choice(users, size=traffic_a)
    test_b = np.random.choice(users, size=traffic_b)
    return test_a, test_b


def pval(users):
    test_a, test_b = split_traffic(users, 0.5)
    actions_in_a = sum(test_a)
    actions_in_b = sum(test_b)
    p_val = stats.nbinom.cdf(actions_in_b, actions_in_a, 0.5)
    return p_val


def pval_bookers(total_users):
    """if we can observe non-bookers"""
    test_a, test_b = split_traffic(simulate_users(total_users, 1.0, 5.0), 0.5)
    actions_in_a = sum(test_a[test_a>0]) / len(test_a)
    actions_in_b = sum(test_b[test_b>0]) / len(test_b)
    p_val = stats.nbinom.cdf(actions_in_b, actions_in_a, 0.5)
    return p_val


def power(effect_size, sample1, alpha):
    return sm.stats.TTestIndPower().power(effect_size, sample1, alpha)


def trim(data, top_percent):
    return sorted(data)[:-int(top_percent*len(data))]


def pval_welch_booker(users):
    """if we can observe non-bookers"""
    test_a, test_b = split_traffic(users, 0.5)
    t_stat, p_val = stats.ttest_ind(test_a, test_b, equal_var=False)
    return p_val

def effect_size(test_a, test_b):
    return np.mean(test_a) - np.mean(test_b) / np.sqrt((np.std(test_a)**2 + np.std(test_b)**2)/2)
    

def pval_welch_d(user_a, user_b):
    test_a = split_traffic(user_a, 0.5)
    test_b = split_traffic(user_b, 0.5)
    t_stat, p_val = stats.ttest_ind(test_a, test_b, equal_var=False)
    test_power = power(effect_size(test_a, test_b), len(test_a), 0.05)
    return p_val, test_power

nsims = 10000

#i
pvals = [pval(simulate_users(10000, 1, 5)) for _ in range(nsims)]
plt.figure()
plt.hist(pvals, bins=100)
plt.savefig('exp2.png')

#ii
pvals_bookers = [pval_bookers(simulate_users(10000, 1, 5)) 
                 for _ in range(nsims)]
plt.figure()
plt.hist(pvals_bookers, bins=100)
plt.savefig('exp2_ii.png')

#iii
pvals_welch = [pval_welch_booker(simulate_users(10000, 1, 5)) 
               for _ in range(nsims)]
plt.figure()
plt.hist(pvals_welch, bins=100)
plt.savefig('exp2_iii.png')

#iv
print("Power of test is ", pval_welch_d(simulate_users(10000, 1, 5),
                                        simulate_users(10000, 1.03, 5))[1])

#v
pvals_trimmed = [pval_welch_booker(trim(simulate_users(10000, 1, 5), 0.02))
                 for _ in range(nsims)]
plt.figure()
plt.hist(pvals_trimmed, bins=100)
plt.savefig('exp2_v_1.png')

print("Power of test after trimmed is ", pval_welch_d(trim(
    simulate_users(10000, 1, 5), 0.02), 
    trim(simulate_users(10000, 1.03, 5), 0.02))[1])
