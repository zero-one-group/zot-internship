'''
Suppose the number of bookings of a user is i.i.d Poisson mixture of Pois(1) and Pois(5) with weights 99% and 1% respectively. Think of the mixture distribution as a combination of regular travellers and travel agents. Moreover, we only observe a user only if they make at least one booking. Think of this as a mixture of legitimate users that may book and illegitimate bots that would never book. The legitimate non-bookers are indistinguishable from the illegitimate bots.
- Consider the negative binomial test (part 2). Assume there are no booking-rate differences in A and B (i.e. assuming the null hypothesis). Simulate A/B tests with 10k users, and show that applying the negative binomial test on bookings yields an inverted-U p-value sampling distribution.
- Show that applying the same test on bookers yields a uniform p-value sampling distribution. Why?
- Now assume that you can observe the non-bookers as well. Design the Welch test that takes into account the number of bookings, and show tha your experiment design yields a uniform p-value sampling distribution.
- Suppose that for the users that received the B variant, their Poisson mixture is Pois(1.03) and Pois(5) with the same weights. How much power does the Welch test have?
- Does our power improve when we trim the top 2%? Does the p-value sampling distribution stay uniform? Research and explain.
'''

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.power import tt_ind_solve_power
from scipy import stats

def simulate_mixture_1(size):
    xs = np.random.poisson(1, size=size)
    ys = np.random.poisson(5, size=size)
    us = np.random.uniform(size=size)
    return np.where(us < 0.99, xs, ys)

def simulate_mixture_2(size):
    xs = np.random.poisson(1.03, size=size)
    ys = np.random.poisson(5, size=size)
    us = np.random.uniform(size=size)
    return np.where(us < 0.99, xs, ys)

def split_data(data):
    a_test = np.random.choice(data, size=int(len(data/2)))
    b_test = np.random.choice(data, size=int(len(data/2)))
    return a_test, b_test

def pooled_std(sample_a, sample_b):
    n1 = len(sample_a)
    n2 = len(sample_b)
    pooled_var = (((n1-1) * np.var(sample_a)**2) +((n2-1) * np.var(sample_b)**2))/(n1+n2-2)
    return np.sqrt(pooled_var)

def effect_size(sample_a, sample_b):
    return np.abs(np.mean(sample_a) - np.mean(sample_b)) / pooled_std(sample_a, sample_b)

def trim_data(data, portion):
    sorted_data = np.sort(data)
    return sorted_data[:int(len(data)*(1-portion))]


num_of_simulations = int(1e4)
pval_booking = []
pval_booker = []
pval_booking_welch = []
pval_booking_welch_trimmed = []

for simulation in range(num_of_simulations):
    # (a, b, c)
    num_of_users = int(1e4)
    simulated_users = simulate_mixture_1(size=num_of_users)
    a_test, b_test = split_data(simulated_users)
    p_value_booking = stats.nbinom.cdf(k=sum(b_test), n=sum(a_test), p=0.5)
    p_value_booker = stats.nbinom.cdf(k=sum(b_test > 0), n=sum(a_test > 0), p=0.5)
    p_value_welch = stats.ttest_ind(a_test, b_test)[1]
    
    # (e)
    a_test_trimmed = trim_data(a_test, 0.02)
    b_test_trimmed = trim_data(b_test, 0.02)
    p_value_welch_trimmed = stats.ttest_ind(a_test_trimmed, b_test_trimmed)[1]

    pval_booking.append(p_value_booking)
    pval_booker.append(p_value_booker)
    pval_booking_welch.append(p_value_welch)
    pval_booking_welch_trimmed.append(p_value_welch_trimmed)

plt.figure()
plt.hist(pval_booking, bins=50, label='booking rate p-value')
plt.legend()
plt.savefig('2_1.png')

plt.figure()
plt.hist(pval_booker, bins=50, label='bookers rate p-value')
plt.legend()
plt.savefig('2_2.png')

plt.figure()
plt.hist(pval_booking_welch, bins=50, label='Welch test p-value')
plt.legend()
plt.savefig('2_3.png')

plt.figure()
plt.hist(pval_booking_welch_trimmed, bins=50, label='Welch test p-value (trimmed)')
plt.legend()
plt.savefig('2_5.png')


# (d, e)
np.random.seed(200)
sample_a = simulate_mixture_1(size=int(5e4))
sample_b = simulate_mixture_2(size=int(5e4))
power = tt_ind_solve_power(
        effect_size=effect_size(sample_a, sample_b), 
        nobs1=len(sample_a), alpha=0.05, alternative='two-sided'
) 
print("Power =", power*100, "%")

trimmed_sample_a = trim_data(sample_a, 0.02)
trimmed_sample_b = trim_data(sample_b, 0.02)
power_trimmed = tt_ind_solve_power(
        effect_size=effect_size(trimmed_sample_a, trimmed_sample_b),
        nobs1=len(sample_a), alpha=0.05, alternative='two-sided'
)
print("Power after trimming =", power_trimmed*100, "%")

# By trimming the top 2% of the data, power is improved and p-value sampling distribution becomes positively skewed (previously, uniformly distributed)
