'''
Suppose the number of bookings of a user is i.i.d Poisson mixture of Pois(1) and Pois(5) with weights 99% and 1% respectively. Think of the mixture distribution as a combination of regular travellers and travel agents. Moreover, we only observe a user only if they make at least one booking. Think of this as a mixture of legitimate users that may book and illegitimate bots that would never book. The legitimate non-bookers are indistinguishable from the illegitimate bots.
■   Consider the negative binomial test (part 2). Assume there are no booking-rate differences in A and B (i.e. assuming the null hypothesis). Simulate A/B tests with 10k users, and show that applying the negative binomial test on bookings yields an inverted-U p-value sampling distribution.
■   Show that applying the same test on bookers yields a uniform p-value sampling distribution. Why?
■   Now assume that you can observe the non-bookers as well. Design the Welch test that takes into account the number of bookings, and show tha your experiment design yields a uniform p-value sampling distribution.
■   Suppose that for the users that received the B variant, their Poisson mixture is Pois(1.03) and Pois(5) with the same weights. How much power does the Welch test have?
■   Does our power improve when we trim the top 2%? Does the p-value sampling distribution stay uniform? Research and explain.
t
'''
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def simulate_mixture(size):
    xs = np.random.poisson(1, size=size)
    ys = np.random.poisson(5, size=size)
    us = np.random.uniform(size=size)
    return np.where(us < 0.99, xs, ys)

def ab_test_bookings(data):
    a_test = sum(data[0:int(len(data)/2)])
    b_test = sum(data[int(len(data)/2)::])
    conversion_rate_a = (a_test / (a_test+b_test))
    conversion_rate_b = (b_test / (a_test+b_test))
    return conversion_rate_a, conversion_rate_b


num_of_simulations = 1000
conversion_rate_a = []
conversion_rate_b = []
for simulation in range(num_of_simulations):
    num_of_users = int(1e4)
    simulated_users = simulate_mixture(size=num_of_users)
    result_a, result_b = ab_test_bookings(simulated_users)
    conversion_rate_a.append(result_a)
    conversion_rate_b.append(result_b)

plt.figure()
plt.hist(conversion_rate_a, bins=50, label='A test')
plt.hist(conversion_rate_b, bins=50, label='B test')
plt.legend()
plt.savefig('2_1.png')


num_of_simulations = 1000
p_value = []
for simulation in range(num_of_simulations):
    num_of_users = int(1e4)
    simulated_users = simulate_mixture(size=num_of_users)
    p_value.append(sum(simulated_users > 0) / len(simulated_users))

plt.figure()
plt.hist(p_value, bins=50, label='Bookers')
plt.legend()
plt.savefig('2_2.png')







