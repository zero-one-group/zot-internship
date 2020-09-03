'''
Suppose the number of bookings of a user is i.i.d Poisson mixture of Pois(1) and Pois(5) with weights 99% and 1% respectively. Think of the mixture distribution as a combination of regular travellers and travel agents. Moreover, we only observe a user only if they make at least one booking.
■   Consider the negative binomial test (part 2). Simulate A/B tests with 500k users, and show that applying the negative binomial test on bookings yields an inverted-U p-value sampling distribution.
■   Show that applying the same test on bookers yields a uniform p-value sampling distribution. Why?
■   Design the Welch test that takes into account the number of bookings, and show that your experiment design yields a uniform p-value sampling distribution.
■   Suppose that for the users that received the B variant, their Poisson mixture is Pois(1.25) and Pois(5) with the same weights. How much power does the Welch test have?
■   Does our power improve when we trim the top 2%? Does the p-value sampling distribution stay uniform? Research and explain.
'''

import numpy as np
import random
from scipy import stats

def simulated_users(num_of_users):
    poisson_dist_1 = np.random.poisson(1, size=int(0.99*num_of_users)) + 1
    poisson_dist_2 = np.random.poisson(5, size=int(0.01*num_of_users)) + 1
    simulated = np.concatenate((poisson_dist_1, poisson_dist_2))
    random.shuffle(simulated)
    a_test = sum(simulated[0:int(num_of_users/2)])
    b_test = sum(simulated[int(num_of_users/2)::])
    conversion_rate_1 = a_test/(a_test + b_test)
    conversion_rate_2 = b_test/(a_test + b_test)
    return conversion_rate_1, conversion_rate_2


num_of_users = int(5e5)
conversion_rate_1, conversion_rate_2 = simulated_users(num_of_users)
prob_1 = conversion_rate_1 / (conversion_rate_1 + conversion_rate_2)
upper_boundary = stats.nbinom(num_of_users/2, prob_1)
lower_boundary = stats.nbinom(num_of_users/2, prob_1)
print(upper_boundary, lower_boundary)
