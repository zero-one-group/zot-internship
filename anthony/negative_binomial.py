"""
Suppose the number of bookings of a user is i.i.d Poisson mixture of Pois(1) and Pois(5) with weights 99% and 1% respectively. Think of the mixture distribution as a combination of regular travellers and travel agents. Moreover, we only observe a user only if they make at least one booking. Think of this as a mixture of legitimate users that may book and illegitimate bots that would never book. The legitimate non-bookers are indistinguishable from the illegitimate bots.
"""


"""
Part 1
Consider the negative binomial test (part 2). Assume there are no booking-rate differences in A and B (i.e. assuming the null hypothesis). Simulate A/B tests with 10k users, and show that applying the negative binomial test on bookings yields a U-shaped p-value sampling distribution.

Null hypothesis => CR_A = CR_B => p = 0.5.
"""

import numpy as np
from scipy import stats

def simulate_bookings(rate_1, rate_2, proportion, n_users=1e4):
    n_users = int(n_users)
    bookings_1 = np.random.poisson(rate_1, size=n_users)
    bookings_2 = np.random.poisson(rate_2, size=n_users)
    uniforms = np.random.uniform(size=n_users)
    return np.where(uniforms < proportion, bookings_1, bookings_2)

def simulate_bookings_nb_test():
    n_users = int(1e4)
    bookings = simulate_bookings(1, 5, 0.99, n_users=n_users)
    variants = np.random.choice(['a', 'b'], size=n_users)
    bookings_a = np.sum(bookings * (variants == 'a'))
    bookings_b = np.sum(bookings * (variants == 'b'))
    return stats.nbinom.cdf(bookings_b, n=bookings_a, p=0.5)

simulated_p_values = [simulate_bookings_nb_test() for _ in range(2000)]
np.histogram(np.array(simulated_p_values) * 100) # => u-shaped p-values


"""
Show that applying the same test on bookers yields a uniform p-value sampling distribution. Why?
"""

def simulate_bookers_nb_test():
    n_users = int(1e4)
    bookings = simulate_bookings(1, 5, 0.99, n_users=n_users)
    variants = np.random.choice(['a', 'b'], size=n_users)
    bookers_a = np.sum((bookings > 0) & (variants == 'a'))
    bookers_b = np.sum((bookings > 0) * (variants == 'b'))
    return stats.nbinom.cdf(bookers_b, n=bookers_a, p=0.5)

simulated_p_values = [simulate_bookers_nb_test() for _ in range(2000)]
np.histogram(np.array(simulated_p_values) * 100) # => uniform p-values


"""
Now assume that you can observe the non-bookers as well. Design the Welch test that takes into account the number of bookings, and show that your experiment design yields a uniform p-value sampling distribution.
"""

def welch_t_stat(xs_a, xs_b):
    return (
        (np.mean(xs_b) - np.mean(xs_a))  /
        np.sqrt(
            (np.var(xs_a) / len(xs_a)) +
            (np.var(xs_b) / len(xs_b))
        )
    )

def welch_test():
    n_users = int(1e4)
    bookings = simulate_bookings(1, 5, 0.99, n_users=n_users)
    variants = np.random.choice(['a', 'b'], size=n_users)
    bookings_a = bookings[variants == 'a']
    bookings_b = bookings[variants == 'b']
    t_stats = welch_t_stat(bookings_a, bookings_b)
    return stats.t.cdf(t_stats, df=n_users - 2)

simulated_p_values = [welch_test() for _ in range(2000)]
np.histogram(np.array(simulated_p_values) * 100) # => uniform p-values

"""
Suppose that for the users that received the B variant, their Poisson mixture is Pois(1.03) and Pois(5) with the same weights. How much power does the Welch test have?
"""

def welch_test():
    n_users = int(2e4)
    counterfactual_bookings_a = simulate_bookings(1, 5, 0.99, n_users=n_users)
    counterfactual_bookings_b = simulate_bookings(1.03, 5, 0.99, n_users=n_users)
    variants = np.random.choice(['a', 'b'], size=n_users)
    bookings_a = counterfactual_bookings_a[variants == 'a']
    bookings_b = counterfactual_bookings_b[variants == 'b']
    t_stats = welch_t_stat(bookings_a, bookings_b)
    return stats.t.cdf(t_stats, df=n_users - 2)

simulated_p_values = [welch_test() for _ in range(2000)]
alpha = 0.05
power = np.mean([p_value > (1 - alpha) for p_value in simulated_p_values])
# power = 0.4 => 60% of the time, you're missing out.
# doubling the n_users => power = 0.6, but expensive.

"""
Does our power improve when we trim the top 2%? Does the p-value sampling distribution stay uniform? Research and explain.
"""

def trim(bookings, perc):
    cutoff = np.percentile(bookings, perc)
    return bookings[bookings < cutoff]

def welch_test():
    n_users = int(2e4)
    counterfactual_bookings_a = simulate_bookings(1, 5, 0.99, n_users=n_users)
    counterfactual_bookings_b = simulate_bookings(1.03, 5, 0.99, n_users=n_users)
    variants = np.random.choice(['a', 'b'], size=n_users)
    bookings_a = trim(counterfactual_bookings_a[variants == 'a'], 98)
    bookings_b = trim(counterfactual_bookings_b[variants == 'b'], 98)
    t_stats = welch_t_stat(bookings_a, bookings_b)
    return stats.t.cdf(t_stats, df=n_users - 2)

simulated_p_values = [welch_test() for _ in range(2000)]
alpha = 0.05
power = np.mean([p_value > (1 - alpha) for p_value in simulated_p_values])

# power = 60% => yes it increases.
