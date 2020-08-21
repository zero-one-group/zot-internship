'''
For the computation of the expectation Ef [h(X)] when f is the
normal pdf and h(x) = exp(−(x − 3)2/2) + exp(−(x − 6)2/2):
a. Show that Ef [h(X)] can be computed in closed form and derive its value.
b. Construct a regular Monte Carlo approximation based on a normal N (0, 1) sample of size Nsim=10^3 and produce an error evaluation.
c. Compare the above with an importance sampling approximation based on
an importance function g corresponding to the U(−8, −1) distribution and
a sample of size Nsim=10^3. (Warning: This choice of g does not provide a
converging approximation of Ef [h(X)]!)
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def function(x):
    return x*((np.exp((-(x-3)**2)/2)) + (np.exp((-(x-6)**2)/2)))

def monte_carlo_integration(upper_limit, lower_limit, number_of_iteration, integrand_value):
    return ((upper_limit - lower_limit)/number_of_iteration) * sum(integrand_value)

def random_numbers_normal(lower_limit, upper_limit, number_of_iteration):
    scaling = upper_limit - lower_limit
    return (np.random.rand(number_of_iteration) * scaling) - (scaling/2)


#(a). E(h(X)) = 9 * sqrt(2*pi) ~ 22.559654..

# Monte-Carlo Integration
number_of_iteration = 1000
lower_limit = -10
upper_limit = 10

random_numbers = random_numbers_normal(lower_limit, upper_limit, number_of_iteration) 
integrand_value = [function(random_numbers[i]) for i in range(len(random_numbers))]
integral = monte_carlo_integration(upper_limit, lower_limit, number_of_iteration, integrand_value)
print("Analytical value =", 9*np.sqrt(2*np.pi))
print("Numerical value =", integral)
print("Error =", (9*np.sqrt(2*np.pi)) - integral)


# Importance Sampling
p_x = stats.norm(0, 1)
q_x = stats.uniform(10)

value_list = []
for i in range(number_of_iteration):
    # sample from different distribution
    x_i = np.random.uniform(-8, -1)
    value = function(x_i) * (p_x.pdf(x_i) / q_x.pdf(x_i))
    value_list.append(value)

