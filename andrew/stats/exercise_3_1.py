'''
For the normal-Cauchy Bayes estimator solve the following questions when x = 0, 2, 4.
a. Plot the integrands, and use Monte Carlo integration based on a Cauchy
simulation to calculate the integrals.
b. Monitor the convergence with the standard error of the estimate. Obtain three
digits of accuracy with probability .95.
c. Repeat the experiment with a Monte Carlo integration based on a normal
simulation and compare both approaches.
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def bayes_integrand_1(x, theta):
    return (theta / (1 + theta**2)) * np.exp(-0.5 * (x - theta)**2)

def bayes_integrand_2(x, theta):
    return (1 / (1 + theta**2)) * np.exp(-0.5 * (x - theta)**2)

def monte_carlo_integration(upper_limit, lower_limit, number_of_iteration, integrand_value):
    return ((upper_limit - lower_limit)/number_of_iteration) * sum(integrand_value)

def mean_confidence_interval(data, confidence):
    data_array = 1.0 * np.array(data)
    num = len(data_array)
    mean, standard_err = np.mean(data_array), stats.sem(data_array)
    space = standard_err * stats.t.ppf((1 + confidence) / 2., num - 1)
    return mean, mean-space, mean+space

def random_numbers_normal(lower_limit, upper_limit, number_of_iteration):
    scaling = upper_limit - lower_limit
    return (np.random.rand(number_of_iteration) * scaling) - (scaling/2)

def random_numbers_uniform(lower_limit, upper_limit, number_of_iteration):
    return np.random.uniform(lower_limit, upper_limit, number_of_iteration)

def random_numbers_cauchy(lower_limit, upper_limit, number_of_iteration):
    mid_point = (lower_limit + upper_limit)/2
    scaling = upper_limit - lower_limit
    return stats.cauchy.rvs(loc=mid_point, scale=scaling, size=number_of_iteration)


# Plotting integrands
theta = np.linspace(-5, 10, 10000)
for param in [0, 2, 4]:
    integrand_1 = bayes_integrand_1(param, theta)
    integrand_2 = bayes_integrand_2(param, theta)
    plt.figure()
    plt.plot(theta, integrand_1, 'r.', label='Integrand 1 (numerator)')
    plt.plot(theta, integrand_2, 'b.', label='Integrand 2 (denominator)')
    plt.legend()
    plt.savefig('integrands x = ' + str(param) + '.png')


# Monte-Carlo Integration
number_of_iteration = 5000
lower_limit = -5
upper_limit = 5
param = 0

integral_1_array = []
integral_2_array = []
for iteration in range(number_of_iteration):
    random_numbers = random_numbers_cauchy(lower_limit, upper_limit, number_of_iteration)
    integrand_1_value = [
            bayes_integrand_1(param, random_numbers[i]) 
            for i in range(len(random_numbers))
            ]
    integrand_2_value = [
            bayes_integrand_2(param, random_numbers[i]) 
            for i in range(len(random_numbers))
            ]
    integral_1 = monte_carlo_integration(upper_limit, lower_limit, number_of_iteration, integrand_1_value)
    integral_2 = monte_carlo_integration(upper_limit, lower_limit, number_of_iteration, integrand_2_value)
    integral_1_array.append(integral_1)
    integral_2_array.append(integral_2)

answer = np.array(integral_1_array) / np.array(integral_2_array)
plt.figure()
plt.hist(answer, bins=50)
plt.savefig('integral x = ' + str(param) + '.png')

# Check number of digits of accuracy with probability 0.95
mean, lower_bound, upper_bound = mean_confidence_interval(answer, 0.95)
print("Mean =", mean)
print("Lower bound =", lower_bound)
print("Upper bound =", upper_bound)
print("Difference =", upper_bound - lower_bound)
