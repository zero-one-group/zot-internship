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

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def function(x):
    return np.exp(-(x-3)**2/2) + np.exp(-(x-6)**2/2)


number_of_iteration = int(1e5)

#(b) Monte carlo approximation
xs = np.random.randn(number_of_iteration)
print(np.mean(function(xs)))

#(c) Importance Sampling
xs = np.random.uniform(-8, -1, size=number_of_iteration)
weights = stats.norm.pdf(xs) / stats.uniform.pdf(xs, loc=-8, scale=7)
estimate = np.mean(weights * function(xs))
print(estimate)


