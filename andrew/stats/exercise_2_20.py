'''
In each of the following cases, construct an Accept–Reject algorithm,
generate a sample of the corresponding random variables, and draw the density function
on top of the histogram.
a. Generate normal random variables using a Cauchy candidate in Accept–Reject.
b. Generate gamma G(4.3, 6.2) random variables using a gamma G(4, 7) candidate.
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def simulate_normal_with_cauchy():
    grid = np.linspace(-5, 5, 1000)
    candidate = np.random.standard_cauchy()
    g = stats.cauchy.pdf
    f = stats.norm.pdf
    M = np.max(f(grid) / g(grid))
    accept = np.random.random() < (f(candidate) / (M * g(candidate)))
    if accept:
        return candidate
    else:
        return simulate_normal_with_cauchy()

def simulate_gamma_with_gamma():
    grid = np.linspace(-5, 5, 1000)
    candidate = np.random.gamma(7/4, 7)
    g = stats.cauchy.pdf
    f = stats.cauchy.pdf
    M = np.max(f(grid) / g(grid))
    accept = np.random.gamma(6.2/4.3, 4.3) < (f(candidate) / (M * g(candidate)))
    if accept:
        return candidate
    else:
        return simulate_gamma_with_gamma()


simulated_normal = [simulate_normal_with_cauchy() for _ in range(int(1e4))]
plot_grid = np.linspace(np.min(simulated_normal), np.max(simulated_normal), 100)
plt.figure(0)
plt.hist(simulated_normal, bins=80, density=True, label='Simulated normal')
plt.plot(plot_grid, stats.norm.pdf(plot_grid), label='Normal distribution')
plt.legend()
plt.savefig('cauchy_candidate.png')

simulated_gamma = [simulate_gamma_with_gamma() for _ in range(int(1e4))]
plot_grid = np.linspace(np.min(simulated_gamma), np.max(simulated_gamma), 100)
plt.figure(1)
plt.hist(simulated_gamma, bins=80, density=True, label='Simulated Gamma G(4, 7)')
plt.plot(plot_grid, stats.gamma.pdf(plot_grid, loc=6.2/4.3, a=6.2/4.3, scale=6.2), label='Gamma distribution G(4, 7)')
plt.show()
plt.legend()
plt.savefig('gamma_candidate.png')
