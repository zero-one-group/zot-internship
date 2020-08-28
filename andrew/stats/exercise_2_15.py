'''
The Poisson distribution P(λ) is connected to the exponential distribution through the Poisson process in that it can be simulated by generating exponential random variables until their sum exceeds 1. That is, if Xi ∼ Exp(λ) and if K is the first value for which sum of Xi > 1 for i = 1 to K+1, then K ∼ P(λ). Compare this algorithm with rpois and the algorithm of Example 2.5 for both small and large values of λ.
'''


import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def poisson_simulation(exponential_dist):
    total = 0
    count = 0
    k_poisson = []
    for idx in range(len(exponential_dist)):
        if total < 1:
            total += exponential_dist[idx]
            count += 1
        else:
            total = 0
            count = 0
            total += exponential_dist[idx]
            count += 1
        k_poisson.append(count - 1)

    simulated_poisson = [
        k_poisson[idx] for idx in range(len(k_poisson) - 1) 
        if k_poisson[idx] >= k_poisson[idx+1]
    ]
    return simulated_poisson


scale = 10 #lambda value
num_of_samples = int(1e6)

exponential_dist = stats.expon.rvs(1/scale, size=num_of_samples)
poisson_dist = stats.poisson.rvs(scale, size=num_of_samples)

plt.figure(0)
plt.hist(poisson_dist, bins=100, alpha=0.5, density=True, label='Poisson distribution')
plt.hist(poisson_simulation(exponential_dist), bins=100, alpha=0.5, density=True, label='Simulation')
plt.legend(loc='upper right')
plt.savefig('Poisson_process.png')

