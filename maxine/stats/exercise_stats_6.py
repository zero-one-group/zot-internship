"""we have 100 i.i.d draws of Beta(1, 2) distribution. Simulate the events 1000
times, and each time saving the 95% confidence interval of the mean
How many of your intervals contain the true mean?
Construct the 90% confidence interval for the probability that the 95%
confidence interval contains the true mean
Explain, in your own words, what does 95% mean in the 95% confidence interval?"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def beta_dist(alpha, beta, draws):
    return np.random.beta(alpha, beta, size=draws)

def confidence_interval_mean(dist, alpha_ci):
    return stats.mstats.trimmed_mean_ci(dist, alpha=alpha_ci)
    #mean, var, std = stats.bayes_mvs(dist, alpha=ci)

plt.plot(beta_dist(1, 2, 100))
plt.savefig("beta.png")

list(confidence_interval_mean(beta_dist(1, 2, 100), 0.05) for _ in range(1000)) 

