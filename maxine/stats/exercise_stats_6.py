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


nsims = 1000
ninety_five_conf_interval = list(confidence_interval_mean(beta_dist(1, 2, 100), 0.05) for _ in range(nsims)) 
true_mean = 1/3
interval_with_true_mean = [ninety_five_conf_interval[index] for index in range(len(ninety_five_conf_interval)) if
     ninety_five_conf_interval[index][0] < true_mean <
     ninety_five_conf_interval[index][1]]
proportion = len(interval_with_true_mean) / len(ninety_five_conf_interval) * 100
print("proportion of interval with true mean: %d percent" % (round(proportion, 3)))
print("90% CI for the probability that 95% CI contains true mean: ", confidence_interval_mean(interval_with_true_mean, 0.1))


