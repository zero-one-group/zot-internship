"""Use the bootstrap to get confidence intervals for the median and skewness of
these data. In particular, present basic bootstrap, bootstrap-t and percentile
confidence intervals."""

import numpy as np
import scipy.stats as stats


def open_file(filename):
    name_file = open(filename, "r")
    file_array = np.array(name_file.read().split(), dtype=float)
    return file_array


def bootstrap(data, simulation):
    return np.random.choice(data, size=simulation, replace=True)


def percentile_interval(data, parameter, nsim, alpha_ci):
    bs = list(parameter(bootstrap(data, nsim)) for _ in range(nsim))
    lower_interval = np.percentile(bs, (alpha_ci) / 2)
    upper_interval = np.percentile(bs, 1 - (alpha_ci) / 2)
    return lower_interval, upper_interval


def bootstrap_ci(data, parameter, nsim, alpha_ci):
    theta = parameter(data)
    bs = list(parameter(bootstrap(data, nsim)) for _ in range(nsim))
    upper_interval = (2*theta) - np.percentile(bs, (alpha_ci) / 2)
    lower_interval = (2*theta) - np.percentile(bs, 1 - (alpha_ci) / 2)
    return lower_interval, upper_interval


def bootstrap_t_ci(data, parameter, nsim, alpha_ci):
    theta = parameter(data)
    data_se = stats.sem(data)
    bs_t = list(((parameter(bootstrap(data_nerve, nsim)) - theta) /
                 stats.sem(bootstrap(data_nerve, nsim))) for _ in range(nsim))
    upper_interval = theta - (np.percentile(bs_t, alpha_ci) * data_se)
    lower_interval = theta - (np.percentile(bs_t, (1 - alpha_ci)) * data_se)
    return lower_interval, upper_interval


sim = 1000
data_nerve = open_file("nerve.dat.txt")
print("90% CI of skewness with percentile", percentile_interval(data_nerve,
                                                                np.median,
                                                                sim, 0.1))
print("90% CI of skewness with percentile", percentile_interval(data_nerve,
                                                                stats.skew,
                                                                sim, 0.1))
print("90% CI of median with basic bootstrap", bootstrap_ci(data_nerve,
                                                            np.median, sim, 0.1))
print("90% CI of skewness with basic bootstrap",
      bootstrap_ci(data_nerve, stats.skew, sim, 0.1))
print("90% CI of median with bootstrap-t", bootstrap_t_ci(data_nerve,
                                                          np.median, sim, 0.1))
print("90% CI of skewness with bootstrap-t", bootstrap_t_ci(data_nerve,
                                                            stats.skew, sim, 0.1))
