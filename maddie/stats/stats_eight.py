#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

data = np.fromfile('nerve.dat', dtype=float) 

# Empirical / Basic Bootstrap

n_points = 100

def take_sample():
    return np.random.choice(data, size=n_points, replace=False)

def generate_mean(sample, axis=None):
    return np.mean(sample, axis=axis)

def generate_median(sample, axis=None):
    return np.median(sample, axis=axis)

bootstrap_reps = 1000

def resample(sample, reps):
    n = len(sample)
    return np.random.choice(sample, size=reps * n).reshape((reps, n))

def bootstrap_stats(sample, reps=bootstrap_reps, stat=generate_mean):
    resamples = resample(sample, reps)
    return stat(resamples, axis=1)

np.random.seed(0)

sample = take_sample()

plt.hist(bootstrap_stats(sample), bins=25, density=True)
plt.show()

# Percentile Bootstrap

def percentile_ci_mean(sample, reps=bootstrap_reps, stat=generate_mean):
    stats = bootstrap_stats(sample, reps, stat)
    return np.percentile(stats, [2.5, 97.5])

np.random.seed(0)
sample = take_sample()
print(percentile_ci_mean(sample))

def percentile_ci_median(sample, reps=bootstrap_reps, stat=generate_median):
    stats = bootstrap_stats(sample, reps, stat)
    return np.percentile(stats, [2.5, 97.5])

np.random.seed(0)
sample_median = take_sample()
print(percentile_ci_median(sample_median))

# Comparing the values of the mean and median calculated above, the mean is greater than the median suggesting the distribution is positively skewed.

# Bootstrap-t / Studentized Test

def studentized_stats(sample, reps=bootstrap_reps, stat=generate_mean):
    resamples = resample(sample, reps)
    resample_stats = stat(resamples, axis=1)
    resample_sd = np.std(resample_stats)

    resample_std_errs = np.std(resamples, axis=1) / np.sqrt(len(sample))

    sample_stat = stat(sample)
    t_statistics = (resample_stats - sample_stat) / resample_std_errs

def studentized_ci(sample, reps=bootstrap_reps, stat=generate_mean):
    t_statistics, resample_sd = studentized_stats(sample, reps, stat)
    lower, upper = np.percentile(t_statistics, [2.5, 97.5])

    sample_stat = stat(sample)
    return (sample_stat - resample_sd * upper,
            sample_stat - resample_sd * lower)

np.random.seed(0)
sample = take_sample()
print(studentized_ci(sample))


