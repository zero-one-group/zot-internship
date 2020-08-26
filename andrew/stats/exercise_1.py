'''
Consider a normal distribution N(1, 2). Plot the histogram to approximate the sampling distributions of the sample mean, sample standard deviation, sample skewness and sample kurtosis. In your own words, describe the difference between the sample standard deviation and the estimated standard error for the sample mean. Contrast their histograms.
'''
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def simulate_normals():
    return 1 + np.sqrt(2)*np.random.randn(int(1e3))

sample_means = [np.mean(simulate_normals()) for _ in range(int(1e3))]
sample_stds = [np.std(simulate_normals()) for _ in range(int(1e3))]
sample_skews = [stats.skew(simulate_normals()) for _ in range(int(1e3))]
sample_kurtosis = [stats.kurtosis(simulate_normals()) for _ in range(int(1e3))]

plt.figure()
plt.hist(sample_means, bins=50, alpha=0.5, density=True, label='Sample mean')
plt.hist(sample_stds, bins=50, alpha=0.5, density=True, label='Sample standard deviation')
plt.hist(sample_skews, bins=50, alpha=0.5, density=True, label='Sample skewness')
plt.hist(sample_kurtosis, bins=50, alpha=0.5, density=True, label='Sample kurtosis')
plt.legend()
plt.savefig('normal_samples.png')


# Standard deviation is a measure of volatility, while SEM measures how far the sample mean of the data is from the real mean (population mean). Note that SEM is standard deviation divided by sqrt of number of data. So, it decreases as number of data increases.
