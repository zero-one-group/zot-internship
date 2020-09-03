"""Consider 100 i.i.d draws from the normal distribution N(1, 2). Plot the
histogram to approximate the sampling distributions of the sample mean, sample
standard deviation, sample skewness and sample kurtosis. In your own words,
describe the difference between the sample standard deviation and the estimated
standard error for the sample mean. Contrast their histograms"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def normal_dist(miu, sigma, draws):
    return np.random.normal(miu, sigma, size=draws)


mean, sd = 1, 2
nsim = 1000
sample_mean = [np.mean((normal_dist(mean, sd, 100))) for _ in range(nsim)]
sample_sd = [np.std((normal_dist(mean, sd, 100))) for _ in range(nsim)]
sample_se = [np.std((normal_dist(mean, sd, 100)))/10 for _ in range(nsim)]
sample_skewness = [stats.skew((normal_dist(mean, sd, 100))) for _ in range(nsim)]
sample_kurtosis = [stats.kurtosis((normal_dist(mean, sd, 100))) for _ in
                   range(nsim)]

plt.hist(sample_se, bins=100, label='sample std err')
plt.hist(sample_mean, bins=100, label='sample mean')
plt.hist(sample_sd, bins=100, label='sample std dev')
plt.legend()
plt.savefig('ex7_mean_se.png')

plt.hist(sample_skewness, bins=100, label='skewness')
plt.hist(sample_kurtosis, bins=100, label='kurtosis')
plt.legend()
plt.savefig('ex7_kurtosis_skew.png')


"""standard error of sample mean (how far the sample mean is to the true
population mean), while std dev measures the variability of the data to the
mean"""
