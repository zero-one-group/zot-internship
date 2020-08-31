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
sample_norm = normal_dist(mean, sd, 100)
plt.hist(sample_norm, bins=10, density=True)
plt.savefig('test2.png')

sample_norm

print("sample mean is %s" % np.mean(sample_norm))
print("sample standard deviation (measures variability of individual data to the mean) %s" 
      % np.std(sample_norm))
print("standard error of sample mean (how far the sample mean is to the true population mean): %s" 
      % (np.std(sample_norm) / np.sqrt(len(sample_norm))))
print("sample kurtosis is %s" % stats.kurtosis(sample_norm))
print("sample skewness is %s" % stats.skew(sample_norm))
