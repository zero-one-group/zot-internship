'''
Consider a normal distribution N(1, 2). Plot the histogram to approximate the sampling distributions of the sample mean, sample standard deviation, sample skewness and sample kurtosis. In your own words, describe the difference between the sample standard deviation and the estimated standard error for the sample mean. Contrast their histograms.
'''

from scipy import stats
import matplotlib.pyplot as plt
import math
import numpy as np

normal_random_variates = stats.norm.rvs(1, math.sqrt(2), size = 100000)
plt.figure(0)
plt.hist(normal_random_variates, bins = 100)
plt.savefig('1.png')

mean = np.mean(normal_random_variates)
skew = stats.skew(normal_random_variates)
std = stats.tstd(normal_random_variates)
std_err = stats.sem(normal_random_variates)
kurtosis = stats.kurtosis(normal_random_variates)
print("Mean =", mean)
print("Standard deviation =", std)
print("SE of mean =", std_err)
print("Skewness =", skew)
print("Kurtosis =", kurtosis)

# Standard deviation is a measure of volatility, while SEM measures how far the sample mean of the data is from the real mean (population mean). Note that SEM is standard deviation divided by sqrt of number of data. So, it decreases as number of data increases.
