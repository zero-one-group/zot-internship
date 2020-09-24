"""Create a power-calculator application in Python.
Given the traffic, variance of the KPI variable, significance level and power
level, calculate a table of MDE and expected duration of runs.
"""
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

def min_sample(baseline_cr, mde, power, sig_level):
    standard_norm = stats.norm(0, 1)
    Z_beta = standard_norm.ppf(power)
    Z_alpha = standard_norm.ppf(1-sig_level/2)
    pooled_prob = (baseline_cr + baseline_cr + mde) / 2
    min_size = (2 * pooled_prob * (1 - pooled_prob) * 
                (Z_beta + Z_alpha)**2) / mde**2
    return min_size


def duration(min_sample):
    return round(min_sample * variant / traffic, ndigits=2)


#example
traffic = 1000 #per run
sig_level = 0.05
power = 0.8
variant = 2
baseline_rate = 0.3

#make a list of MDEs
mde = np.linspace(0.01, 0.50, num=50)
samples = [min_sample(baseline_rate, mde[i], power, sig_level) for i in
           range(len(mde))]
runs = list(duration(samples[i]) for i in range(len(samples)))

for x, y in zip(mde2, runs):
    print("MDE: ", x, "runs: ", y)
