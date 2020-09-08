#!/usr/bin/env python

import numpy as np
from scipy.stats import normaltest
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

n_points = 100

def generate_one_realisation():
    return np.random.beta(1.8, 1, size=n_points)

def infer(sample):
    return np.mean(sample)

data = [infer(generate_one_realisation()) for _ in range(100)]

stat, p = normaltest(data)
print('stat=%.3f, p=%.3f'% (stat, p))
if p > 0.1:
    print ('Null hypothesis accepted')
else:
    print('Null hypothesis rejected')

# The probability that the researcher rejects the null hypothesis (having a p-value less than 0.10) is 90%.

# 10% significance level (alpha) is the probability of rejecting the null hypothesis when it is true (type I error). There is a 10% risk of the researcher concluding that there is a difference when there is no actual significant difference.

data_frame = pd.DataFrame(data)
ax = data_frame.plt.hist()
p = 0.10
ax.axvline(2501, color='r', linewidth=2)
extra = Rectangle((0, 0),  100, 100, fc="w", fill=False, edgecolor='none', linewidth=0)
ax.legend([extra],('p = {}'.format(p), "x"))

# Type I error is the probability of rejecting the null hypothesis when the null hypothesis is actually true.
# Type II error is the acceptance of the null hypothesis when the null hypothesis is actually false.
