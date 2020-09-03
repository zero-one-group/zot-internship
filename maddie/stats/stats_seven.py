#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

n_points = 100
n_sims = 1000

def simulate_one_realisation():
    return np.random.normal(1, 2, size=n_points)

def infer(sample):
    return {'mean': np.mean(sample), 'std': np.std(sample)}

inference = [infer(simulate_one_realisation()) for _ in range(n_sims)]

means = np.percentile([x['mean'] for x in inference], [25, 50, 75])
print(means)

plt.hist([x['mean'] for x in inference], bins=25)
plt.show()

standard_error = np.percentile([x['std'] for x in inference], [25, 50, 75])
print(standard_error)

plt.hist([x['std'] for x in inference], bins=25)
plt.show()

# The sample standard deviation and estimated standard error of the sample mean both have a slight right skew. But the skewness of the sample standard deviation is much more than the sample mean.
# The sample standar deviation histogram has many modes signifying a random distributionwhereas the sample mean histogram has a more uniform distribution shape with only one mode. 
