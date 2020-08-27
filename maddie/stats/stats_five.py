#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

n_points = 100
n_sims = 1000

def simulate_one_realisation():
    return np.random.uniform(0, 5, size=n_points)

def infer(sample):
    return {'mean': np.mean(sample), 'std': np.std(sample)}

inference = [infer(simulate_one_realisation()) for _ in range(n_sims)]

means = np.percentile([x['mean'] for x in inference], [25, 50, 75])

sample_distribution_estimator = np.percentile([x['std'] for x in inference], [25, 50, 75])
print(sample_distribution_estimator)

plt.hist([x['std'] for x in inference], bins=25)
plt.show()

# Increasing the sample size shows convergence and thus consistency in the sample distribution of the RMSE of the estimator
# Increasing the sample size shows that there is bias because the predicted differs from the true value




