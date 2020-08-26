'''
Suppose we have 100 i.i.d. draws from U[0, θ]. Consider the maximum of the 100 draws as an estimator for θ. Approximate the estimator’s sampling distribution by simulating the events 1000 times. What is the sampling distribution of the estimated RMSE of the estimator? Is it biased? Is it consistent?
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def RMSE(predicted_array, actual):
    actual_array = [actual for idx in range(len(predicted_array))]
    error = np.array(predicted_array) - np.array(actual_array)
    return np.sqrt((error**2) / len(predicted_array))


theta = 5
number_of_simulation = 1000

estimator = [max(np.random.uniform(0, theta, size=100))
        for simulation in range(number_of_simulation)]

plt.figure(0)
plt.hist(estimator, bins=50, label='Estimator')
plt.legend()
plt.savefig('estimator.png')

estimated_RMSE = RMSE(estimator, theta)
plt.figure(1)
plt.hist(estimated_RMSE, bins=50, label='RMSE')
plt.legend()
plt.savefig('rmse.png')

# Check if estimator is biased
bias = np.mean(estimator) - theta
print("Bias =", bias)

# Maximum is a biased estimator, but it is consistent. The definition says that the sampling distribution of an unbiased estimator has expected value equal to θ, the population value.
