'''
Consider the log-normal distribution logN(0, 1), which can be derived by exponentiating the N(0, 1) distribution. Show that exp of the mean does not equal to the mean of the exp. Explain your findings. 
'''

import numpy as np

normal_dist = np.random.normal(0, 1, size = 10000)
log_normal_dist = np.exp(normal_dist)

mean_normal_dist = np.mean(normal_dist)
mean_log_normal_dist = np.mean(log_normal_dist)

print("Exponential of mean of normal distribution =", np.exp(mean_normal_dist))
print("Mean of log-normal distribution =", mean_log_normal_dist)

# Exponential and logs are not linear, while mean calculation is done linearly
