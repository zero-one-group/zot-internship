#!/usr/bin/env python

import numpy as np

n_points = 100
n_sims = 1000

# Log-Normal Expectations
xs = np.random.randn(n_sims)
np.exp(np.mean(xs))

xs = np.random.randn(n_sims)
np.mean(np.exp(xs))


