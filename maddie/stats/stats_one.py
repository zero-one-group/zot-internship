#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

n_points = 100
n_sims = 1000

# Let X and Y be independent N(0, 1) random variables. Let Z = Xˆ2 + Y. What's the expected sample correlation between X and Y? What about X and Z? Elaborate on your findings.

n_sims = int(1e3)
xs = np.random.randn(n_sims)
ys = np.random.randn(n_sims)
zs = xs*xs + ys
print('Corr(X, Y) =', np.corrcoef(xs, ys)[0,1])
print('Corr(X, Z) =', np.corrcoef(xs, zs)[0,1])

plt.scatter(xs, zs)
plt.show()

n_sims = int(5e3)
plt.hist(np.random.lognormal(0.5, 0.9, size=n_sims), bins=40)
plt.show()
