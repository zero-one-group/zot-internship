"""Consider the log-normal distribution logN(0, 1), which can be derived by
exponentiating the N(0, 1) distribution. Show that exp of the mean does not
equal to the mean of the exp. Explain your findings"""

import numpy as np
import matplotlib.pyplot as plt

norm_dist = np.random.normal(0, 1, size=100)
log_normal = np.exp(norm_dist)
plt.plot(norm_dist)
plt.plot(log_normal)
plt.savefig("test3.png")

print("exp of mean is %s" % np.exp(np.mean(norm_dist)))
print("mean of the exp is %s" % np.mean(log_normal))
