'''
Part 1 of numpy, scipy, and matplotlib exercises
'''

import numpy as np
import matplotlib.pyplot as plt

x = 2
print(np.square(x))
print(np.power(x, 3))

theta = np.pi/2
print(np.sin(theta))
print(np.cos(theta))

meshPoints = np.linspace(-1, 1, 500)
print(meshPoints[52])

plt.plot(meshPoints, np.sin(2*np.pi*meshPoints))
plt.savefig('sinusoid.jpg')
