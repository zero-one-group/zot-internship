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

mesh_points = np.linspace(-1, 1, 500)
print(mesh_points[52])

plt.plot(mesh_points, np.sin(2*np.pi*mesh_points))
plt.savefig('sinusoid.jpg')
