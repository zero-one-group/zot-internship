import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as la

x = 15

def squared(x):
    return x ** 2


def cube(x):
    return x ** 3


theta = 90 
math.sin(theta)         #theta is in radians
math.cos(theta)

meshPoints = np.linspace(-1, 1, num=500)
meshPoints[52]

plt.plot(meshPoints,np.sin(2*math.pi*meshPoints))
plt.savefig('testplot.jpg')
plt.show()


