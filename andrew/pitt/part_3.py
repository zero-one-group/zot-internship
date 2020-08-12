import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
f_x = np.exp(-x/10) * np.sin(np.pi*x)
g_x = x * np.exp(-x/3)

plt.plot(x, f_x, label = 'y = e^(-x/10) * sin(pi*x)')
plt.plot(x, g_x, label = 'y = x*e^(-x/3)')
plt.legend(loc = 'lower right')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('plotting.jpg')



theta = np.linspace(0, 2*np.pi, 100)
r_0 = [0.8, 1, 1.2]
plt.figure()
for r_0 in r_0:
    r = r_0 + np.cos(theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    plt.plot(x, y, label = 'r_0 = ' + str(r_0))
    plt.legend(loc = 'lower right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Limacon')
    plt.savefig('limacon.pdf')


