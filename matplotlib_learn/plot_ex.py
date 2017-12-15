from matplotlib import pyplot as plt
import numpy as np

n = 256
X = np.linspace(-np.pi, np.pi, n, endpoint=True)
Y = np.sin(2*X)
plt.axes([0.025, 0.025, 0.95, 0.95])

plt.plot(X, Y+1, color='blue', alpha=1)
plt.fill_between(X, 1, Y+1, color='blue', alpha=0.25)

plt.plot(X, Y-1, color='blue', alpha=1)
plt.fill_between(X, -1, Y-1, (Y-1) > -1, color='blue', alpha=0.25)
plt.fill_between(X, -1, Y-1, (Y-1) < -1, color='red', alpha=0.25)


plt.xlim(-np.pi*1.2, np.pi*1.2)
plt.ylim(-2.5, 2.5)
plt.xticks([])
plt.yticks([])
plt.show()

