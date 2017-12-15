import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 2.1, 0.2)
plt.plot(t, t, color='red', linestyle='--', linewidth=2, label='t')
plt.plot(t, t**2, linestyle='-', color='green', linewidth=2, label='t^2')
plt.plot(t, t**3, linestyle='-', color='blue', linewidth=2, label='t^3')

plt.legend(loc='upper left')
plt.show()