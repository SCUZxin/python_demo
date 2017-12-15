import numpy as np

a = np.array([(1, 2, 3), (4, 5, 6)])
print(a)

a = np.linspace(0, 2, 9)
print(a)

a = np.empty((2, 2))
print(a)

a = np.ones((2, 2))
print(a)

def f(x, y):
    return 10*x+y

a = np.fromfunction(f, (5, 4), dtype=int)
print(a)

a = np.r_[0:4, 5, 9]
print(a)





# import scipy.misc
# import matplotlib.pyplot as plt

# lena = scipy.misc.face()
# # lena = scipy.misc.ascent()
# acopy = lena.copy()
# aview = lena.view()
# plt.subplot(221)
# plt.imshow(lena)
# plt.subplot(222)
# plt.imshow(acopy)
# plt.subplot(223)
# plt.imshow(aview)
# aview.flat = 0
# plt.subplot(224)
# plt.imshow(aview)
# plt.show()

