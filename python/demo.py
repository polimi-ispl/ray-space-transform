import numpy as np
from RST import rst
from FRST import frst
from scipy.io import savemat, loadmat


p = loadmat('p.mat')['p']

d = 0.1
L = 64
W = 101
mbar = 0.060
qbar = 0.1
f = 1000
c = 340
sigma = 5*d
N=128

B=5

Z, m, q, z = rst(p, f, c, d, L, mbar, W, qbar, sigma)

Z_hat, m, q, z = frst(p, f, c, d, L, mbar, W, qbar, sigma, N, B)



import matplotlib.pyplot as plt

fig = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'm', fontsize=16)
plt.ylabel(r'q', fontsize=16)
plt.imshow(np.abs(Z_hat),cmap=plt.get_cmap('bone'))
plt.gca().invert_yaxis()
plt.show()

savemat('Z_hat.mat', mdict={'Z_hat': Z_hat})


