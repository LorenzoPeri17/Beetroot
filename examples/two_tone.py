import numpy as np
from matplotlib import pyplot as plt

from Beetroot.LZS import get_LZS_Signal

kt = 1
Gamma = 0.25

N = 1 # N-th harmonic

omega_rf = 2
de_rf = 0.4

omega_MW = 15
de_MW = 20

eps = np.linspace(-2.5 * de_MW,  2.5 * de_MW, 501)

Y = get_LZS_Signal(N, eps, de_rf, de_MW, Gamma, kt, omega_rf, omega_MW)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.plot(eps, np.abs(Y))

ax.set_xlabel(r'$\varepsilon_0 / k_B T$')
ax.set_ylabel(r'$|Y_1(\varepsilon_0)|$')

plt.tight_layout()
plt.show()