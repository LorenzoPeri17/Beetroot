import numpy as np
from matplotlib import pyplot as plt

from Beetroot.signal import get_Signal

kt = 1
de = 0.5
omega = 1e-2
Gamma = 0.25

N = 1 # N-th harmonic

eps = np.linspace(-5* kt,  5 * kt, 1001)

Y   = get_Signal(N, eps, de, Gamma, omega, kt)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.plot(eps, np.abs(Y))

ax.set_xlabel(r'$\varepsilon_0 / k_B T$')
ax.set_ylabel(r'$|Y_1(\varepsilon_0)|$')

plt.tight_layout()
plt.show()