import numpy as np
from matplotlib import pyplot as plt

from Beetroot.parallel import get_map
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    kt = 1
    omega = 1e-2
    Gamma = 0.25

    N =1 # N-th harmonic

    de = np.linspace(0, 10, 201)

    eps = np.linspace(-12,  12, 501)

    Y = get_map(N, eps, de, Gamma, omega, kt)

    fig, ax = plt.subplots(1, 1, figsize = (6, 4))

    im = ax.pcolormesh(eps, de, np.abs(Y), rasterized = True)
    cd = fig.colorbar(im, ax = ax, label = r'$|Y_1(\varepsilon_0, \delta \varepsilon)|$')

    ax.set_xlabel(r'$\varepsilon_0 / k_B T$')
    ax.set_ylabel(r'$\delta \varepsilon / k_B T$')

    plt.tight_layout()
    plt.show()