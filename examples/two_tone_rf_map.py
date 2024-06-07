import numpy as np
from matplotlib import pyplot as plt

from Beetroot.LZS import get_LZS_map_rf

from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    
    kt    = 1
    Gamma = 4

    N = 1 # N-th harmonic

    omega_rf = 2
    de_rf_array = np.linspace(0, 25, 201)

    omega_MW = 15
    
    de_MW_list = [2, 25]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    for de_MW, ax in zip(de_MW_list, axes):
        
        eps = np.linspace(-1.2 * de_rf_array.max(),  1.2 * de_rf_array.max(), 201)

        Y = get_LZS_map_rf(N, eps, de_MW, de_rf_array, Gamma, kt, omega_rf, omega_MW)

        im = ax.pcolormesh(eps, de_rf_array, np.abs(Y))
        fig.colorbar(im, ax=ax, label = r'$|Y_1(\varepsilon_0, \delta \varepsilon_{rf})|$')

        ax.set_xlabel(r'$\varepsilon_0 / k_B T$')
        ax.set_ylabel(r'$\delta \varepsilon_{rf} / k_B T$')
        
        ax.set_title(r'$\delta \varepsilon_{MW} / k_B T= %.0f$' % de_MW)

    plt.tight_layout()
    plt.show()

