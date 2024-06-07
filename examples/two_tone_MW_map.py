import numpy as np
from matplotlib import pyplot as plt

from Beetroot.LZS import get_LZS_map_MW

from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    
    kt    = 1
    Gamma = 4

    N = 1 # N-th harmonic

    omega_rf = 2
    de_rf = 0.4

    omega_MW_list = [4, 7, 15]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for omega_MW, ax in zip(omega_MW_list, axes):
        
        de_MW_array = np.linspace(0, 10*omega_MW, 101)
        eps = np.linspace(-1.2 * de_MW_array.max(),  1.2 * de_MW_array.max(), 1001)

        Y = get_LZS_map_MW(N,eps,de_MW_array, de_rf,Gamma, kt, omega_rf, omega_MW)

        im = ax.pcolormesh(eps, de_MW_array, np.abs(Y))
        fig.colorbar(im, ax=ax, label = r'$|Y_1(\varepsilon_0, \delta \varepsilon_{MW})|$')

        ax.set_xlabel(r'$\varepsilon_0 / k_B T$')
        ax.set_ylabel(r'$\delta \varepsilon_{MW} / k_B T$')
        
        ax.set_title(r'$\hbar \omega_{MW} / k_B T= %.0f$' % omega_MW)

    plt.tight_layout()
    plt.show()

