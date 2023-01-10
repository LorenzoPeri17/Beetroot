import numpy as np

from matplotlib import pyplot as plt

from Beetroot.signal import *

def plot_map_single(N : int, eps : np.ndarray, de:np.ndarray, 
                Gamma:float, omega:float, kt:float, 
                toll:float = 1e-4, **plotkwargs):

    M = get_map_single(N, eps, de, Gamma, omega, kt, toll)

    M = -M
    M /= M.max()

    fig, ax = plt.subplots(1, 1)

    im = ax.pcolormesh(eps, de, M, **plotkwargs)

    fig.suptitle(rf'$N = {N} : \Gamma = {Gamma}, k_B T = {kt}$', usetex = True)

    ax.set_xlabel(r'$\varepsilon_0$', usetex = True)
    ax.set_ylabel(r'$\delta \varepsilon$', usetex = True)

    fig.tight_layout(pad = 1.075)

    return fig, ax

