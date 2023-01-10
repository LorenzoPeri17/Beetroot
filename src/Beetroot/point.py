from scipy.integrate import quad_vec
from functools import lru_cache

import multiprocessing as mp
from numba import njit, jit, prange
from numba import types as nbtypes

from scipy import special
import numba_scipy

from Beetroot.engine import *

@njit(nbtypes.int64(nbtypes.double, nbtypes.double),
     fastmath = True)
def get_convergence_index(dew, toll):

    m = float(0)

    tot = special.jv(m, dew)**2

    end = 1-toll

    while tot < end:
        m+=1
        tot += 2*special.jv(m, dew)**2
    
    return int(m)

@njit(nbtypes.int64(nbtypes.double, nbtypes.double),
     fastmath = True)
def get_convergence_index_I(dew, toll):

    m = float(0)

    tot = special.ive(m, dew)

    end = 1-toll

    while tot < end:
        m+=1
        tot += 2*special.ive(m, dew)
    
    return int(m)

def F_M(eps,m , Gamma, omega, kt):

    @njit(fastmath = True)
    def _integrand(e):
        return FermiDirac(e/kt) * Lorentz(e, eps + m *omega, Gamma)

    res, _ = quad_vec(_integrand, -np.infty, np.infty)

    return res

def get_single_point(N, eps, de, Gamma, omega, kt, toll = 1e-4):
    
    _dew = de/omega

    Y = J(0, _dew)*(J(N, _dew) +  J(-N, _dew)) * F_M(eps, 0,  Gamma, omega, kt)

    index = max(get_convergence_index(_dew, toll), N) + N + 1

    for m in range(1, index):

        Y += J(m, _dew)*(J(m+N, _dew) + J(m-N, _dew)) * F_M(eps, m, Gamma, omega, kt)
        Y += J(-m, _dew)*(J(-m+N, _dew) +J(-m-N, _dew)) * F_M(eps, -m, Gamma, omega, kt)

    return Y

@lru_cache
def get_bessel_factor(dew, m, N):
    return J(m, dew)*(J(m+N, dew) + J(m-N,  dew))

def get_point(N, eps, de, Gamma, omega, kt, toll = 1e-4):

    _dew = de/omega

    _dew_2 = (_dew)**(2)

    index_bessel = get_convergence_index(_dew, toll)

    index_I = get_convergence_index_I(_dew_2, toll)

    index = max(index_bessel, N) + N + 1

    factor = Ie(0, _dew_2)

    Y = factor * get_bessel_factor(_dew, 0, N) * (F_M(eps, 0,  Gamma, omega, kt))

    for m in range(1, index):

        Y += factor * get_bessel_factor(_dew,  m, N) * (F_M(eps, m, Gamma, omega, kt))
        Y += factor * get_bessel_factor(_dew, -m, N) * (F_M(eps, -m, Gamma, omega, kt))

    
    for i in range(1, index_I):

        factor = Ie(i, _dew_2)

        Y += factor * get_bessel_factor(_dew, 0, N) * (F_M(eps, i,  Gamma, omega, kt) + F_M(eps, -i,  Gamma, omega, kt))

        for m in range(1, index):

            Y += factor * get_bessel_factor(_dew,  m, N) * (F_M(eps, m-i, Gamma, omega, kt) + F_M(eps, m+i, Gamma, omega, kt))
            Y += factor * get_bessel_factor(_dew, -m, N) * (F_M(eps, -m-i, Gamma, omega, kt) +F_M(eps, -m+i, Gamma, omega, kt))

    return Y