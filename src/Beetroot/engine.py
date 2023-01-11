import numpy as np

from fast_interp import interp1d

from scipy.integrate import quad_vec

import multiprocessing as mp
from numba import njit, jit, prange
from numba import types as nbtypes

from scipy import special
import numba_scipy

jit_kwargs = {'fastmath' : True, 'parallel' : False, 'nogil' : True}

@njit(nbtypes.double(nbtypes.int64, nbtypes.double),
    **jit_kwargs)
def J(m : int, x : float):
    return special.jv(float(m), x)

@njit(nbtypes.double(nbtypes.int64, nbtypes.double),
    **jit_kwargs)
def Ie(m : int, x : float):
    return special.ive(float(m), x)

@njit([nbtypes.double(nbtypes.double), 
        nbtypes.double[:](nbtypes.double[:])],
    cache=True, **jit_kwargs)
def FermiDirac(x):
    return 1/(np.exp(x)+1)

@njit([nbtypes.double(nbtypes.double)],
    cache=True, **jit_kwargs)
def H(x):
    if x==0:
        return 1.0
    return (1-np.exp(-x))/x

@njit([nbtypes.double[:](nbtypes.double[:], nbtypes.double[:], nbtypes.double), 
        nbtypes.double[:](nbtypes.double, nbtypes.double[:], nbtypes.double),
        nbtypes.double[:](nbtypes.double[:], nbtypes.double, nbtypes.double), 
        nbtypes.double(nbtypes.double, nbtypes.double, nbtypes.double),
        nbtypes.double[:](nbtypes.double, nbtypes.Array(nbtypes.double, 1, 'C', readonly=True), nbtypes.double)],
    cache=True, **jit_kwargs)
def Lorentz(e, e0, Gamma):
    return (Gamma/np.pi) /(Gamma**2 + (e-e0)**2)

@njit(nbtypes.double[::1](nbtypes.int64, nbtypes.double, nbtypes.double),
     **jit_kwargs)
def get_Js(N, dew, toll):

    res = []

    m = 0

    BJ = J(m, dew)

    tot = BJ**2

    res.append(BJ)

    end = 1-toll

    while tot < end:

        m+=1

        BJ = J(m, dew)

        tot += 2*BJ**2

        res.append(BJ)

    for _ in range(N):

        m+=1

        BJ = J(m, dew)

        res.append(BJ)

    res_array = np.empty(m, dtype=np.double)

    for i, r in enumerate(res):
        res_array[i] = r

    return res_array

@njit(nbtypes.double[:](nbtypes.double, nbtypes.double),
     **jit_kwargs)
def get_Ies(z, toll):

    res = []

    m = 0

    BIe = Ie(m, z)

    tot = BIe

    res.append(BIe)

    end = (1-toll)

    while tot < end:

        m+=1

        BIe = Ie(m, z)

        tot += 2*BIe

        res.append(BIe)
    
    return np.array(res)
   
def generate_convolved_spline(x, dx, Gamma, kt, k = 3):

    @njit(fastmath = True)
    def _integrand(e):
        return (1-FermiDirac(e/kt)) * Lorentz(e, x, Gamma)

    res, _ = quad_vec(_integrand, -np.infty, np.infty)

    return interp1d(x.min(), x.max(), dx, res, k)

def _convolve(x, Gamma, kt):

    @njit(fastmath = True)
    def _integrand(e):
        return (1-FermiDirac(e/kt)) * Lorentz(e, x, Gamma)

    res, _ = quad_vec(_integrand, -np.infty, np.infty)

    return res

@njit(nbtypes.int64(nbtypes.int64), cache = True, **jit_kwargs)
def power_sign(n):
    return (((-1 << (n & 0b1)) << 1) + 0b11) # (-1) ** n in 2's complement

@njit(nbtypes.double(nbtypes.int64, nbtypes.double[:]), cache = True, **jit_kwargs)
def handle_negative_index(n, BJ_array):
    if n >= 0:
        return BJ_array[n]
    else:
        return BJ_array[-n] * power_sign(n)

# @njit(nbtypes.UniTuple(nbtypes.double[:], 2)(InterpType, 
#                                     nbtypes.double[:], nbtypes.double[:],
#                                     nbtypes.double[:], nbtypes.double[:],
#                                     nbtypes.int64, nbtypes.double[:], nbtypes.double), 
#                                     fastmath = True, nogil = True, parallel = True)


   


# def get_Signal(N, eps, de, Gamma, omega, kt, toll = 1e-4):

#     _dew = de/omega

#     _dew_2 = (_dew)**(2)

#     BJ_array = get_Js(N, _dew, toll)

#     convergence_index_J = BJ_array.shape[0]

#     BIe_array = get_Ies(_dew_2, toll)

#     convergence_index_I = BIe_array.shape[0]

#     min_x = eps.min() - (convergence_index_J + convergence_index_I + 1 - N) * omega
#     max_x = eps.max() + (convergence_index_J + convergence_index_I + 1 - N) * omega

#     _d_eps = (eps.max() - eps.min())/eps.shape[0]

#     _x, _dx = np.linspace(min_x, max_x, int(np.ceil((max_x - min_x)/_d_eps)), retstep=True)

#     min_x0 = eps.min() - ( convergence_index_I + 1) * omega
#     max_x0 = eps.max() + ( convergence_index_I + 1) * omega

#     _x0, _dx0 = np.linspace(min_x0, max_x0, int(np.ceil((max_x0 - min_x0)/_d_eps)), retstep=True)

#     FD_spline = generate_convolved_spline(_x, _dx, Gamma, kt)

#     BesselJ_factor_0 = (BJ_array[0]* BJ_array[N] * (power_sign(N)))

#     Y0 =  BesselJ_factor_0 * FD_spline(_x0)

#     m_sign = 1

#     for m in range(1, convergence_index_J-N):

#         m_sign = (m_sign^-0b1) + 1 # invert sign in 2's complement
#         J_m = BJ_array[m]

#         J_factor_p = (J_m * (BJ_array[m+N] + handle_negative_index(m-N, BJ_array)))
#         J_factor_m = m_sign * (J_m * (handle_negative_index(-m+N, BJ_array) + handle_negative_index(-m-N, BJ_array)))

#         Y0 += J_factor_p * FD_spline(_x0 - m * omega)
#         Y0 += J_factor_m * FD_spline(_x0 + m * omega)

#     Y_0_interp = interp1d(min_x0, max_x0, _dx0, Y0)

#     Y = BIe_array[0] * Y_0_interp(eps)

#     for i in range(1, convergence_index_I):

#         factor = BIe_array[i]

#         Y += 2 * factor * (Y_0_interp(eps + i*omega) + Y_0_interp(eps - i*omega))

#     return Y

        
if __name__ == '__main__':

    x = 1e2

    j_array = get_Js(2, x, 1e-4)

    for i, j in enumerate(j_array):

        assert j == J(i, x), f'{i}, {j}, {J(i, x)}'