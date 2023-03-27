import multiprocessing as mp
from numba import njit, jit, prange
from numba import types as nbtypes

from scipy import special
import numba_scipy

from Beetroot.engine import *
from typing import Union, Tuple


@jit(nbtypes.double[:](nbtypes.double, nbtypes.double, nbtypes.double[:], nbtypes.int32, nbtypes.double[:], nbtypes.double),
    fastmath = True, nopython = False, forceobj = True)
def _compute_signal(Gamma, kt, 
                    BJ_array, 
                    N, eps, omega):

    convergence_index_J = BJ_array.shape[0]

    BesselJ_factor_0 = (BJ_array[0]* BJ_array[N] * (1+power_sign(N)))

    Y0 =  BesselJ_factor_0 * broadened_FD_digamma(eps, Gamma, kt)

    m_sign = 1

    for m in range(1, convergence_index_J-N):

        m_sign = (m_sign^-0b1) + 1 # invert sign in 2's complement
        J_m = BJ_array[m]

        J_factor_p = (J_m * (BJ_array[m+N] + handle_negative_index(m-N, BJ_array)))
        J_factor_m = m_sign * (J_m * (handle_negative_index(-m+N, BJ_array) + handle_negative_index(-m-N, BJ_array)))

        Y0 += J_factor_p * broadened_FD_digamma(eps - m*omega, Gamma, kt)
        Y0 += J_factor_m * broadened_FD_digamma(eps + m*omega, Gamma, kt)
    
    return Y0

@jit(nbtypes.double[:](nbtypes.double, nbtypes.double, 
                       nbtypes.double[:], nbtypes.double[:],
                       nbtypes.int32, nbtypes.double[:], nbtypes.double, nbtypes.double, nbtypes.double),
    fastmath = True, nopython = False, forceobj = True)
def _compute_signal_loss(Gamma, kt, 
                    BJ_array, MIB_array,
                    N, eps, omega, kappa, prefact):

    convergence_index_J = BJ_array.shape[0]

    BesselJ_factor_0 = (BJ_array[0]* BJ_array[N] * (1+power_sign(N)))

    _Gamma_m = Gamma + prefact

    Y0 = np.zeros_like(eps)

    for j_MIB in range(MIB_array.shape[0]):

        _MIB_factor = MIB_array[j_MIB]

        _Gamma_MIB = _Gamma_m + j_MIB * kappa
        
        Y0 += _MIB_factor * BesselJ_factor_0 * broadened_FD_digamma(eps, _Gamma_MIB, kt)

    m_sign = 1

    for m in range(1, convergence_index_J-N):

        m_sign = (m_sign^-0b1) + 1 # invert sign in 2's complement
        J_m = BJ_array[m]

        J_factor_p = (J_m * (BJ_array[m+N] + handle_negative_index(m-N, BJ_array)))
        J_factor_m = m_sign * (J_m * (handle_negative_index(-m+N, BJ_array) + handle_negative_index(-m-N, BJ_array)))

        for j_MIB in range(MIB_array.shape[0]):

            _MIB_factor = MIB_array[j_MIB]

            _Gamma_MIB = _Gamma_m + j_MIB * kappa
            
            Y0 += _MIB_factor * J_factor_p * broadened_FD_digamma(eps - m * omega, _Gamma_MIB, kt)
            Y0 += _MIB_factor * J_factor_m * broadened_FD_digamma(eps + m * omega, _Gamma_MIB, kt)
    
    return Y0

# ! Here for backwards compatibility
# ! please use get_Signal instead
def get_Single_Signal(N : int, eps : np.ndarray, de:float, 
                Gamma:float, omega:float, kt:float, 
                toll:float = 1e-4) -> np.ndarray:
    return get_Signal(N, eps, de, Gamma, omega, kt, toll = toll)

def get_Signal(N : int, eps : np.ndarray, de:float, 
                Gamma:float, omega:float, kt:float, 
                toll:float = 1e-4) -> np.ndarray:

    _dew = de/omega

    BJ_array = get_Js(N+1, _dew, toll)

    Y0 = _compute_signal(Gamma, kt, BJ_array, N, eps, omega)

    return Y0

def get_Signal_Loss(N : int, eps : np.ndarray, de:float, 
                Gamma:float, omega:float, kt:float, g:float,kappa :float,
                toll:float = 1e-4, maxiter_MIB :int = 30) -> np.ndarray:

    if eps.shape[0]&1!=1:
        raise ValueError('eps0 mst be ODD')

    _dew = de/omega

    BJ_array = get_Js(N+1, _dew, toll)

    MIB_array = get_MIB_factors(de, g, toll, maxiter_MIB)

    prefact = kappa * (de/(2*g))**2

    Y0 = _compute_signal_loss(Gamma, kt, BJ_array, MIB_array, N, eps, omega, kappa, prefact)

    return Y0 

