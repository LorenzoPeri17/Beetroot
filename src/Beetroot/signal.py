from scipy.integrate import quad_vec

import multiprocessing as mp
from numba import njit, jit, prange
from numba import types as nbtypes

from scipy import special
import numba_scipy

from Beetroot.engine import *

@jit(fastmath = True, nopython = False, forceobj = True)
def _compute_signal(FD_spline, 
                    BJ_array, BIe_array, 
                    _x0, _dx0, 
                    N, eps, omega):

    convergence_index_J = BJ_array.shape[0]
    convergence_index_I = BIe_array.shape[0]

    BesselJ_factor_0 = (BJ_array[0]* BJ_array[N] * (power_sign(N)))

    Y0 =  BesselJ_factor_0 * FD_spline(_x0)

    m_sign = 1

    for m in range(1, convergence_index_J-N):

        m_sign = (m_sign^-0b1) + 1 # invert sign in 2's complement
        J_m = BJ_array[m]

        J_factor_p = (J_m * (BJ_array[m+N] + handle_negative_index(m-N, BJ_array)))
        J_factor_m = m_sign * (J_m * (handle_negative_index(-m+N, BJ_array) + handle_negative_index(-m-N, BJ_array)))

        Y0 += J_factor_p * FD_spline(_x0 - m * omega)
        Y0 += J_factor_m * FD_spline(_x0 + m * omega)

    Y_0_interp = interp1d(_x0.min(), _x0.max(), _dx0, Y0)

    Y = BIe_array[0] * Y_0_interp(eps)

    for i in range(1, convergence_index_I):

        factor = BIe_array[i]

        Y += 2 * factor * (Y_0_interp(eps + i*omega) + Y_0_interp(eps - i*omega))

    return Y, Y0


@jit(fastmath = True, nopython = False, forceobj = True)
def _compute_single_signal(FD_spline, 
                    BJ_array, 
                    N, eps, omega):

    convergence_index_J = BJ_array.shape[0]

    BesselJ_factor_0 = (BJ_array[0]* BJ_array[N] * (power_sign(N)))

    Y0 =  BesselJ_factor_0 * FD_spline(eps)

    m_sign = 1

    for m in range(1, convergence_index_J-N):

        m_sign = (m_sign^-0b1) + 1 # invert sign in 2's complement
        J_m = BJ_array[m]

        J_factor_p = (J_m * (BJ_array[m+N] + handle_negative_index(m-N, BJ_array)))
        J_factor_m = m_sign * (J_m * (handle_negative_index(-m+N, BJ_array) + handle_negative_index(-m-N, BJ_array)))

        Y0 += J_factor_p * FD_spline(eps - m * omega)
        Y0 += J_factor_m * FD_spline(eps + m * omega)
    
    return Y0


def get_Signal(N, eps, de, Gamma, omega, kt, toll = 1e-4, *, return_single = False):

    _dew = de/omega

    _dew_2 = (_dew)**(2)

    BJ_array = get_Js(N, _dew, toll)

    convergence_index_J = BJ_array.shape[0]

    BIe_array = get_Ies(_dew_2, toll)

    convergence_index_I = BIe_array.shape[0]

    epsmax = eps.max()
    epsmin = eps.min()

    min_x = epsmin - (convergence_index_J + convergence_index_I + 1 - N) * omega
    max_x = epsmax + (convergence_index_J + convergence_index_I + 1 - N) * omega

    _d_eps = (epsmax - epsmin)/eps.shape[0]

    _x, _dx = np.linspace(min_x, max_x, int(np.ceil((max_x - min_x)/_d_eps)), retstep=True)

    min_x0 = epsmin - ( convergence_index_I + 1) * omega
    max_x0 = epsmax + ( convergence_index_I + 1) * omega

    _x0, _dx0 = np.linspace(min_x0, max_x0, int(np.ceil((max_x0 - min_x0)/_d_eps)), retstep=True)

    FD_spline = generate_convolved_spline(_x, _dx, Gamma, kt)

    Y, Y0 = _compute_signal(FD_spline, BJ_array, BIe_array, _x0, _dx0, N, eps, omega)

    if return_single:
        return Y, Y0
    else:
        return Y

def get_Single_Signal(N, eps, de, Gamma, omega, kt, toll = 1e-4):

    _dew = de/omega

    _dew_2 = (_dew)**(2)

    BJ_array = get_Js(N, _dew, toll)

    convergence_index_J = BJ_array.shape[0]

    epsmax = eps.max()
    epsmin = eps.min()

    min_x = epsmin - (convergence_index_J +  1 - N) * omega
    max_x = epsmax + (convergence_index_J +  1 - N) * omega

    _d_eps = (epsmax - epsmin)/eps.shape[0]

    _x, _dx = np.linspace(min_x, max_x, int(np.ceil((max_x - min_x)/_d_eps)), retstep=True)

    FD_spline = generate_convolved_spline(_x, _dx, Gamma, kt)

    Y0 = _compute_single_signal(FD_spline, BJ_array, N, eps, omega)

    return Y0
