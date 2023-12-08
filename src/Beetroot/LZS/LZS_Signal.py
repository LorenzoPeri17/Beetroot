from Beetroot.engine import *
from Beetroot.signal import _compute_signal, _compute_signal_loss
from typing import Union, Tuple

@jit(nbtypes.double[:](nbtypes.double, nbtypes.double, 
                       nbtypes.double[:], 
                       nbtypes.double[:],
                       nbtypes.int32, nbtypes.double[:], 
                       nbtypes.double, nbtypes.double),
    fastmath = True, nopython = False, forceobj = True)
def _compute_LZS_signal(Gamma, kt, 
                    BJ_rf_array, 
                    BJ_MW_array, 
                    N, eps, 
                    omega_rf, omega_MW):

    convergence_index_MW = BJ_MW_array.shape[0]

    Y = BJ_MW_array[0]**2 * _compute_signal(Gamma, kt, BJ_rf_array, N, eps, omega_rf)

    for j_MW in range(1, convergence_index_MW):
            
            _MW_factor = BJ_MW_array[j_MW]**2
    
            Y += _MW_factor * _compute_signal(Gamma, kt, BJ_rf_array, N, eps + j_MW*omega_MW , omega_rf)
            Y += _MW_factor * _compute_signal(Gamma, kt, BJ_rf_array, N, eps - j_MW*omega_MW , omega_rf)
    
    return Y

def get_LZS_Signal( N : int, eps : np.ndarray, 
                    de_rf:float, de_MW:float, 
                    Gamma:float, kt:float, 
                    omega_rf:float, omega_MW:float,
                    toll:float = 1e-4) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

    _dew_rf = de_rf/omega_rf
    _dew_MW = de_MW/omega_MW

    BJ_rf_array = get_Js(N+1, _dew_rf, toll) 
    BJ_MW_array = get_Js(1, _dew_MW, toll)
    # ^ The MW only couple as J_i**2
    # ^ the RF as J_i * J_(i+N)

    Y = _compute_LZS_signal(Gamma, kt, BJ_rf_array, BJ_MW_array, N, eps, omega_rf, omega_MW)

    return Y
    
@jit(nbtypes.double[:](nbtypes.double, nbtypes.double, 
                       nbtypes.double[:], 
                       nbtypes.double[:],
                       nbtypes.double[:],
                       nbtypes.int32, nbtypes.double[:], 
                       nbtypes.double, nbtypes.double,
                       nbtypes.double, nbtypes.double),
    fastmath = True, nopython = False, forceobj = True)
def _compute_LZS_signal_loss(Gamma, kt, 
                    BJ_rf_array, 
                    BJ_MW_array, 
                    MIB_array,
                    N, eps, 
                    omega_rf, omega_MW,
                    kappa, prefact):
    
    '''
    Assume the loss is FOR THE MW
    '''

    convergence_index_MW = BJ_MW_array.shape[0]

    _Gamma_m = Gamma + prefact

    Y = np.zeros_like(eps)

    _Bessel_factor = BJ_MW_array[0]**2

    for j_MIB in range(MIB_array.shape[0]):

        _MIB_factor = MIB_array[j_MIB]

        _Gamma_MIB = _Gamma_m + j_MIB * kappa

        Y = _MIB_factor * _Bessel_factor * _compute_signal(_Gamma_MIB, kt, BJ_rf_array, N, eps, omega_rf)

    for j_MW in range(1, convergence_index_MW):
            
            _MW_factor = BJ_MW_array[j_MW]**2

            for j_MIB in range(MIB_array.shape[0]):

                _MIB_factor = MIB_array[j_MIB]

                _Gamma_MIB = _Gamma_m + j_MIB * kappa

                Y += _MIB_factor * _MW_factor * _compute_signal(_Gamma_MIB, kt, BJ_rf_array, N, eps + j_MW*omega_MW , omega_rf)
                Y += _MIB_factor * _MW_factor * _compute_signal(_Gamma_MIB, kt, BJ_rf_array, N, eps - j_MW*omega_MW , omega_rf)
    
    return Y