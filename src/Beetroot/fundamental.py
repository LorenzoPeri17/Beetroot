from Beetroot.signal import get_Signal

from mpmath import psi as mpm_psi
from scipy.special import psi as sp_psi
import numpy as np

def get_Small_Signal_Fundamental_noFloquet(eps : np.ndarray,
                Gamma:float, kt:float):
    
    _2pkt_i = 1/(2*np.pi*kt)
    _gkt = Gamma * _2pkt_i

    _xkt = eps * _2pkt_i

    _z =  0.5 + _gkt +1j*_xkt
    
    res = np.array([np.cdouble(mpm_psi(1, z0)).real for z0 in _z])

    return res * _2pkt_i / (2*np.pi)

def get_Small_Signal_Fundamental_mpm(eps : np.ndarray,
                Gamma:float, omega:float, kt:float):
    
    _2pkt_i = 1/(2*np.pi*kt)
    _gkt = Gamma * _2pkt_i

    _xkt = eps * _2pkt_i

    _w2p =1j*omega *  _2pkt_i

    _z =  0.5 + _gkt +1j*_xkt
    
    res_p = np.array([np.cdouble(mpm_psi(0, z0 + _w2p)).imag for z0 in _z])
    res_m = np.array([np.cdouble(mpm_psi(0, z0 - _w2p)).imag for z0 in _z])

    return (res_p - res_m )  / (4*np.pi*omega)

def get_Small_Signal_Fundamental(eps : np.ndarray,
                Gamma:float, omega:float, kt:float):
    
    _2pkt_i = 1/(2*np.pi*kt)
    _gkt = Gamma * _2pkt_i

    _xkt = eps * _2pkt_i

    _w2p =1j*omega * _2pkt_i

    _z =  0.5 + _gkt + 1j*_xkt
    
    res_p = sp_psi(_z + _w2p).imag
    res_m = sp_psi(_z - _w2p).imag

    return (res_p - res_m )  / (4*np.pi*omega)

if __name__ == '__main__':

    kt = 1
    omega = 5
    Gamma = 1.2

    de = 0.1

    eps = np.linspace(-5, 5, 101)

    from matplotlib import pyplot as plt

    plt.figure()

    yb = get_Signal(1, eps, de, Gamma, omega, kt)/de/2

    ys = get_Small_Signal_Fundamental(eps, Gamma, omega, kt)

    plt.plot(eps, yb, label = 'Big Signal')

    plt.plot(eps, ys, '--', label = 'Small Signal')

    plt.show()

