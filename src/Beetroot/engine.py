import numpy as np

import multiprocessing as mp
from numba import njit, jit, prange, typeof
from numba import types as nbtypes
from numba.typed import List

# from scipy.special import factorial

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


@jit(nbtypes.double[::1](nbtypes.double, nbtypes.double, nbtypes.double, nbtypes.int32),
     **jit_kwargs, nopython = False, forceobj =True)
def get_MIB_factors(de, g, toll, maxiter):

    _n = -(de/(2*g))**2

    _term = 1
    i = 0

    res = [1]

    _norm = np.exp((de/(2*g))**2)

    while np.abs(_term) < toll and i < maxiter:

        i+=1

        _term = _norm * (_n)**i / special.factorial(i)

        res.append(_term)

    return np.array(res)


@jit(nbtypes.double[:](nbtypes.double[:], nbtypes.double, nbtypes.double),
    fastmath = True, cache = True, forceobj=True)
def broadened_FD_digamma(x, Gamma, kt):

    _2pkt_i = 1/(2*np.pi*kt)
    _gkt = Gamma * _2pkt_i

    _xkt = x * _2pkt_i

    res = special.digamma(0.5 + _gkt +1j*_xkt) #- special.digamma(0.5 + _gkt -1j*_xkt)

    return 0.5 - res.imag/np.pi

@njit(nbtypes.int64(nbtypes.int64), cache = True, **jit_kwargs)
def power_sign(n):
    return (((-1 << (n & 0b1)) << 1) + 0b11) # (-1) ** n in 2's complement

@njit(nbtypes.double(nbtypes.int64, nbtypes.double[:]), cache = True, **jit_kwargs)
def handle_negative_index(n, BJ_array):
    if n >= 0:
        return BJ_array[n]
    else:
        return BJ_array[-n] * power_sign(n)
    