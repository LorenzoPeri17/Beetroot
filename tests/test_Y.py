# import sys

# sys.path.append('~/Lindblad/Beetroot/src')

from Beetroot.engine import *
from Beetroot.signal import *

import pytest
import numpy as np

def _generate_inputs():

    N = np.random.randint(1, 3)

    de = np.random.uniform(0, 10)
    omega = np.random.uniform(0, 3)
    Gamma = np.random.uniform(0, 3)
    kt = np.random.uniform(0, 3)

    return N, de, Gamma, omega, kt

@pytest.mark.parametrize('N, de, Gamma, omega, kt',[_generate_inputs() for _ in range(3)])
def test_Y(N, de, Gamma, omega, kt):

    m = max(Gamma, omega, de)
    
    eps = np.linspace(-10*m, 10*m, 101)

    get_Signal(N, eps, de, Gamma, omega, kt)

@pytest.mark.parametrize('N, de, Gamma, omega, kt',[_generate_inputs() for _ in range(3)])
def test_single_Y(N, de, Gamma, omega, kt):

    m = max(Gamma, omega, de)
    
    eps = np.linspace(-10*m, 10*m, 101)

    get_Single_Signal(N, eps, de, Gamma, omega, kt)

def _generate_inputs_dec():

    N = np.random.randint(1, 3)

    de = np.random.uniform(0, 10)
    omega = np.random.uniform(0, 3)
    Gamma = np.random.uniform(0, 3)
    kt = np.random.uniform(0, 3)
    Gamma_phi = np.random.uniform(0.1, 10)


    return N, de, Gamma, omega, kt, Gamma_phi


@pytest.mark.parametrize('N, de, Gamma, omega, kt, Gamma_phi',[_generate_inputs_dec() for _ in range(3)])
def test_dec_Y(N, de, Gamma, omega, kt, Gamma_phi):

    m = max(Gamma, omega, de)
    
    eps = np.linspace(-10*m, 10*m, 101)

    get_Signal_Dec(N, eps, de, Gamma, omega, kt, Gamma_phi)