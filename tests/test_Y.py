import sys

sys.path.append('src')

from engine import *

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