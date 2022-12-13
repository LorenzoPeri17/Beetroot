import sys

sys.path.append('../src')

from engine import *

import pytest
import numpy as np

def _func_to_spline(x, a):
    return np.exp(np.sin(a*x) - x**2)

@pytest.mark.parametrize('a',np.random.uniform(0.5, 3, 10))
def test_spline(a):
    
    x, dx =np.linspace(-10 /a, 10/a, 1001, retstep=True)

    spline = interp1d(-10 /a, 10/a, dx, _func_to_spline(x, a), k = 3)

    x_rnd = np.random.uniform(-10 /a, 10/a, 1)

    assert np.allclose(_func_to_spline(x_rnd, a), spline(x_rnd), rtol = 1e-5)

@pytest.mark.parametrize('kt, Gamma',[(a, b) for a, b in zip(np.random.uniform(0, 3, 10), np.random.uniform(0, 3, 10))])
def test_interpolate(kt, Gamma):

    x, dx =np.linspace(-10 *kt, 10*kt, 1001, retstep=True)

    generate_convolved_spline(x, dx, Gamma, kt)