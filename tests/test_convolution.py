from Beetroot.engine import *
from Beetroot.engine import _convolve, _numconvolve

import pytest
import numpy as np

@pytest.mark.parametrize('kt, Gamma', [(kt, Gamma) for kt, Gamma in zip(np.random.uniform(1e-6, 1e4, 10), np.random.uniform(1e-6, 1e4, 10))])
def test_num_vs_quad(kt, Gamma):

    xmax = 10*max(Gamma, kt)

    x = np.linspace(-xmax, xmax, num = 101)

    quad = _convolve(x, Gamma, kt)
    num = _numconvolve(x, Gamma, kt)

    print(num)
    print(quad)

    assert np.allclose(quad, num)