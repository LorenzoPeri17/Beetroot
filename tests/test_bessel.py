import sys

sys.path.append('../src/Beetroot')

from Beetroot.engine import *

import pytest
import numpy as np

from scipy import special as scipyspecial

@pytest.mark.parametrize('N, x', [(N, x) for N, x in zip(np.random.randint(0, 5, 10), np.random.uniform(0, 1e4, 10))])
def test_J_single(N, x):

    assert np.isclose(scipyspecial.jv(N, x), J(N, x))

@pytest.mark.parametrize('N, x', [(N, x) for N, x in zip(np.random.randint(0, 5, 10), np.random.uniform(0, 1e4, 10))])
def test_ie_single(N, x):

    assert np.isclose(scipyspecial.ive(N, x), Ie(N, x))

@pytest.mark.parametrize('N, x', [(N, x) for N, x in zip(np.random.randint(0, 5, 10), np.random.uniform(0, 1e4, 10))])
def test_J(N, x):

    j_array = get_Js(2, x, 1e-4)

    for i, j in enumerate(j_array):

        assert np.isclose(j, J(i, x))

    assert (j_array[0]**2 + np.sum(2*j_array[1:]**2)) > 1-1e-4

@pytest.mark.parametrize('x', [x for x in np.random.uniform(0, 1e4, 10)])
def test_ie(x):

    ie_array = get_Ies(x, 1e-4)

    for i, j in enumerate(ie_array):

        assert np.isclose(j, Ie(i, x))

    assert (ie_array[0] + np.sum(2*ie_array[1:])) > 1-1e-4

@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.parametrize('x', [x for x in np.random.uniform(-1e4, 1e4, 10)])
def test_fd(x):

    assert np.isclose(FermiDirac(x), 1/(np.exp(x)+1))

@pytest.mark.parametrize('x, y, z', [(x, y, z) for x, y, z in zip(np.random.uniform(0, 1e4, 10), np.random.uniform(0, 1e4, 10), np.random.uniform(0, 1e4, 10))])
def test_Lorentz(x, y, z):

    assert np.isclose(Lorentz(x, y, z), z/(z**2 + (x-y)**2))

@pytest.mark.parametrize('N', [N for N in np.random.randint(0, 50, 10)])
def test_sign(N):
    assert power_sign(N) == pow(-1, N)