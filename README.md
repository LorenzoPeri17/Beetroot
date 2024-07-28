# Beetroot

![win](https://github.com/LorenzoPeri17/Beetroot/actions/workflows/windows.yaml/badge.svg)
![ubu](https://github.com/LorenzoPeri17/Beetroot/actions/workflows/ubuntu.yaml/badge.svg)
![mac](https://github.com/LorenzoPeri17/Beetroot/actions/workflows/macOs.yaml/badge.svg)

Beetroot is a project to compute the signal arising from a Single Electron Box.

It uses the Bessel function method to speedup the calculation and it already includes Lifetime Broadening.

The theory behind this module can be found in the [related article in Quantum](https://quantum-journal.org/papers/q-2024-03-21-1294/).

## Citing

If you use `Beetroot`, please cite

Peri, L., Oakes, G. A., Cochrane, L., Ford, C. J. B., & Gonzalez-Zalba, M. F. (2024). Beyond-adiabatic Quantum Admittance of a Semiconductor Quantum Dot at High Frequencies: Rethinking Reflectometry as Polaron Dynamics. Quantum, 8, 1294. https://doi.org/10.22331/q-2024-03-21-1294

von Horstig, F.-E., Peri, L., Barraud, S., Shevchenko, S. N., Ford, C. J. B., Gonzalez-Zalba, M. F. Floquet Interferometry of a Dressed Semiconductor Quantum Dot. http://arxiv.org/abs/2407.14241.

## Installation

### From pypi

Beetroot is available from pypi via

```bash
pip install Beetroot-SEB
```

### From source

To install this package from source, download this repo and simply

```bash
pip install .
```

### Testing

To test `Beetroot`, install from source as

``` bash
pip install .[test]
```

> Please remember to escape the brackets if using `zsh` (i.e. `pip install .\[test\]`)

Testing can now be conducted via

``` bash
pytest
```

## Usage

The simples calculation one might want to do is to compute the admittance of a single electron box at the N-th harmonic.
This is done by the following code:

```python
from Beetroot.signal import get_Signal

kt = 1
de = 0.5
omega = 1e-2
Gamma = 0.25

N = 1 # N-th harmonic

eps = np.linspace(-5* kt,  5 * kt, 1001)

Y = get_Signal(N, eps, de, Gamma, omega, kt)
```

The complete script can be found in `examples/fundamental.py` and generates this figure.

![Admittance of the SEB for $N=1$](https://raw.github.com/LorenzoPeri17/Beetroot/main/Figures/fundamental.png)

> Note : Beetroot by default computes the *non-normalized* admittance. I.e. it does *NOT* include the phase. This can be fixed by including
>
> ```python
> from Beetroot import get_precoeff
> Complex_Admittance = Y * get_precoeff(N, Gamma, omega)
> ```

It is sometimes useful to compute maps as a function of power. This is already implemented and *parallelized*.

```python
from Beetroot.parallel import get_map

kt = 1
omega = 1e-2
Gamma = 0.25

N =1 # N-th harmonic

de = np.linspace(0, 10, 201)

eps = np.linspace(-12,  12, 501)

Y = get_map(N, eps, de, Gamma, omega, kt)
```

The complete script can be found in `examples/map.py` and generates this figure.

![Admittance of the SEB for $N=1$](https://raw.github.com/LorenzoPeri17/Beetroot/main/Figures/map.png)

> Functions in `Beetroot.parallel` are parallelized via `multiprocessing.pool`. To avoid recusive imports on non-linux platforms, please wrap the main file in

> ```python
> from multiprocessing import freeze_support
> 
> if __name__ == '__main__':
>    freeze_support()
>```
>
> or consult the [multiprocessing documentation](https://docs.python.org/3/library/multiprocessing.html).

## Small Signal regime

It is desirable to obtain the small signal response. This can be done by specifying a small `de` in `get_Signal`. However, this can lead to **slow conde and poor numerical precision**.
The `fundamental` submodule contains a **analytical implementations** of the small-signal response.

```python
from Beetroot.fundamental import get_Small_Signal_Fundamental


kt = 1
omega = 1e-2
Gamma = 0.25

eps = np.linspace(-5* kt,  5 * kt, 1001)

Yss = get_Small_Signal_Fundamental(eps, Gamma, omega, kt)
```

The complete script can be found in `examples/small_signal.py` and generates this figure.

![Small signal admittance of the SEB for $N=1$](https://raw.github.com/LorenzoPeri17/Beetroot/main/Figures/small_signal.png)

> The small signal admittance is **normalizetd to the input amplitude**, so that `Yss = Y/(2*de)`


## Two-Tone experiments

Beetroot also includes a module to compute the response of an SEB to a two-tone excitation.

The theory and experimental implementation behind this module can be found in [this paper](https://arxiv.org/abs/2407.14241).

This is done by the following code:

```python
from Beetroot.LZS import get_LZS_Signal

kt = 1
Gamma = 0.25

N = 1 # N-th harmonic

omega_rf = 2
de_rf = 0.4

omega_MW = 15
de_MW = 20

eps = np.linspace(-2.5 * de_MW,  2.5 * de_MW, 501)

Y = get_LZS_Signal(N, eps, de_rf, de_MW, Gamma, kt, omega_rf, omega_MW)
```

The complete script can be found in `examples/two_tone.py` and generates this figure.

![Admittance of a dressed SEB for $N=1$](https://raw.github.com/LorenzoPeri17/Beetroot/main/Figures/two_tone.png)

One can compute the two-tone maps in parallel.
To sweep the `de_MW` parameter, use

```python
from Beetroot.LZS import get_LZS_map_MW

kt    = 1
Gamma = 4

N = 1 # N-th harmonic

omega_rf = 2
de_rf = 0.4

de_MW_array = np.linspace(0, 10*omega_MW, 101)
eps = np.linspace(-1.2 * de_MW_array.max(),  1.2 * de_MW_array.max(), 1001)

Y = get_LZS_map_MW(N,eps,de_MW_array, de_rf,Gamma, kt, omega_rf, omega_MW)
```

The complete script can be found in `examples/two_tine_MW_map.py` and generates this figure.

![Admittance of a dressed SEB for $N=1$](https://raw.github.com/LorenzoPeri17/Beetroot/main/Figures/MW_map.png)

To sweep the `de_rf` parameter instead, use

```python
from Beetroot.LZS import get_LZS_map_rf

kt    = 1
Gamma = 4

N = 1 # N-th harmonic

omega_rf = 2

omega_MW = 15
de_MW    = 2

de_rf_array = np.linspace(0, 10*omega_MW, 101)
eps = np.linspace(-1.2 * de_rf_array.max(),  1.2 * de_rf_array.max(), 501)

Y = get_LZS_map_rf(N, eps, de_MW, de_rf_array, Gamma, kt, omega_rf, omega_MW)
```

The complete script can be found in `examples/two_tine_rf_map.py` and generates this figure.

![Admittance of a dressed SEB for $N=1$](https://raw.github.com/LorenzoPeri17/Beetroot/main/Figures/rf_map.png)
