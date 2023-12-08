# Beetroot
 
Beetroot is a project to compute the signal arising from a Single Electron Box.

It uses the Bessel function method to speedup the calculation and it already includes Lifetime Broadening.

## Usage

The simples calculation one might want to do is to compute the admittance of a single electron box at the N-th harmonic.
This is done by the following code:

```python
from Beetroot import get_Signal

kt = 1
de = 0.5
omega = 1e-2
Gamma = 0.25

N =1 # N-th harmonic

eps = np.linspace(-5* kt,  5 * kt, 1001)

Y = get_Single_Signal(N, eps, de, Gamma, omega, kt)
```

> Note : Beetroot by default computes the *non-normalized* admittance. I.e. it does *NOT* include the phase. This can be fixed by including
> ```python
> from Beetroot import get_precoeff
> Complex_Admittance = Y * get_precoeff(N, Gamma, omega)
> ```

It is sometimes useful to compute maps as a function of power. This is already implemented and *parallelized*.

```python
from Beetroot import get_map

kt = 1
omega = 1e-2
Gamma = 0.25

N =1 # N-th harmonic

de = np.linspace(0, 10, 101)

eps = np.linspace(-50,  50, 1001)

Y = get_map(N, eps, de, Gamma, omega, kt)
```

# Two-Tone experiments

Beetroot also includes a module to compute the two-tone signal. This is done by the following code:

```python
from Beetroot.LZS import get_LZS_Signal

kt = 1
Gamma = 0.25

N =1 # N-th harmonic

omega_rf = 1e-2
de_rf = 0.5

omega_MW = 0.5
de_MW =2

eps = np.linspace(-5* kt,  5 * kt, 1001)

get_LZS_Signal(N, eps, de_rf, de_MW, Gamma, kt, omega_rf, omega_MW)
```

<!-- Similarly, one can compute the two-tone map:

```python

kt = 1
Gamma = 0.25

N =1 # N-th harmonic

omega_rf = 1e-2
de_rf = 0.5

omega_MW = 0.5
de_MW =2

eps = np.linspace(-5* kt,  5 * kt, 1001)

``` -->