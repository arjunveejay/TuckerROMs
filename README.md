# TuckerROMs

A two-stage method for constructing projection-based reduced-order models (ROMs) of parameterized PDEs, based on Tucker tensor factorization of solution snapshots.

## Overview

Solution snapshots are encoded offline via a multi-linear Tucker factorization of the snapshot tensor, enabling a reduced basis that varies nonlinearly with PDE parameters to be rapidly constructed online for use in a Galerkin ROM. Two extensions of this strategy are developed: reduced bases orthonormalized with respect to a general discrete inner product, and interpolation of Tucker-encoded states via Radial Basis Functions and the Mamonov/Olshanskii method. The approach targets regimes where monolithic-basis ROMs struggle, including sparse parameter sampling and problems with slowly decaying Kolmogorov $n$-width.

Three basis construction methods are compared:

- **Monolithic**: a single global SVD basis computed from all training snapshots, used as a baseline
- **Radial Basis Functions (RBF)**: a parameter-interpolated basis using Gaussian RBF weights on the Tucker parameter factor
- **Mamonov/Olshanskii (MO)**: a locally informed basis using distance-weighted interpolation on the K nearest training neighbors

## Problems

### Heat equation (`experiments/Heat/`)

2D heat equation on $[0, 2\pi]^2$ with uniform diffusion, homogeneous Dirichlet boundary conditions, and zero initial conditions. The system is forced by a Gaussian source at location $(x_0, y_0)$ with amplitude $\varepsilon$ and a $\sin(x/2)\sin(y/2)$ envelope, decaying as $e^{-t}$. Parameters: $\mu = (\varepsilon, x_0, y_0)$.

### Wave equation (`experiments/Wave/`)

2D wave equation on $[0, 2\pi]^2$ with uniform wave speed and zero initial conditions. The system is forced by a Gaussian source at $(x_0, y_0)$ with a $\sin(x/2)\sin(y/2)$ envelope, oscillating at frequency $\omega$. Parameters: $\mu = (\omega, x_0, y_0)$. Separate reduced bases are built for displacement $Q$ and momentum $P$.

### Maxwell equations (`experiments/Maxwell/`)

Maxwell equations on a 3D domain, tracking the electric field $\boldsymbol{E}$ and magnetic flux density $\boldsymbol{B}$. The system is driven by a Gaussian current source at location $(x_0, y_0, z_0)$ with a fixed width, direction, and smooth temporal envelope. Parameters: $\mu = (x_0, y_0, z_0) \in [0, 2]^3$.

Two additional features distinguish this problem:

- **Curl-enriched bases**: the $\boldsymbol{E}$ basis is augmented with the weak curl of the $\boldsymbol{B}$ modes, and vice versa.

- **Q-DEIM hyperreduction**: approximates the parametric current source in a low-rank form, with a greedy QR-based selection of interpolation points (Q-DEIM) to reduce online assembly cost to scale with the approximation rank rather than the mesh size

## Repository structure

```
src/
  Bases/
    rbf.py          # Radial Basis Function interpolant
    mo.py           # Mamonov/Olshanskii interpolant
  Heat/
    FOM.py          # Finite element heat equation model
    ROM.py          # POD-ROM for the heat equation
    plots.py
  Wave/
    FOM.py          # Finite element wave equation model
    ROM.py          # POD-ROM for the wave equation
    plots.py
  Maxwell/
    FOM.py          # Finite element Maxwell model
    ROM.py          # Enriched hyperreduced-POD-ROM for Maxwell
    error_sweeps.py
    plots.py
  Utils/
    utils.py        # Shared utilities (Tucker loading, norms, orthonormalization)

experiments/
  Heat/
    generate_data.py  # Run FOM over training/test parameters
    tucker.py         # Compute Tucker decomposition and monolithic SVD
    run.ipynb         # ROM experiments and plots
  Wave/
    generate_data.py
    tucker.py
    run.ipynb
  Maxwell/
    generate_data.py
    tucker.py
    generate_current.py  # Compute Q-DEIM basis and interpolation points for the current source
    run.ipynb
```

## Running the experiments

Each experiment is self-contained in its `run.ipynb` notebook under `experiments/Heat/`, `experiments/Wave/`, and `experiments/Maxwell/`.

## Dependencies

- [NGSolve](https://ngsolve.org/) — finite element framework
- [TensorLy](https://tensorly.github.io/) — Tucker decomposition
- NumPy, SciPy, Matplotlib
