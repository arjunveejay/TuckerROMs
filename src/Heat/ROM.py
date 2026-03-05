"""Reduced-order model for the parametric heat equation."""

import numpy as np
import scipy.integrate
import scipy.sparse as sp


class HeatPODROM:
    """Galerkin POD-ROM for the parametric heat equation.


    Parameters
    ----------
    fom : HeatFEM subclass
        Assembled full-order model.  Must expose: M, Minv, As, q0,
        forcing(eps, x0, y0), free, pad, dim.
    U : (Nx_free, r_max) ndarray
        POD basis columns (in the free-DOF space).
    morth_tol : float
        Relative Frobenius-norm tolerance for declaring U M-orthonormal
        (triggers the identity-mass shortcut).

    Reconstruction
    --------------
    Given ROM output ``Qr, Ur``:
        Q_free = Ur @ Qr         # (Nx_free, Nt)
        Q_full = fom.pad(Q_free) # (Nx, Nt)
    """

    def __init__(self, fom, U, morth_tol=1e-12):
        self.fom = fom
        self.U = np.asarray(U, dtype=float)
        self.morth_tol = float(morth_tol)

        # Precompute reduced stiffness matrices: Ar_i = U^T A_i U
        self._Ars_full = []
        for A in fom.As:
            if sp.issparse(A):
                self._Ars_full.append(self.U.T @ (A @ self.U))
            else:
                self._Ars_full.append(self.U.T @ np.asarray(A) @ self.U)

        # M Gram matrix – used for M-orthonormality check and IC projection
        self._Mr_full = self.U.T @ (fom.M @ self.U)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_mass_handler(self, r):
        """Return (use_identity, Mr) for the leading r x r sub-block."""
        Mr = self._Mr_full[:r, :r]
        I = np.eye(r, dtype=Mr.dtype)
        rel_err = np.linalg.norm(Mr - I, "fro") / r**0.5
        use_identity = rel_err <= self.morth_tol
        return use_identity, Mr

    def _project_ic(self, r):
        """Project FOM initial conditions onto the r-dimensional reduced space."""
        Ur = self.U[:, :r]
        use_I, Mr = self._get_mass_handler(r)

        if use_I:
            q0_r = Ur.T @ (self.fom.M @ self.fom.q0)
        else:
            q0_r = np.linalg.solve(Mr, Ur.T @ (self.fom.M @ self.fom.q0))

        return q0_r

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, eps, x0, y0, t, r=None):
        """Run the ROM for one parameter instance.

        Parameters
        ----------
        eps : float
            Forcing amplitude parameter.
        x0, y0 : float
            Forcing centre coordinates.
        t : (Nt,) array_like
            Time grid (same convention as FOM.solve).
        r : int or None
            Reduced dimension to use (<= r_max).  Defaults to r_max.

        Returns
        -------
        Qr : (r, Nt) ndarray   – reduced temperature history
        Ur : (Nx_free, r) ndarray – basis columns used (for reconstruction)
        """
        t = np.asarray(t, dtype=float)
        rmax = self.U.shape[1]
        if r is None:
            r = rmax
        r = int(r)
        if not (1 <= r <= rmax):
            raise ValueError(f"r must be in [1, {rmax}], got {r}")

        Ur = self.U[:, :r]

        # Reduced system matrix: Ar = sum(Ar_i)
        Ar = sum(Ar_i[:r, :r] for Ar_i in self._Ars_full)

        # Reduced mass handling
        use_I, Mr = self._get_mass_handler(r)

        # Reduced forcing vector
        f_fom = self.fom.forcing(eps, x0, y0)  # (Nx_free,)
        fr = Ur.T @ f_fom                       # (r,)

        # Initial conditions
        q0_r = self._project_ic(r)

        # Build the reduced RHS: Mr^{-1} (Ar qr + fr * exp(-t))
        if use_I:
            def fun(tt, y):
                return Ar @ y + fr * np.exp(-tt)

            jac = Ar
        else:
            Mr_inv = np.linalg.inv(Mr)

            def fun(tt, y):
                return Mr_inv @ (Ar @ y + fr * np.exp(-tt))

            jac = Mr_inv @ Ar

        # Solve with BDF (mirrors FOM.solve)
        sol = scipy.integrate.solve_ivp(
            fun=fun,
            t_span=[t[0], t[-1]],
            y0=q0_r,
            method="BDF",
            t_eval=t,
            jac=jac,
            vectorized=False,
        )

        return sol.y, Ur

    def solve_multi(self, params, t, r=None):
        """Run the ROM for multiple parameter vectors.

        Parameters
        ----------
        params : (N, 3) array_like
            Each row is [eps, x0, y0].
        t : (Nt,) array_like
        r : int or None

        Returns
        -------
        Qr_arr : (N, r, Nt) ndarray
        Ur     : (Nx_free, r) ndarray
        """
        Qr_list = []
        Ur = None

        for param in params:
            Qr, Ur = self.solve(param[0], param[1], param[2], t, r=r)
            Qr_list.append(Qr)

        return np.array(Qr_list), Ur
