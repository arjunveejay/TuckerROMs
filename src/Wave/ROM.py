"""Reduced-order model for the parametric wave equation."""

import numpy as np
import scipy.sparse as sp


class WavePODROM:
    """Galerkin POD-ROM for the parametric wave equation.

    Uses a single reduced basis U for both the displacement field q and the
    momentum field p (both live in the L2 space W).

    Parameters
    ----------
    fom : WaveFEM subclass
        Assembled full-order model.  Must expose: MW, MWinv, S, MV_A, D,
        q0, p0, forcing(x0, y0), dim.
    U : (Nx, r_max) ndarray
        POD basis columns.
    morth_tol : float
        Relative Frobenius-norm tolerance for declaring U MW-orthonormal
        (triggers the identity-mass shortcut.

    Reconstruction
    --------------
    Given ROM output ``Qr, Pr, Ur``:
        Q_full = Ur @ Qr   # (Nx, Nt)
        P_full = Ur @ Pr   # (Nx, Nt)
    """

    def __init__(self, fom, U, morth_tol=1e-12):
        self.fom = fom
        self.U = np.asarray(U, dtype=float)
        self.morth_tol = float(morth_tol)

        # Precompute reduced stiffness  A_r = U^T (S^T MV_A) U
        # The FOM operator is D_fom = MWinv @ S^T @ MV_A, so
        # MW @ D_fom = S^T @ MV_A  (cancels the MWinv).
        # Galerkin projection with MW-weighted test functions gives:
        #   pr_dot = -(U^T S^T MV_A U) qr + (U^T f) cos   (MW-orth case)
        D_fom = fom.D
        MW_D_fom = fom.MW @ D_fom          # = S^T @ MV_A
        if sp.issparse(MW_D_fom):
            self._Ar_full = self.U.T @ (MW_D_fom @ self.U)
        else:
            self._Ar_full = self.U.T @ np.asarray(MW_D_fom) @ self.U

        # MW Gram matrix – used for M-orthonormality check and IC projection
        self._MWr_full = self.U.T @ (fom.MW @ self.U)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_mass_handler(self, r):
        """Return (use_identity, MWr) for the leading r×r sub-block."""
        MWr = self._MWr_full[:r, :r]
        I = np.eye(r, dtype=MWr.dtype)
        rel_err = np.linalg.norm(MWr - I, "fro") / r**0.5
        use_identity = rel_err <= self.morth_tol
        return use_identity, MWr

    def _project_ic(self, r):
        """Project FOM initial conditions onto the r-dimensional reduced space."""
        Ur = self.U[:, :r]
        use_I, MWr = self._get_mass_handler(r)

        if use_I:
            # U is MW-orthonormal: (U^T MW U)^{-1} U^T MW q0 = U^T MW q0
            q0_r = Ur.T @ (self.fom.MW @ self.fom.q0)
            p0_r = Ur.T @ (self.fom.MW @ self.fom.p0)
        else:
            # Galerkin projection: (Ur^T MW Ur) q0_r = Ur^T MW q0
            q0_r = np.linalg.solve(MWr, Ur.T @ (self.fom.MW @ self.fom.q0))
            p0_r = np.linalg.solve(MWr, Ur.T @ (self.fom.MW @ self.fom.p0))

        return q0_r, p0_r

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, t, omega, x0, y0, r=None):
        """Run the ROM for one parameter instance.

        Parameters
        ----------
        t : (Nt,) array_like
            Uniform time grid (same convention as FOM.solve).
        omega : float
            Driving frequency.
        x0, y0 : float
            Forcing centre coordinates.
        r : int or None
            Reduced dimension to use (≤ r_max).  Defaults to r_max.

        Returns
        -------
        Qr : (r, Nt) ndarray   – reduced displacement history
        Pr : (r, Nt) ndarray   – reduced momentum history
        Ur : (Nx, r) ndarray   – basis columns used (for reconstruction)
        """
        t = np.asarray(t, dtype=float)
        rmax = self.U.shape[1]
        if r is None:
            r = rmax
        r = int(r)
        if not (1 <= r <= rmax):
            raise ValueError(f"r must be in [1, {rmax}], got {r}")

        dt = t[1] - t[0]
        Ur = self.U[:, :r]
        use_I, MWr = self._get_mass_handler(r)

        # Reduced stiffness: Ar = U^T (S^T MV_A) U
        Ar = self._Ar_full[:r, :r]

        # When U is not MW-orthonormal the Galerkin projection gives
        #   MWr dpr/dt = -Ar qr + fr cos,   MWr dqr/dt = pr
        # so the effective reduced operator is MWr^{-1} Ar.
        if use_I:
            Dr = dt / 2.0 * Ar + 2.0 / dt * np.eye(r)
        else:
            MWr_inv = np.linalg.inv(MWr)
            Dr = dt / 2.0 * (MWr_inv @ Ar) + 2.0 / dt * np.eye(r)
        Dr_inv = np.linalg.inv(Dr)

        # Reduced forcing vector
        f_fom = self.fom.forcing(x0, y0)           # (Nx,)
        fr = Ur.T @ f_fom                           # (r,)
        if not use_I:
            fr = MWr_inv @ fr

        # Initial conditions
        q0_r, p0_r = self._project_ic(r)

        # Time stepping (mirrors FOM.solve exactly, but in r dimensions)
        Qr = np.zeros((r, len(t)))
        Pr = np.zeros((r, len(t)))
        Qr[:, 0] = q0_r
        Pr[:, 0] = p0_r

        for i in range(1, len(t)):
            t_half = 0.5 * (t[i - 1] + t[i])
            f_half = fr * np.cos(omega * t_half)

            qr_half = Dr_inv @ (2.0 / dt * Qr[:, i - 1] + Pr[:, i - 1] + 0.5 * dt * f_half)
            Qr[:, i] = 2.0 * qr_half - Qr[:, i - 1]
            Pr[:, i] = 4.0 / dt * (qr_half - Qr[:, i - 1]) - Pr[:, i - 1]

        return Qr, Pr, Ur

    def solve_multi(self, muarr, t, r=None):
        """Run the ROM for multiple parameter vectors.

        Parameters
        ----------
        muarr : (N, 3) array_like
            Each row is [omega, x0, y0] (2-D) or [omega, x0, 0] (1-D).
        t : (Nt,) array_like
        r : int or None

        Returns
        -------
        Qr_arr : (N, r, Nt) ndarray
        Pr_arr : (N, r, Nt) ndarray
        Ur     : (Nx, r) ndarray
        """
        Qr_list, Pr_list = [], []
        Ur = None

        for mu in muarr:
            if self.fom.dim == 2:
                Qr, Pr, Ur = self.solve(t, mu[0], mu[1], mu[2], r=r)
            else:
                Qr, Pr, Ur = self.solve(t, mu[0], mu[1], 0, r=r)
            Qr_list.append(Qr)
            Pr_list.append(Pr)

        return np.array(Qr_list), np.array(Pr_list), Ur
