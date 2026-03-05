
from src.Maxwell.FOM import assemble_current, smooth_pulse

import numpy as np
import scipy.linalg

class MaxwellPODROM:
    def __init__(self, sim, U_E, U_B, morth_tol=1e-12):
        self.sim = sim
        self.U_E = np.asarray(U_E)
        self.U_B = np.asarray(U_B)
        self.morth_tol = float(morth_tol)

        M_E = sim.emass
        M_B = sim.bmass
        Wk  = sim.wkcurl
        St  = sim.stcurl

        # Precompute reduced operators (at max r)
        self.MEr_full = self.U_E.T @ (M_E @ self.U_E)
        #self.MEr_full = 0.5*(self.MEr_full + self.MEr_full.T)

        self.Cwr_full = self.U_E.T @ (Wk @ self.U_B)
        self.Csr_full = self.U_B.T @ M_B @ (St @ self.U_E)
        #self.Csr_full = self.U_B.T  @ (St @ self.U_E)

        # forcing cache optional
        self._Jr_cache = None
        self._Jr_sig = None

    def _signature(self, t0, tf, nsteps):
        mid = getattr(self.sim, "mid", None)
        width = getattr(self.sim, "width", None)
        direc = getattr(self.sim, "dir", None)
        if mid is not None: mid = tuple(np.asarray(mid).ravel().tolist())
        if direc is not None: direc = tuple(np.asarray(direc).ravel().tolist())
        return (mid, width, direc, float(t0), float(tf), int(nsteps))

    def precompute_reduced_current(self, t0, tf, nsteps):
        sig = self._signature(t0, tf, nsteps)
        if self._Jr_cache is not None and self._Jr_sig == sig:
            return

        dt = (tf - t0) / nsteps
        times = np.array([t0 + k*dt for k in range(nsteps+1)], dtype=float)

        rEmax = self.U_E.shape[1]
        Jr = np.zeros((rEmax, nsteps+1), dtype=float)

        cc_group = self.sim.file_hdf5['Current_Construction']
        for k, t in enumerate(times):
            source_j = lambda x, tt=t: self.sim.jfunc(tt, x, tf)
            j_full = assemble_current(source_j, cc_group)
            Jr[:, k] = self.U_E.T @ j_full

        self._Jr_cache = (times, Jr)
        self._Jr_sig = sig

    def _get_mass_handler(self, r):
        """
        Returns:
          use_identity: bool
          MEr: (r,r) matrix (only meaningful if not identity)
          solve(rhs): rhs if identity else Cholesky solve
        """
        MEr = self.MEr_full[:r, :r]
        I = np.eye(r, dtype=MEr.dtype)

        rel_err = np.linalg.norm(MEr - I, 'fro') / np.linalg.norm(I, 'fro')
        use_identity = (rel_err <= self.morth_tol)

        if use_identity:
            def solve(rhs):
                return rhs
            return True, None, solve

        chol = scipy.linalg.cho_factor(MEr, lower=False, check_finite=False)
        def solve(rhs):
            return scipy.linalg.cho_solve(chol, rhs, check_finite=False)
        return False, MEr, solve

    def timeLoop(self, t0, tf, nsteps, r=None, record_freq=1, use_forcing_cache=True):
        rmax = min(self.U_E.shape[1], self.U_B.shape[1])
        if r is None: r = rmax
        r = int(r)
        if r < 1 or r > rmax:
            raise ValueError(f"r must be in [1,{rmax}]")

        dt = (tf - t0)/nsteps

        # sub-blocks
        Cwr = self.Cwr_full[:r, :r]
        Csr = self.Csr_full[:r, :r]

        use_I, MEr, solve = self._get_mass_handler(r)

        # forcing
        if use_forcing_cache:
            self.precompute_reduced_current(t0, tf, nsteps)
            _, Jr_full = self._Jr_cache
            Jr = Jr_full[:r, :]
            j_at = lambda k: Jr[:, k]
        else:
            cc_group = self.sim.file_hdf5['Current_Construction']
            def j_at(k):
                tt = t0 + k*dt
                source_j = lambda x, ttt=tt: self.sim.jfunc(ttt, x, tf)
                j_full = assemble_current(source_j, cc_group)
                return self.U_E[:, :r].T @ j_full

        # state storage (arrays, not lists)
        rec_ids = list(range(0, nsteps+1, record_freq))
        if rec_ids[-1] != nsteps:
            rec_ids.append(nsteps)

        Ehist = np.zeros((r, len(rec_ids)))
        Bhist = np.zeros((r, len(rec_ids)))
        Thist = np.zeros(len(rec_ids))

        e = np.zeros(r)
        b = np.zeros(r)

        rp = 0
        Ehist[:, rp] = e
        Bhist[:, rp] = b
        Thist[rp] = t0

        for i in range(nsteps):
            # Ampere half
            if use_I:
                rhs = e + 0.5*dt*(Cwr @ b) - 0.5*dt*j_at(i)
            else:
                rhs = (MEr @ e) + 0.5*dt*(Cwr @ b) - 0.5*dt*j_at(i)
            e_half = solve(rhs)

            # Faraday full
            b_new = b - dt*(Csr @ e_half)

            # Ampere full
            if use_I:
                rhs = e_half + 0.5*dt*(Cwr @ b_new) - 0.5*dt*j_at(i+1)
            else:
                rhs = (MEr @ e_half) + 0.5*dt*(Cwr @ b_new) - 0.5*dt*j_at(i+1)
            e_new = solve(rhs)

            e, b = e_new, b_new

            if (i+1) in rec_ids:
                rp += 1
                Ehist[:, rp] = e
                Bhist[:, rp] = b
                Thist[rp] = t0 + (i+1)*dt

        return (Ehist, Bhist), Thist


def assemble_deim_entries(source_spatial, coords_r, weighted_basis_r, gids_r, deim_idx):
    """Partial assembly on reduced elements, extract at DEIM DOFs.

    This is a free function so it can be called once per parameter
    (outside the ROM / outside the r-loop) and the result reused.
    """
    func_values = source_spatial(coords_r)
    elmt_vec = np.einsum('ciqd,cqd->ci', weighted_basis_r, func_values)
    vec = np.zeros(np.amax(gids_r) + 1)
    np.add.at(vec, gids_r, elmt_vec)
    return vec[deim_idx]


class MaxwellHyperROM(MaxwellPODROM):
    """ROM with Q-DEIM hyperreduction for the current source term.

    Exploits time-space separability:
        J(t, x; mu) = smooth_pulse(t, T) * J_spatial(x; mu)

    Only assembles the current on a reduced set of elements identified
    by Q-DEIM, then reconstructs the reduced forcing via a precomputed
    interpolation matrix.
    """

    def __init__(self, sim, U_E, U_B, U_J, deim_idx, P_UJ_inv,
                 coords_r, weighted_basis_r, gids_r, morth_tol=1e-12):
        super().__init__(sim, U_E, U_B, morth_tol=morth_tol)

        self.U_J = np.asarray(U_J)
        self.deim_idx = np.asarray(deim_idx)
        self.P_UJ_inv = np.asarray(P_UJ_inv)

        # Restricted assembly data (only elements touching DEIM DOFs)
        self.coords_r = np.asarray(coords_r)
        self.weighted_basis_r = np.asarray(weighted_basis_r)
        self.gids_r = np.asarray(gids_r)

        # Precompute C_J = U_E^T @ U_J @ inv(P^T U_J)  — shape (r_E, p)
        self.C_J = self.U_E.T @ (self.U_J @ self.P_UJ_inv)

    def precompute_reduced_current(self, t0, tf, nsteps, j_deim=None):
        """Override: use Q-DEIM hyperreduction instead of full assembly.

        Parameters
        ----------
        j_deim : (p,) array, optional
            Pre-assembled DEIM entries for the current parameter.
            If None, the partial assembly is done here.  Pass this in
            when the same parameter is used across multiple r values
            to avoid redundant assembly.
        """
        sig = self._signature(t0, tf, nsteps)
        if self._Jr_cache is not None and self._Jr_sig == sig:
            return

        dt = (tf - t0) / nsteps
        times = np.array([t0 + k * dt for k in range(nsteps + 1)], dtype=float)

        if j_deim is None:
            # Spatial source (no time scaling)
            mid = self.sim.mid
            width = self.sim.width
            direc = self.sim.dir

            def source_spatial(x):
                dist2 = np.sum((x - mid)**2, axis=-1)
                return np.exp(-dist2 / (2.0 * width))[..., None] * direc

            j_deim = assemble_deim_entries(
                source_spatial, self.coords_r, self.weighted_basis_r,
                self.gids_r, self.deim_idx)

        jr_spatial = self.C_J @ j_deim  # (r_E,)

        # Scale by smooth_pulse at each timestep
        rEmax = self.U_E.shape[1]
        Jr = np.zeros((rEmax, nsteps + 1), dtype=float)
        for k, t in enumerate(times):
            Jr[:, k] = smooth_pulse(t, tf) * jr_spatial

        self._Jr_cache = (times, Jr)
        self._Jr_sig = sig
