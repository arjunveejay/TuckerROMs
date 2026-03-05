"""
rom_and_proj_error_sweeps.py

Simple sweeps for:
  1) ROM errors on TRAIN and TEST sets (E and B)
  2) Projection errors on TRAIN and TEST sets (E and B)

Supports three basis types:
  - mono: fixed global bases U_mono_E, U_mono_B
  - rbf : parametric bases from RBF weights + buildParBasis
  - lid : parametric bases from LID weights + buildParBasis

Any basis type can be wrapped with enrich_builder() to add curl enrichment:
  E basis augmented with Wk @ U_B, then M_E-orthonormalized
  B basis augmented with St @ U_E, then M_B-orthonormalized

How it avoids duplicated work:
  - For each parameter mu:
      * set source once
      * build bases once (at rmax)
      * build ROM once and precompute reduced forcing once
      * then loop over r by slicing

Saves all arrays into one NPZ.

REQUIRES you already have these objects in your environment:
  sim, MaxwellPODROM
  width, dir
  M_E, M_B (optional; if you only have M_E you can still compute E mass errors, and use L2 for B)
  Mnorm(X, M)  (mass norm for vectors or time histories)
  params_train, params_test
  snapshots_E_train, snapshots_B_train   (N, nT, n_train)
  snapshots_E_test,  snapshots_B_test    (N, nT, n_test)
  r_arr

For mono:
  U_mono_E, U_mono_B

For rbf:
  rbfw_E, rbfw_B
  tucker_tensor_E, tucker_tensor_B
  buildParBasis(tucker_tensor, weights) -> (U, s, extra)

For lid:
  lidw(params_train, k, mu, eps=..., rcond=...) -> weights
  tucker_tensor_E, tucker_tensor_B
  buildParBasis

For enrichment (enrich_builder):
  Wk : weak curl operator (B-space -> E-space)
  St : strong curl operator (E-space -> B-space)
  m_orthonormalize_chol(V, M) -> V_orth

Also projection error helpers are included below:
  projection_error_M(X, U, M)  : mass-norm relative projection error
  projection_error_L2(X, U)    : L2 relative projection error
"""

import numpy as np
from src.Utils.utils import projection_error_M, m_orthonormalize_chol
from src.Maxwell.ROM import MaxwellPODROM as _MaxwellPODROM
from src.Maxwell.ROM import MaxwellHyperROM as _MaxwellHyperROM
from src.Maxwell.ROM import assemble_deim_entries as _assemble_deim_entries


# ----------------------------
# Core per-parameter runner
# ----------------------------

def run_rom_errors_for_one_mu(sim, Ue_max, Ub_max, r_arr,
                             E_truth, B_truth, M_E, M_B, Mnorm,
                             t0=0.0, tf=2.5, nsteps=120, record_freq=1,
                             hyper=False, qdeim_data=None):
    """
    ROM errors for one parameter mu, for all r in r_arr.
    E error in M_E norm; B error in L2 norm (matches your earlier usage).
    """
    r_arr = np.asarray(r_arr, dtype=int)
    rmax = int(np.max(r_arr))

    ROM = _make_rom(sim, Ue_max[:, :rmax], Ub_max[:, :rmax], hyper, qdeim_data)
    ROM.precompute_reduced_current(t0=t0, tf=tf, nsteps=nsteps)

    denomB = Mnorm(B_truth, M_B)
    denomE = Mnorm(E_truth, M_E)
    if denomB == 0.0:
        raise ValueError("||B_truth|| is zero.")
    if denomE == 0.0:
        raise ValueError("||E_truth||_M is zero.")

    Eerrs = np.zeros(len(r_arr))
    Berrs = np.zeros(len(r_arr))

    for r_idx, r in enumerate(r_arr):
        (Ehist, Bhist), _ = ROM.timeLoop(t0, tf, nsteps, record_freq=record_freq, r=int(r), use_forcing_cache=True)

        E_full = Ue_max[:, :r] @ Ehist
        B_full = Ub_max[:, :r] @ Bhist

        Berrs[r_idx] = Mnorm(B_truth - B_full, M_B) / denomB
        Eerrs[r_idx] = Mnorm(E_truth - E_full, M_E) / denomE

    return Eerrs, Berrs


def run_proj_errors_for_one_mu(Ue_max, Ub_max, r_arr,
                              E_truth, B_truth, M_E, M_B, Mnorm):
    """
    Projection errors for one parameter mu, for all r in r_arr.
    E projection error in M_E norm; B projection error in L2 norm.
    """
    r_arr = np.asarray(r_arr, dtype=int)
    rmax = int(np.max(r_arr))

    Eperr = np.zeros(len(r_arr))
    Bperr = np.zeros(len(r_arr))

    for r_idx, r in enumerate(r_arr):
        Ue_r = Ue_max[:, :int(r)]
        Ub_r = Ub_max[:, :int(r)]
        Eperr[r_idx] = projection_error_M(E_truth, Ue_r, M_E, Mnorm)
        Bperr[r_idx] = projection_error_M(B_truth, Ub_r, M_B, Mnorm)
        #projection_error_L2(B_truth, Ub_r)

    return Eperr, Bperr


# ----------------------------
# Basis builders
# ----------------------------

def build_mono_bases(U_mono_E, U_mono_B):
    def builder(mu, rmax):
        return U_mono_E[:, :rmax], U_mono_B[:, :rmax]
    return builder


def build_rbf_bases(rbfw_E, rbfw_B, tucker_tensor_E, tucker_tensor_B, buildParBasis):
    def builder(mu, rmax):
        W_E = rbfw_E.weights(mu)
        W_B = rbfw_B.weights(mu)
        U_E, _, _ = buildParBasis(tucker_tensor_E, W_E)
        U_B, _, _ = buildParBasis(tucker_tensor_B, W_B)
        return U_E[:, :rmax], U_B[:, :rmax]
    return builder


def build_lid_bases(params_train, lidw, tucker_tensor_E, tucker_tensor_B, buildParBasis,
                    k=15, eps=1e-14, rcond=1e-14):
    def builder(mu, rmax):
        W_E = lidw(params_train, k, mu, eps=eps, rcond=rcond)
        W_B = lidw(params_train, k, mu, eps=eps, rcond=rcond)
        U_E, _, _ = buildParBasis(tucker_tensor_E, W_E)
        U_B, _, _ = buildParBasis(tucker_tensor_B, W_B)
        return U_E[:, :rmax], U_B[:, :rmax]
    return builder


def enrich_builder(basis_builder, Wk, St, M_E, M_B):
    """
    Wraps any basis builder to add curl-based enrichment.

    For each (mu, r):
      E basis is augmented with weak curl of B:   hstack([U_E[:,:r], Wk @ U_B[:,:r]])
      B basis is augmented with strong curl of E:  hstack([U_B[:,:r], St @ U_E[:,:r]])
    then each is M-orthonormalized via Cholesky.

    Returns a two-level builder:
      mu_builder(mu, rmax) -> r_builder(r) -> (U_E_orth, U_B_orth)

    Use with sweep_dataset(..., enriched=True).
    """
    def mu_builder(mu, rmax):
        U_E_full, U_B_full = basis_builder(mu, rmax)

        def r_builder(r):
            U_E_aug = np.hstack([U_E_full[:, :r], Wk @ U_B_full[:, :r]])
            U_E_orth = m_orthonormalize_chol(U_E_aug, M_E)

            U_B_aug = np.hstack([U_B_full[:, :r], St @ U_E_full[:, :r]])
            U_B_orth = m_orthonormalize_chol(U_B_aug, M_B)

            r_use = min(U_E_orth.shape[1], U_B_orth.shape[1])
            return U_E_orth[:, :r_use], U_B_orth[:, :r_use]

        return r_builder

    return mu_builder


# ----------------------------
# Sweeps over a dataset (train or test)
# ----------------------------

def _make_rom(sim, U_E, U_B, hyper, qdeim_data):
    """Instantiate the appropriate ROM class."""
    if hyper:
        return _MaxwellHyperROM(sim, U_E, U_B, **qdeim_data)
    return _MaxwellPODROM(sim, U_E, U_B)


def sweep_dataset(sim, ROMClass, params, r_arr,
                  snapshots_E, snapshots_B,
                  width, direction, M_E, M_B, Mnorm,
                  basis_builder,
                  t0=0.0, tf=2.5, nsteps=120, record_freq=1,
                  label="", enriched=False,
                  hyper=False, qdeim_data=None):
    """
    Returns 4 arrays each shape (len(r_arr), n_cases):
      rom_E, rom_B, proj_E, proj_B

    If enriched=True, basis_builder should be the output of enrich_builder(),
    i.e. basis_builder(mu, rmax) returns r_builder(r) -> (U_E, U_B).
    In this mode, ROM and projection are rebuilt per-r since enriched bases
    at different r are not nested.

    hyper : bool
        If True, use MaxwellHyperROM with Q-DEIM hyperreduction.
    qdeim_data : dict, required when hyper=True
        Dict with keys: U_J, deim_idx, P_UJ_inv, coords_r,
        weighted_basis_r, gids_r.
    """
    r_arr = np.asarray(r_arr, dtype=int)
    rmax = int(np.max(r_arr))
    n_cases = params.shape[0]

    rom_E = np.zeros((len(r_arr), n_cases))
    rom_B = np.zeros((len(r_arr), n_cases))
    proj_E = np.zeros((len(r_arr), n_cases))
    proj_B = np.zeros((len(r_arr), n_cases))

    for idx in range(n_cases):
        mu = params[idx]
        mid = mu[:3]
        sim.set_source(mid, width, direction)

        E_truth = snapshots_E[..., idx]
        B_truth = snapshots_B[..., idx]

        if enriched:
            r_builder = basis_builder(mu, rmax)

            denomE = Mnorm(E_truth, M_E)
            denomB = Mnorm(B_truth, M_B)
            if denomE == 0.0:
                raise ValueError("||E_truth||_M is zero.")
            if denomB == 0.0:
                raise ValueError("||B_truth|| is zero.")

            # For hyper: assemble DEIM entries once per parameter
            j_deim = None
            if hyper:
                def source_spatial(x, _mid=mid):
                    dist2 = np.sum((x - _mid)**2, axis=-1)
                    return np.exp(-dist2 / (2.0 * sim.width))[..., None] * sim.dir

                j_deim = _assemble_deim_entries(
                    source_spatial,
                    qdeim_data['coords_r'], qdeim_data['weighted_basis_r'],
                    qdeim_data['gids_r'], qdeim_data['deim_idx'])

            for r_idx, r in enumerate(r_arr):
                Ue_r, Ub_r = r_builder(int(r))

                # ROM error
                ROM = _make_rom(sim, Ue_r, Ub_r, hyper, qdeim_data)
                if hyper:
                    ROM.precompute_reduced_current(t0, tf, nsteps, j_deim=j_deim)
                (Ehist, Bhist), _ = ROM.timeLoop(t0, tf, nsteps, record_freq=record_freq)

                E_full = Ue_r @ Ehist
                B_full = Ub_r @ Bhist

                rom_E[r_idx, idx] = Mnorm(E_truth - E_full, M_E) / denomE
                rom_B[r_idx, idx] = Mnorm(B_truth - B_full, M_B) / denomB

                # Projection error
                proj_E[r_idx, idx] = projection_error_M(E_truth, Ue_r, M_E, Mnorm)
                proj_B[r_idx, idx] = projection_error_M(B_truth, Ub_r, M_B, Mnorm)
        else:
            Ue_max, Ub_max = basis_builder(mu, rmax)

            # ROM errors
            Eerrs, Berrs = run_rom_errors_for_one_mu(
                sim, Ue_max, Ub_max, r_arr,
                E_truth, B_truth, M_E, M_B, Mnorm,
                t0=t0, tf=tf, nsteps=nsteps, record_freq=record_freq,
                hyper=hyper, qdeim_data=qdeim_data
            )
            rom_E[:, idx] = Eerrs
            rom_B[:, idx] = Berrs

            # Projection errors
            Eperr, Bperr = run_proj_errors_for_one_mu(
                Ue_max, Ub_max, r_arr,
                E_truth, B_truth, M_E, M_B, Mnorm
            )
            proj_E[:, idx] = Eperr
            proj_B[:, idx] = Bperr

        if label:
            print(f"{label}: {idx+1}/{n_cases}", end="\r")
        else:
            print(f"{idx+1}/{n_cases}", end="\r")

    print()
    return rom_E, rom_B, proj_E, proj_B
