
import sys
import os
import numpy as np
import scipy.linalg
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.Maxwell.FOM import assemble_current, decaying_gaussian_current, MaxwellSim
import h5py


filename = 'src/Maxwell/maxwell_data.hdf5'
file_hdf5 = h5py.File(filename, 'r')
savedir = "data/Maxwell"
os.makedirs(savedir, exist_ok=True)
print(f"Saving to: {savedir}")

# Q-DEIM rank (number of interpolation points)
p = 80

# Fixed source parameters (same as generate_data.py)
width = 0.1
dir = np.array([1, 1, 1])

# Load training parameters
with np.load(os.path.join(savedir, "params_train.npz")) as z:
    params_train = z["params"]

print(f"Training parameters: {params_train.shape[0]}")
print(f"Q-DEIM rank: {p}")

# -------------------------------------------------------
# 1) Assemble spatial current snapshots for all training params
# -------------------------------------------------------
cc_group = file_hdf5['Current_Construction']
quad_coords = np.array(cc_group['coords'])        # (n_cells, n_quad, 3)
weighted_basis = np.array(cc_group['weighted_basis'])  # (n_cells, n_quad, n_local, 3)
gids = np.array(cc_group['gids'])                  # (n_cells, n_local)

N_E = np.amax(gids) + 1
N_train = params_train.shape[0]

print(f"N_E (DOFs): {N_E}")
print(f"n_cells: {gids.shape[0]}, n_local_dofs: {gids.shape[1]}")

# Assemble the spatial-only current (smooth_pulse = 1) for each parameter
J_snap = np.zeros((N_E, N_train))

for i in range(N_train):
    mid = params_train[i, :3]

    # Spatial Gaussian source without time scaling
    def source_spatial(x, xmid=mid):
        dist2 = np.sum((x - xmid)**2, axis=-1)
        return np.exp(-dist2 / (2.0 * width))[..., None] * dir

    J_snap[:, i] = assemble_current(source_spatial, cc_group)
    if (i + 1) % 20 == 0 or i == 0:
        print(f"  assembled {i+1}/{N_train}")

print(f"J_snap shape: {J_snap.shape}")
print(f"J_snap norm range: [{np.linalg.norm(J_snap, axis=0).min():.4e}, "
      f"{np.linalg.norm(J_snap, axis=0).max():.4e}]")

# -------------------------------------------------------
# 2) Truncated SVD of current snapshots
# -------------------------------------------------------
print(f"\nComputing truncated SVD (rank={p})...")
U_J, S_J, Vt_J = np.linalg.svd(J_snap, full_matrices=False)
U_J = U_J[:, :p]
S_J = S_J[:p]
Vt_J = Vt_J[:p, :]

print(f"Singular values (first 10): {S_J[:10]}")
print(f"Singular values (last 5):   {S_J[-5:]}")
print(f"Relative energy in first {p} modes: "
      f"{np.sum(S_J**2) / np.sum(np.linalg.svd(J_snap, compute_uv=False)**2):.10f}")

# -------------------------------------------------------
# 3) Q-DEIM via pivoted QR
# -------------------------------------------------------
print(f"\nComputing Q-DEIM indices (p={p})...")
_, _, perm = scipy.linalg.qr(U_J.T, pivoting=True)
deim_idx = np.sort(perm[:p])  # sort for locality

P_UJ = U_J[deim_idx, :]  # (p, p) interpolation matrix
cond_P = np.linalg.cond(P_UJ)
print(f"DEIM indices (first 10): {deim_idx[:10]}")
print(f"Condition number of P^T U_J: {cond_P:.4e}")

# -------------------------------------------------------
# 4) Find reduced element set
# -------------------------------------------------------
print(f"\nFinding reduced element set...")
deim_set = set(deim_idx.tolist())
cell_mask = np.zeros(gids.shape[0], dtype=bool)
for c in range(gids.shape[0]):
    if deim_set.intersection(gids[c, :].tolist()):
        cell_mask[c] = True

n_reduced = cell_mask.sum()
print(f"Reduced cells: {n_reduced} / {gids.shape[0]} "
      f"({100*n_reduced/gids.shape[0]:.1f}%)")

coords_r = quad_coords[cell_mask]
weighted_basis_r = weighted_basis[cell_mask]
gids_r = gids[cell_mask]

# -------------------------------------------------------
# 5) Verify DEIM approximation on training data
# -------------------------------------------------------
print(f"\nVerifying DEIM approximation on training data...")
P_UJ_inv = np.linalg.inv(P_UJ)
recon_errors = np.zeros(N_train)
for i in range(N_train):
    j_true = J_snap[:, i]
    j_deim_vals = j_true[deim_idx]
    j_approx = U_J @ (P_UJ_inv @ j_deim_vals)
    recon_errors[i] = np.linalg.norm(j_true - j_approx) / np.linalg.norm(j_true)

print(f"DEIM reconstruction error (training):")
print(f"  mean:   {recon_errors.mean():.4e}")
print(f"  max:    {recon_errors.max():.4e}")
print(f"  median: {np.median(recon_errors):.4e}")

# -------------------------------------------------------
# 6) Save
# -------------------------------------------------------
outfile = os.path.join(savedir, "qdeim_current.npz")
print(f"\nSaving to {outfile}...")
np.savez(
    outfile,
    U_J=U_J,                    # (N_E, p) DEIM basis
    S_J=S_J,                    # (p,) singular values
    deim_idx=deim_idx,           # (p,) DOF indices
    P_UJ_inv=P_UJ_inv,          # (p, p) precomputed inverse
    coords_r=coords_r,          # reduced quad coords
    weighted_basis_r=weighted_basis_r,  # reduced weighted basis
    gids_r=gids_r,              # reduced global IDs
    cell_mask=cell_mask,         # which cells are selected
    p=np.int64(p),
    width=np.float64(width),
    dir=dir,
)

print("Done!")
