import gc
import os
import sys
import h5py
import importlib
import numpy as np
import tensorly as tl
from pathlib import Path
from scipy.sparse.linalg import svds
import tensorly.decomposition as dec
from sksparse.cholmod import cholesky

from tensorly.decomposition import Tucker
from tensorly.tenalg.svd import svd_interface as _orig_svd_interface

ROOT = Path(__file__).resolve().parents[2]  
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.Utils.utils import save_tucker_npz
from src.Maxwell.FOM import read_hdf5_sparse

savedir = "data/Maxwell"
os.makedirs(savedir, exist_ok=True)
print(f"Saving to: {savedir}")

# Function for sparse SVD
def svd(A, k=120):
    # svds returns singular values in ascending order
    U, S, Vt = svds(A, k=k)
    idx = np.argsort(S)[::-1]
    return U[:, idx], S[idx], Vt[idx, :]


# Load mass matrix for E
filename = "src/Maxwell/maxwell_data.hdf5"
file_hdf5 = h5py.File(filename, "r")
emass = read_hdf5_sparse(file_hdf5['Emass'])

# Cholesky factorization for M-orthonormality
F = cholesky(emass.tocsc())                 # allow permutation (keeps fill small)
L, Dsp = F.L_D()                    # LDL-consistent L
d = np.sqrt(F.D())                  # 1D diag entries of D  :contentReference[oaicite:1]{index=1}

def R_mul(X):                       # X: (n,k)
    return d[:,None] * (L.T @ F.apply_P(X))          # R X, where R = sqrt(D) L^T P

def Rinv_mul(X):                    # X: (n,k)
    return F.apply_Pt(F.solve_Lt(X / d[:,None]))     # R^{-1} X = P^T (L^T)^{-1} sqrt(D)^{-1} X


# ----------------- Monkey patch tucker's svd_interface -------------------------------
# Find the real implementation module of dec.tucker
impl_mod = importlib.import_module(dec.tucker.__module__)
print("tucker is implemented in:", impl_mod.__name__)

def svd_interface_randomized(matrix, n_eigenvecs=None, **kwargs):
    # Make sure we see it even with tqdm/verbose output
    print(f"[patched] HOOI SVD: shape={matrix.shape}, k={n_eigenvecs}",
          file=sys.stderr, flush=True)

    kwargs.pop("method", None)
    return _orig_svd_interface(
        matrix,
        n_eigenvecs=n_eigenvecs,
        method="randomized_svd",
        **kwargs
    )

# Patch *that* module's svd_interface
impl_mod.svd_interface = svd_interface_randomized

# Sanity check: show what it is now
print("patched svd_interface =", impl_mod.svd_interface)

# ------------------------------------------------------------------------

# ----------------- E ---------------------------------------------------
print("Loading E")
# Load data
E = np.load(os.path.join(savedir,"maxwell_E_train_f32.npy"), mmap_mode="r")
n, T, M = E.shape
E = (R_mul(E.reshape(n, T*M))).reshape(n, T, M)  

print("Tucker")
# Tucker decomposition
decomp = Tucker(rank=[150,120,150], init="random", verbose=True, n_iter_max=5)
tucker_tensor = decomp.fit_transform(E)
tucker_tensor.factors[0] = Rinv_mul(tucker_tensor.factors[0])
save_tucker_npz(os.path.join(savedir,"tucker_E_150x120x150_Mortho.npz"), tucker_tensor.core, tucker_tensor.factors)

print("SVD")
# SVD
A = np.hstack(E.transpose(2,0,1))
U, S, Vt = svd(A, k=120)
U = Rinv_mul(U)
np.savez_compressed(os.path.join(savedir,"E_svd_rank120_Mortho.npz"), U=U, S=S, Vt=Vt)
del E
gc.collect()


# -------------------------- B -------------------------------------------
print("Loading B")
B = np.load(os.path.join(savedir,"maxwell_B_train_f32.npy"), mmap_mode="r")

print("Tucker")
# Tucker decomposition
decomp = Tucker(rank=[150,120,150], init="random", verbose=True, n_iter_max=5)
tucker_tensor = decomp.fit_transform(B)
save_tucker_npz(os.path.join(savedir,"tucker_B_150x120x150.npz"), tucker_tensor.core, tucker_tensor.factors)

print("SVD")
# SVD
A = np.hstack(B.transpose(2,0,1))
U, S, Vt = svd(A, k=120)
np.savez_compressed(os.path.join(savedir,"B_svd_rank120.npz"), U=U, S=S, Vt=Vt)
del B
gc.collect()