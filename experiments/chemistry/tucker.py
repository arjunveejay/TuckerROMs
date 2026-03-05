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

savedir = "data/Chemistry"
os.makedirs(savedir, exist_ok=True)
print(f"Saving to: {savedir}")

snaps = np.load(savedir+'/chemical_traj_uva_2024_600.npy').transpose((1,2,0))
#snaps[snaps==0] = 1e-16
print(snaps.shape)
snaps_flat = snaps.reshape(-1, np.prod(snaps.shape[1:]))
print(snaps_flat.shape)
mu = np.mean(snaps_flat, axis=1, keepdims=1)
std = np.std(snaps_flat, axis=1, keepdims=1)
std[std == 0 ] = 1
snaps0 = (snaps.transpose((1,0,2)) - mu) / std
snaps0 = snaps0.transpose((1,0,2))
print(snaps0.shape)

print("Tucker")
# Tucker decomposition
# decomp = Tucker(rank=[50,50,50], init="svd", verbose=True, n_iter_max=5)
# tucker_tensor = decomp.fit_transform(snaps)
# save_tucker_npz(os.path.join(savedir,"tucker_50x50x50.npz"), tucker_tensor.core, tucker_tensor.factors)


# decomp = Tucker(rank=[100,100,100], init="svd", verbose=True, n_iter_max=5)
# tucker_tensor = decomp.fit_transform(snaps)
# save_tucker_npz(os.path.join(savedir,"tucker_100x100x100.npz"), tucker_tensor.core, tucker_tensor.factors)


decomp = Tucker(rank=[100,200,200], init="svd", verbose=True, n_iter_max=5)
tucker_tensor = decomp.fit_transform(snaps0)
save_tucker_npz(os.path.join(savedir,"tucker_100x200x200.npz"), tucker_tensor.core, tucker_tensor.factors)
