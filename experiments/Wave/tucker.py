import os
import sys
import numpy as np
from pathlib import Path
from tensorly.decomposition import Tucker

ROOT = Path(__file__).resolve().parents[2]  
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.Utils.utils import save_tucker_npz
from src.Wave.FOM import WaveFEM2D

savedir = "data/Wave"
os.makedirs(savedir, exist_ok=True)
print(f"Saving to: {savedir}")

print("Loading data")
# Load data
with np.load(os.path.join(savedir,"wave_train.npz")) as z:
    Q = z["Q"]
    P = z["P"]

X = np.concatenate((Q, P), axis=2).transpose(1,2,0)
print(f"{X.shape=}")


fom = WaveFEM2D(L=2 * np.pi, h=0.5, orderW=2, orderV=2, width=.5)
M = fom.MW.toarray()
R = np.linalg.cholesky(M,upper=True)
Rinv = np.linalg.inv(R)

for i in range(X.shape[-1]):
    X[...,i] = R@X[...,i] 

# Tucker decomposition
print("Tucker")
decomp = Tucker(rank=[60,60,60], init="svd", verbose=True, n_iter_max=5)
tucker_tensor = decomp.fit_transform(X)
tucker_tensor.factors[0] = Rinv @ tucker_tensor.factors[0]
save_tucker_npz(os.path.join(savedir,"tucker_60x60x60_Mortho.npz"), tucker_tensor.core, tucker_tensor.factors)

# Tucker decomposition
print("Tucker")
decomp = Tucker(rank=[120,120,120], init="svd", verbose=True, n_iter_max=5)
tucker_tensor = decomp.fit_transform(X)
tucker_tensor.factors[0] = Rinv @ tucker_tensor.factors[0]
save_tucker_npz(os.path.join(savedir,"tucker_120x120x120_Mortho.npz"), tucker_tensor.core, tucker_tensor.factors)

print("SVD")
# SVD
A = np.hstack(X.transpose(2,0,1))
U, S, Vt = np.linalg.svd(A, full_matrices=False)
U = Rinv @ U[:,:120]
print('Saving SVD')
np.savez_compressed(os.path.join(savedir,"svd_rank120_Mortho.npz"), U=U, S=S, Vt=Vt)