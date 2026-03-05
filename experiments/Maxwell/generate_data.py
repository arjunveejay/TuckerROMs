
import sys
import os
import gc
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[2]  
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.Maxwell.FOM import *
from src.Utils.utils import sample_parameters


filename = 'src/Maxwell/maxwell_data.hdf5'
file_hdf5 = h5py.File(filename, 'r')
savedir = "data/Maxwell"
os.makedirs(savedir, exist_ok=True)
print(f"Saving to: {savedir}")


Ns = 200
Ntrain = int(0.8*Ns)
params, _ = sample_parameters([0.5, 0.5, 0.5], [1.5, 1.5, 1.5], Ns, train_ratio=1, randseed=0)
np.savez(os.path.join(savedir, "params_train.npz"), params=params[:Ntrain])
np.savez(os.path.join(savedir, "params_test.npz"), params=params[Ntrain:])
print("Total sampled parameters: ", Ns)
print("Training parameters: ", Ntrain)


print("Single sample run: ")
# Center of the gaussian pulse
mid = params[0,:3]
# Width
width = 0.1
# Direction
dir = np.array([1,1,1])
sim = MaxwellSim(file_hdf5)
print("mid: ", mid, " dir: ", dir)
sim.set_source(mid, width, dir)
states,times = sim.timeLoop(t0=0.0,tf=2.5,nsteps=120,record_freq=1)

print("Run for all parameters:")

snapshots_E = np.zeros((states[0][0].shape[0], len(times), params.shape[0]))
snapshots_B = np.zeros((states[0][1].shape[0], len(times), params.shape[0]))

for i in range(params.shape[0]):
    mid = params[i,:3]
    dir = np.array([1,1,1])
    print(i, mid, dir)
    sim.set_source(mid, width, dir)
    states,timesteps = sim.timeLoop(t0=0.0,tf=2.5,nsteps=120,record_freq=1)
    E = np.array([states[i][0] for i in range(len(states))]).T
    B = np.array([states[i][1] for i in range(len(states))]).T
    snapshots_E[...,i] = E
    snapshots_B[...,i] = B

print("Saving files:")

# -------------------------
# float64 NPZ (keep t, params)
# -------------------------
np.savez(os.path.join(savedir, "maxwell_E_train.npz"),
         E=snapshots_E[..., :Ntrain],
         t=times,
         params=params[:Ntrain])

np.savez(os.path.join(savedir, "maxwell_E_test.npz"),
         E=snapshots_E[..., Ntrain:],
         t=times,
         params=params[Ntrain:])

np.savez(os.path.join(savedir, "maxwell_B_train.npz"),
         B=snapshots_B[..., :Ntrain],
         t=times,
         params=params[:Ntrain])

np.savez(os.path.join(savedir, "maxwell_B_test.npz"),
         B=snapshots_B[..., Ntrain:],
         t=times,
         params=params[Ntrain:])

# -------------------------
# float32 NPY (ONLY arrays)
# Save B first, then free it
# -------------------------
np.save(os.path.join(savedir, "maxwell_B_train_f32.npy"),
        snapshots_B[..., :Ntrain].astype(np.float32, copy=False))

np.save(os.path.join(savedir, "maxwell_B_test_f32.npy"),
        snapshots_B[..., Ntrain:].astype(np.float32, copy=False))

del snapshots_B
gc.collect()

# -------------------------
# float32 NPY (ONLY arrays)
# Save E, then free it
# -------------------------
np.save(os.path.join(savedir, "maxwell_E_train_f32.npy"),
        snapshots_E[..., :Ntrain].astype(np.float32, copy=False))

np.save(os.path.join(savedir, "maxwell_E_test_f32.npy"),
        snapshots_E[..., Ntrain:].astype(np.float32, copy=False))

del snapshots_E
gc.collect()

print("Saving files done!")