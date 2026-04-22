import sys
import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.Wave.FOM import WaveFEM2D
from src.Utils.utils import sample_parameters

savedir = "data/Wave"
os.makedirs(savedir, exist_ok=True)
print(f"Saving to: {savedir}")

Ns = 200
train_ratio = 0.8
params_train, params_test = sample_parameters([0.01, 0, 0], [0.05, 2*np.pi, 2*np.pi], Ns, train_ratio=train_ratio, randseed=0)

np.savez(os.path.join(savedir, "params_train.npz"), params=params_train)
np.savez(os.path.join(savedir, "params_test.npz"), params=params_test)

print("Total sampled parameters: ", Ns)
print("Training parameters: ", int(train_ratio*Ns))

print("Single sample run: ")
t = np.linspace(0, 8*np.pi, 501)
fom = WaveFEM2D(L=2 * np.pi, h=0.5, orderW=2, orderV=2, width=.5/np.sqrt(2))
print(fom)
Q,P = fom.solve(t, 1, 0.8*np.pi, np.pi)

print("Run for all parameters:")

train_Q, train_P = fom.solve_multi(params_train, t)
test_Q, test_P     = fom.solve_multi(params_test, t)
np.savez(os.path.join(savedir,"wave_train.npz"), Q=train_Q, P=train_P, times=t)
np.savez(os.path.join(savedir,"wave_test.npz"), Q=test_Q, P=test_P, times=t)

print("Saving files done!")