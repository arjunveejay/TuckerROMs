import sys
import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.Heat.FOM import  HeatFEM2D
from src.Utils.utils import sample_parameters

savedir = "data/Heat"
os.makedirs(savedir, exist_ok=True)
print(f"Saving to: {savedir}")

Ns = 200
train_ratio = 0.8
params_train, params_test = sample_parameters([0, 0, 0], [1, 2*np.pi, 2*np.pi], Ns, train_ratio=train_ratio, randseed=0)

np.savez(os.path.join(savedir, "params_train.npz"), params=params_train)
np.savez(os.path.join(savedir, "params_test.npz"), params=params_test)

print("Total sampled parameters: ", Ns)
print("Training parameters: ", int(train_ratio*Ns))

print("Single sample run: ")
t = np.linspace(0, np.pi, 1201)
fom = HeatFEM2D(L=2 * np.pi, h=0.2, order=1)
print(fom)
Q = fom.solve(1, np.pi, np.pi, t)

print("Run for all parameters:")
train_Q = fom.solve_multi(params_train, t)
test_Q = fom.solve_multi(params_test, t)

np.savez(os.path.join(savedir,"heat_train.npz"), Q=train_Q, times=t)
np.savez(os.path.join(savedir,"heat_test.npz"), Q=test_Q, times=t)