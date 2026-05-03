"""Generate Heat FOM/ROM comparison animations.

Usage:
    python animation.py --method rbf --idx 0 --rank 20

Loads data from ../../data/Heat by default.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tensorly.tucker_tensor import TuckerTensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.Bases.mo import mo
from src.Bases.rbf import RBFWeights
from src.Heat.FOM import HeatFEM2D
from src.Heat.ROM import HeatPODROM
from src.Heat.plots import eval_grid
from src.Utils.animations import save_scalar_comparison_animation, select_frame_indices
from src.Utils.utils import buildParBasis, load_tucker_npz


METHOD_LABELS = {
    "mono": "Monolithic",
    "rbf": "RBF",
    "mo": "MO",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("mono", "rbf", "mo"), default="rbf")
    parser.add_argument("--dataset", choices=("train", "test"), default="test")
    parser.add_argument("--idx", type=int, default=0, help="Parameter index in the selected dataset.")
    parser.add_argument("--rank", type=int, default=30, help="ROM rank.")
    parser.add_argument("--grid", type=int, default=80, help="Regular visualization grid size.")
    parser.add_argument("--frames", type=int, default=120, help="Maximum number of animation frames.")
    parser.add_argument("--stride", type=int, default=1, help="Use every nth saved time step before frame capping.")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--levels", type=int, default=20, help="Number of filled contour levels.")
    parser.add_argument("--no-contours", action="store_true", help="Hide contour line overlays.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "Heat")
    parser.add_argument("--tucker", type=Path, default=None)
    parser.add_argument("--svd", type=Path, default=None)
    parser.add_argument("--rbf-eps", type=float, default=1.0)
    parser.add_argument("--rbf-basis", default="gaussian")
    parser.add_argument("--rbf-order", type=int, default=-1)
    parser.add_argument("--rbf-nugget", type=float, default=0.0)
    parser.add_argument("--mo-k", type=int, default=15)
    parser.add_argument("--mo-eps", type=float, default=1e-16)
    parser.add_argument("--mo-rcond", type=float, default=1e-16)
    return parser.parse_args()


def load_heat_data(data_dir, dataset):
    params_path = data_dir / f"params_{dataset}.npz"
    solution_path = data_dir / f"heat_{dataset}.npz"

    with np.load(params_path) as z:
        params = z["params"]
    with np.load(solution_path) as z:
        snapshots = z["Q"]
        times = z["times"]

    return params, snapshots, times


def load_training_params(data_dir):
    with np.load(data_dir / "params_train.npz") as z:
        return z["params"]


def build_heat_basis(args, fom, mu, params_train):
    r = int(args.rank)
    if args.method == "mono":
        svd_path = args.svd or (args.data_dir / "svd_rank120_Mortho.npz")
        with np.load(svd_path) as z:
            U = z["U"]
        return U[:, :r]

    tucker_path = args.tucker or (args.data_dir / "tucker_120x120x120_Mortho.npz")
    core, factors = load_tucker_npz(tucker_path)
    tucker_tensor = TuckerTensor((core, factors))

    if args.method == "rbf":
        rbfw = RBFWeights(
            mus=params_train,
            basis=args.rbf_basis,
            eps=args.rbf_eps,
            order=args.rbf_order,
            nugget=args.rbf_nugget,
        )
        weights = rbfw.weights(mu)
    else:
        weights = mo(
            params_train,
            args.mo_k,
            mu,
            eps=args.mo_eps,
            rcond=args.mo_rcond,
        )

    U, _, _ = buildParBasis(tucker_tensor, weights)
    if r > U.shape[1]:
        raise ValueError(f"Requested rank {r}, but basis only has {U.shape[1]} columns.")
    return U[:, :r]


def evaluate_frames(fom, Q, frame_indices, grid):
    fields = []
    for idx in frame_indices:
        fields.append(eval_grid(fom, Q[:, idx], N=grid))
    return np.asarray(fields)


def main():
    args = parse_args()
    args.data_dir = args.data_dir.resolve()

    params, snapshots, times = load_heat_data(args.data_dir, args.dataset)
    if not (0 <= args.idx < params.shape[0]):
        raise IndexError(f"--idx must be in [0, {params.shape[0] - 1}], got {args.idx}.")

    mu = params[args.idx]
    Q_fom = snapshots[args.idx]

    fom = HeatFEM2D(L=2 * np.pi, h=0.2, order=1)
    params_train = load_training_params(args.data_dir)
    U = build_heat_basis(args, fom, mu, params_train)

    rom = HeatPODROM(fom, U)
    Qr, Ur = rom.solve(mu[0], mu[1], mu[2], times, r=args.rank)
    Q_rom = Ur @ Qr

    frame_indices = select_frame_indices(
        len(times),
        max_frames=args.frames,
        stride=args.stride,
    )
    Z_fom = evaluate_frames(fom, Q_fom, frame_indices, args.grid)
    Z_rom = evaluate_frames(fom, Q_rom, frame_indices, args.grid)

    if args.out is None:
        args.out = args.data_dir / f"heat_{args.dataset}_{args.idx}_{args.method}_r{args.rank}.gif"

    save_scalar_comparison_animation(
        Z_fom,
        Z_rom,
        times[frame_indices],
        args.out,
        extent=(0.0, fom.L, 0.0, fom.L),
        fps=args.fps,
        dpi=args.dpi,
        title=f"Heat {args.dataset}[{args.idx}]",
        rom_label=METHOD_LABELS[args.method],
        nlevels=args.levels,
        show_contours=not args.no_contours,
    )
    print(f"Saved animation to {args.out}")


if __name__ == "__main__":
    main()
