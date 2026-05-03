"""Generate Wave FOM/ROM comparison animations.

Usage:
    python animation.py --method rbf --field q --idx 0 --rank 20 --no-contours --levels 40

Loads data from ../../data/Wave by default.
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
from src.Utils.animations import save_scalar_comparison_animation, select_frame_indices
from src.Utils.utils import buildParBasis, load_tucker_npz
from src.Wave.FOM import WaveFEM2D
from src.Wave.ROM import WavePODROM
from src.Wave.plots import eval_grid


METHOD_LABELS = {
    "mono": "Monolithic",
    "rbf": "RBF",
    "mo": "MO",
}

FIELD_LABELS = {
    "q": "q",
    "p": "p",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("mono", "rbf", "mo"), default="rbf")
    parser.add_argument("--field", choices=("q", "p"), default="q")
    parser.add_argument("--dataset", choices=("train", "test"), default="test")
    parser.add_argument("--idx", type=int, default=0, help="Parameter index in the selected dataset.")
    parser.add_argument("--rank", type=int, default=40, help="ROM rank.")
    parser.add_argument("--grid", type=int, default=80, help="Regular visualization grid size.")
    parser.add_argument("--frames", type=int, default=120, help="Maximum number of animation frames.")
    parser.add_argument("--stride", type=int, default=1, help="Use every nth saved time step before frame capping.")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--levels", type=int, default=20, help="Number of filled contour levels.")
    parser.add_argument("--no-contours", action="store_true", help="Hide contour line overlays.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "Wave")
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


def load_wave_data(data_dir, dataset):
    params_path = data_dir / f"params_{dataset}.npz"
    solution_path = data_dir / f"wave_{dataset}.npz"

    with np.load(params_path) as z:
        params = z["params"]
    with np.load(solution_path) as z:
        snapshots_q = z["Q"]
        snapshots_p = z["P"]
        times = z["times"]

    return params, snapshots_q, snapshots_p, times


def load_training_params(data_dir):
    with np.load(data_dir / "params_train.npz") as z:
        return z["params"]


def build_wave_basis(args, mu, params_train):
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


def evaluate_frames(fom, field, frame_indices, grid):
    fields = []
    for idx in frame_indices:
        fields.append(eval_grid(fom, field[:, idx], N=grid))
    return np.asarray(fields)


def main():
    args = parse_args()
    args.data_dir = args.data_dir.resolve()

    params, snapshots_q, snapshots_p, times = load_wave_data(args.data_dir, args.dataset)
    if not (0 <= args.idx < params.shape[0]):
        raise IndexError(f"--idx must be in [0, {params.shape[0] - 1}], got {args.idx}.")

    mu = params[args.idx]
    truth = snapshots_q[args.idx] if args.field == "q" else snapshots_p[args.idx]

    fom = WaveFEM2D(L=2 * np.pi, h=0.5, orderW=2, orderV=2, width=0.5 / np.sqrt(2))
    params_train = load_training_params(args.data_dir)
    U = build_wave_basis(args, mu, params_train)

    rom = WavePODROM(fom, U)
    Qr, Pr, Ur = rom.solve(times, mu[0], mu[1], mu[2], r=args.rank)
    rom_field = Ur @ Qr if args.field == "q" else Ur @ Pr

    frame_indices = select_frame_indices(
        len(times),
        max_frames=args.frames,
        stride=args.stride,
    )
    Z_fom = evaluate_frames(fom, truth, frame_indices, args.grid)
    Z_rom = evaluate_frames(fom, rom_field, frame_indices, args.grid)

    if args.out is None:
        args.out = args.data_dir / f"wave_{args.field}_{args.dataset}_{args.idx}_{args.method}_r{args.rank}.gif"

    save_scalar_comparison_animation(
        Z_fom,
        Z_rom,
        times[frame_indices],
        args.out,
        extent=(0.0, fom.L, 0.0, fom.L),
        fps=args.fps,
        dpi=args.dpi,
        title=f"Wave {FIELD_LABELS[args.field]} {args.dataset}[{args.idx}]",
        rom_label=METHOD_LABELS[args.method],
        nlevels=args.levels,
        show_contours=not args.no_contours,
    )
    print(f"Saved animation to {args.out}")


if __name__ == "__main__":
    main()
