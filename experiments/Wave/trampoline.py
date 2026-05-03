"""Generate a 3D Wave q animation with FOM surface and ROM overlay.

Usage:
    python trampoline.py --method rbf --idx 0 --rank 40

Styles:
    --overlay wireframe
    --overlay balls

Loads data from ../../data/Wave by default.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tensorly.tucker_tensor import TuckerTensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.Bases.mo import mo
from src.Bases.rbf import RBFWeights
from src.Utils.animations import data_limits, select_frame_indices, symmetric_limits
from src.Utils.utils import buildParBasis, load_tucker_npz
from src.Wave.FOM import WaveFEM2D
from src.Wave.ROM import WavePODROM
from src.Wave.plots import eval_grid


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
    parser.add_argument("--rank", type=int, default=40, help="ROM rank.")
    parser.add_argument("--grid", type=int, default=60, help="Regular visualization grid size.")
    parser.add_argument("--overlay", choices=("wireframe", "balls"), default="wireframe")
    parser.add_argument("--balls", type=int, default=16, help="Number of ROM balls per coordinate direction.")
    parser.add_argument("--frames", type=int, default=120, help="Maximum number of animation frames.")
    parser.add_argument("--stride", type=int, default=1, help="Use every nth saved time step before frame capping.")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--color-error", action="store_true", help="Color ROM balls by signed error.")
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
    parser.add_argument("--elev", type=float, default=28.0)
    parser.add_argument("--azim", type=float, default=-55.0)
    return parser.parse_args()


def load_wave_data(data_dir, dataset):
    with np.load(data_dir / f"params_{dataset}.npz") as z:
        params = z["params"]
    with np.load(data_dir / f"wave_{dataset}.npz") as z:
        snapshots_q = z["Q"]
        times = z["times"]
    return params, snapshots_q, times


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
    return np.asarray([eval_grid(fom, field[:, idx], N=grid) for idx in frame_indices])


def make_writer(out_path, fps):
    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        return animation.PillowWriter(fps=fps)
    if suffix == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("Matplotlib cannot find ffmpeg. Use a .gif output or install ffmpeg for .mp4.")
        return animation.FFMpegWriter(fps=fps)
    raise ValueError("out_path must end in .mp4 or .gif.")


def save_trampoline_animation(
    Z_fom,
    Z_rom,
    times,
    out_path,
    *,
    L,
    overlay="wireframe",
    balls=16,
    fps=20,
    dpi=140,
    elev=28.0,
    azim=-55.0,
    rom_label="ROM",
    color_error=False,
):
    Z_fom = np.asarray(Z_fom)
    Z_rom = np.asarray(Z_rom)
    times = np.asarray(times)
    if Z_fom.shape != Z_rom.shape:
        raise ValueError("Z_fom and Z_rom must have matching shapes.")

    nframes, ny, nx = Z_fom.shape
    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, L, ny)
    X, Y = np.meshgrid(x, y)

    if overlay == "balls":
        ball_idx = np.linspace(0, nx - 1, int(balls), dtype=int)
        ball_idy = np.linspace(0, ny - 1, int(balls), dtype=int)
        BX, BY = np.meshgrid(x[ball_idx], y[ball_idy])
        BX_flat = BX.ravel()
        BY_flat = BY.ravel()

    zmin, zmax = data_limits(Z_fom)
    zpad = 0.08 * (zmax - zmin if zmax > zmin else 1.0)
    err_vmin, err_vmax = symmetric_limits(Z_fom - Z_rom)

    fig = plt.figure(figsize=(7.5, 6.5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    def draw(frame):
        ax.clear()
        ax.plot_surface(
            X,
            Y,
            Z_fom[frame],
            cmap="viridis",
            vmin=zmin,
            vmax=zmax,
            linewidth=0,
            antialiased=True,
            alpha=0.72,
        )

        if overlay == "wireframe":
            ax.plot_wireframe(
                X,
                Y,
                Z_rom[frame],
                rstride=max(1, ny // 28),
                cstride=max(1, nx // 28),
                color="black",
                linewidth=0.55,
                alpha=0.9,
            )
        else:
            Zb = Z_rom[frame][np.ix_(ball_idy, ball_idx)].ravel()
            err = (Z_fom[frame] - Z_rom[frame])[np.ix_(ball_idy, ball_idx)].ravel()
            scatter_kwargs = {
                "s": 30,
                "depthshade": False,
                "edgecolors": "k",
                "linewidths": 0.25,
            }
            if color_error:
                scatter_kwargs.update({"c": err, "cmap": "RdBu_r", "vmin": err_vmin, "vmax": err_vmax})
            else:
                scatter_kwargs.update({"color": "black"})
            ax.scatter(BX_flat, BY_flat, Zb, **scatter_kwargs)

        ax.set_xlim(0.0, L)
        ax.set_ylim(0.0, L)
        ax.set_zlim(zmin - zpad, zmax + zpad)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_zlabel(r"$q$")
        ax.set_title(f"t = {times[frame]:.4f}")
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 0.45))
        ax.text2D(0.02, 0.95, f"surface: FOM\n{overlay}: {rom_label}", transform=ax.transAxes)
        return []

    anim = animation.FuncAnimation(fig, draw, frames=nframes, interval=1000 / fps, blit=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=make_writer(out_path, fps), dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    args.data_dir = args.data_dir.resolve()

    params, snapshots_q, times = load_wave_data(args.data_dir, args.dataset)
    if not (0 <= args.idx < params.shape[0]):
        raise IndexError(f"--idx must be in [0, {params.shape[0] - 1}], got {args.idx}.")

    mu = params[args.idx]
    Q_fom = snapshots_q[args.idx]

    fom = WaveFEM2D(L=2 * np.pi, h=0.5, orderW=2, orderV=2, width=0.5 / np.sqrt(2))
    params_train, _, _ = load_wave_data(args.data_dir, "train")
    U = build_wave_basis(args, mu, params_train)

    rom = WavePODROM(fom, U)
    Qr, _, Ur = rom.solve(times, mu[0], mu[1], mu[2], r=args.rank)
    Q_rom = Ur @ Qr

    frame_indices = select_frame_indices(len(times), max_frames=args.frames, stride=args.stride)
    Z_fom = evaluate_frames(fom, Q_fom, frame_indices, args.grid)
    Z_rom = evaluate_frames(fom, Q_rom, frame_indices, args.grid)

    if args.out is None:
        args.out = args.data_dir / f"wave_trampoline_{args.dataset}_{args.idx}_{args.method}_r{args.rank}.gif"

    save_trampoline_animation(
        Z_fom,
        Z_rom,
        times[frame_indices],
        args.out,
        L=fom.L,
        overlay=args.overlay,
        balls=args.balls,
        fps=args.fps,
        dpi=args.dpi,
        elev=args.elev,
        azim=args.azim,
        rom_label=METHOD_LABELS[args.method],
        color_error=args.color_error,
    )
    print(f"Saved animation to {args.out}")


if __name__ == "__main__":
    main()
