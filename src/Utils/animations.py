"""Shared animation helpers for FOM/ROM visual comparisons."""

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def select_frame_indices(n_frames_total, max_frames=None, stride=1):
    """Return representative frame indices for a trajectory."""
    n_frames_total = int(n_frames_total)
    if n_frames_total < 1:
        raise ValueError("n_frames_total must be positive.")

    stride = int(stride)
    if stride < 1:
        raise ValueError("stride must be positive.")

    indices = np.arange(0, n_frames_total, stride, dtype=int)
    if indices[-1] != n_frames_total - 1:
        indices = np.append(indices, n_frames_total - 1)

    if max_frames is not None:
        max_frames = int(max_frames)
        if max_frames < 1:
            raise ValueError("max_frames must be positive.")
        if indices.size > max_frames:
            indices = np.linspace(0, n_frames_total - 1, max_frames, dtype=int)
            indices = np.unique(indices)

    return indices


def symmetric_limits(values):
    """Return symmetric color limits around zero for an array."""
    values = np.asarray(values)
    vmax = float(np.nanmax(np.abs(values)))
    if vmax == 0.0 or not np.isfinite(vmax):
        vmax = 1.0
    return -vmax, vmax


def data_limits(*arrays):
    """Return finite min/max limits across one or more arrays."""
    vals = []
    for arr in arrays:
        arr = np.asarray(arr)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            vals.append(finite)
    if not vals:
        return 0.0, 1.0

    merged = np.concatenate(vals)
    vmin = float(merged.min())
    vmax = float(merged.max())
    if vmin == vmax:
        pad = 1.0 if vmin == 0.0 else abs(vmin) * 0.05
        return vmin - pad, vmax + pad
    return vmin, vmax


def save_scalar_comparison_animation(
    Z_fom,
    Z_rom,
    times,
    out_path,
    *,
    extent=None,
    fps=20,
    dpi=150,
    title="Heat",
    rom_label="ROM",
    cmap="viridis",
    error_cmap="RdBu_r",
    nlevels=20,
    show_contours=True,
):
    """Save an animation with FOM, ROM, and signed-error panels.

    Parameters
    ----------
    Z_fom, Z_rom : ndarray, shape (n_frames, ny, nx)
        Scalar fields to animate.
    times : ndarray, shape (n_frames,)
        Time values corresponding to frames.
    out_path : str or Path
        Output path ending in .mp4 or .gif.
    extent : tuple, optional
        Matplotlib image extent, e.g. ``(xmin, xmax, ymin, ymax)``.
    """
    Z_fom = np.asarray(Z_fom)
    Z_rom = np.asarray(Z_rom)
    times = np.asarray(times)

    if Z_fom.shape != Z_rom.shape:
        raise ValueError(f"Z_fom and Z_rom must have the same shape, got {Z_fom.shape} and {Z_rom.shape}.")
    if Z_fom.ndim != 3:
        raise ValueError("Z_fom and Z_rom must have shape (n_frames, ny, nx).")
    if times.shape[0] != Z_fom.shape[0]:
        raise ValueError("times length must match the number of frames.")

    Z_err = Z_fom - Z_rom
    sol_vmin, sol_vmax = data_limits(Z_fom, Z_rom)
    err_vmin, err_vmax = symmetric_limits(Z_err)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    if extent is None:
        x = np.arange(Z_fom.shape[2])
        y = np.arange(Z_fom.shape[1])
    else:
        x = np.linspace(extent[0], extent[1], Z_fom.shape[2])
        y = np.linspace(extent[2], extent[3], Z_fom.shape[1])
    X, Y = np.meshgrid(x, y)

    sol_levels = np.linspace(sol_vmin, sol_vmax, nlevels + 1)
    err_levels = np.linspace(err_vmin, err_vmax, nlevels + 1)

    panel_data = [
        (Z_fom[0], "FOM", cmap, sol_levels),
        (Z_rom[0], rom_label, cmap, sol_levels),
        (Z_err[0], "Signed error", error_cmap, err_levels),
    ]

    contours = []
    lines = []
    for ax, (Z, label, panel_cmap, levels) in zip(axes, panel_data):
        Z = np.clip(Z, levels[0], levels[-1])
        cf = ax.contourf(X, Y, Z, levels=levels, cmap=panel_cmap)
        cs = None
        if show_contours:
            cs = ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.35, alpha=0.65)
        ax.set_title(label)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_aspect("equal")
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        contours.append(cf)
        lines.append(cs)

    suptitle = fig.suptitle("")

    def remove_contour_set(contour_set):
        if hasattr(contour_set, "remove"):
            contour_set.remove()
            return
        for coll in contour_set.collections:
            coll.remove()

    def update(frame):
        for cf, cs in zip(contours, lines):
            remove_contour_set(cf)
            if cs is not None:
                remove_contour_set(cs)

        frame_data = [
            (Z_fom[frame], cmap, sol_levels),
            (Z_rom[frame], cmap, sol_levels),
            (Z_err[frame], error_cmap, err_levels),
        ]
        for i, (Z, panel_cmap, levels) in enumerate(frame_data):
            Z = np.clip(Z, levels[0], levels[-1])
            contours[i] = axes[i].contourf(X, Y, Z, levels=levels, cmap=panel_cmap)
            lines[i] = None
            if show_contours:
                lines[i] = axes[i].contour(
                    X, Y, Z, levels=levels, colors="k", linewidths=0.35, alpha=0.65
                )

        suptitle.set_text(f"t = {times[frame]:.4f}")
        return [suptitle]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=Z_fom.shape[0],
        interval=1000 / fps,
        blit=False,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    elif suffix == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("Matplotlib cannot find ffmpeg. Use a .gif output or install ffmpeg for .mp4.")
        writer = animation.FFMpegWriter(fps=fps)
    else:
        raise ValueError("out_path must end in .mp4 or .gif.")

    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path
