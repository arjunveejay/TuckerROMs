import os
import sys
import numpy as np
import pyvista as pv
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from scipy.interpolate import griddata

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
    "font.size": 20,
    "font.family": "serif",
    "axes.titlesize": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

ROOT = Path(__file__).resolve().parents[2]  
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.Maxwell.FOM import evaluate_at_cell_center


def _render_field_png(mesh_full, name, out_png,
                      plane_origin=(0.5, 0.5, 0.5),
                      plane_normal=(0, 0, 1),
                      cmap="jet",
                      image_size=(1200, 300),   # (width, height)
                      top_view_axis="+z",
                      panel_clims=None):
    """
    Render a 1x4 panel (|field|, x, y, z) from mesh_full.cell_data[name] and save to PNG.

    panel_clims (optional): dict with keys {"mag","x","y","z"} -> (vmin,vmax)
      - "mag" is used for the magnitude panel
      - "x","y","z" used for component panels
      If a key is missing or None, PyVista will auto-scale that panel.
    """

    W, H = map(int, image_size)
    panel_clims = panel_clims or {}

    # Slice for plotting
    mesh = mesh_full.slice(normal=plane_normal, origin=plane_origin)

    pl = pv.Plotter(shape=(1, 4), off_screen=True)
    pl.set_background("white")

    # Force window size explicitly
    pl.window_size = (W, H)

    # Unique-but-invisible titles so each scalar bar is distinct (older PyVista uses title as key)
    zws = ["\u200b", "\u200b\u200b", "\u200b\u200b\u200b", "\u200b\u200b\u200b\u200b"]

    def sb_args(sb_title):
        return dict(
            title=sb_title,
            n_labels=2,          # only extreme ticks
            fmt="%.2e",
            color="black",
            vertical=False,      # horizontal bar in older PyVista
            position_x=0.20,     # centered-ish
            position_y=0.02,
            width=0.60,
            height=0.06,
        )

    def add_panel(col, mesh_to_plot, component, panel_title, sb_title, clim=None):
        pl.subplot(0, col)
        if component is None:
            pl.add_mesh(
                mesh_to_plot,
                scalars=name,
                cmap=cmap,
                lighting=False,
                show_scalar_bar=True,
                scalar_bar_args=sb_args(sb_title),
                clim=clim,
            )
        else:
            pl.add_mesh(
                mesh_to_plot,
                scalars=name,
                component=component,
                cmap=cmap,
                lighting=False,
                show_scalar_bar=True,
                scalar_bar_args=sb_args(sb_title),
                clim=clim,
            )
        pl.add_text(panel_title, color="k", font_size=10)

    # Panels: magnitude + components
    add_panel(0, mesh,        None, f"|{name}|", zws[0], clim=panel_clims.get("mag"))
    add_panel(1, mesh.copy(), 0,    f"{name}x",  zws[1], clim=panel_clims.get("x"))
    add_panel(2, mesh.copy(), 1,    f"{name}y",  zws[2], clim=panel_clims.get("y"))
    add_panel(3, mesh.copy(), 2,    f"{name}z",  zws[3], clim=panel_clims.get("z"))

    pl.link_views()

    # Top view camera
    if top_view_axis == "+z":
        pl.view_xy()
    elif top_view_axis == "-z":
        pl.view_xy(negative=True)
    elif top_view_axis == "+y":
        pl.view_xz()
    elif top_view_axis == "-y":
        pl.view_xz(negative=True)
    elif top_view_axis == "+x":
        pl.view_yz()
    elif top_view_axis == "-x":
        pl.view_yz(negative=True)
    else:
        raise ValueError(f"Unknown top_view_axis={top_view_axis}")

    pl.camera.SetViewUp(0, 1, 0)
    pl.camera.Zoom(1.15)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    pl.render()
    pl.screenshot(out_png)
    pl.close()


def save_pngs(mesh, file_hdf5, prefix,
              recorded_times, requested_times,
              recorded_states=None, E_arr=None, B_arr=None,
              plane_origin=(0.5, 0.5, 0.5),
              plane_normal=(0, 0, 1),
              cmap="jet",
              image_size=(1400, 1100),
              atol=1e-12, rtol=0.0,
              on_missing="error",
              reference_clims=None,
              return_clims=False,
              renderer="matplotlib",   # "matplotlib" or "pyvista"
              show_colorbar=True):
    """
    renderer: "matplotlib" (default) uses _render_field_png_mpl;
              "pyvista" uses _render_field_png.

    reference_clims format:
      {
        "E": {"mag": (vmin,vmax), "x":(...), "y":(...), "z":(...)},
        "B": {"mag": (vmin,vmax), "x":(...), "y":(...), "z":(...)}
      }
    """

    times = np.asarray(recorded_times, dtype=float)
    req = np.asarray(requested_times, dtype=float)
    nt = len(times)

    using_states = recorded_states is not None
    using_arrays = (E_arr is not None) and (B_arr is not None)
    if using_states == using_arrays:
        raise ValueError("Provide either recorded_states OR (E_arr and B_arr), but not both.")

    if using_states:
        if len(recorded_states) != nt:
            raise ValueError("recorded_times and recorded_states must have the same length.")
    else:
        E_arr = np.asarray(E_arr); B_arr = np.asarray(B_arr)
        if E_arr.ndim != 2 or B_arr.ndim != 2:
            raise ValueError("E_arr and B_arr must be 2D arrays.")
        # orient to (ndof, nt)
        if E_arr.shape[1] == nt: E_mat = E_arr
        elif E_arr.shape[0] == nt: E_mat = E_arr.T
        else: raise ValueError(f"E_arr shape {E_arr.shape} incompatible with nt={nt}.")
        if B_arr.shape[1] == nt: B_mat = B_arr
        elif B_arr.shape[0] == nt: B_mat = B_arr.T
        else: raise ValueError(f"B_arr shape {B_arr.shape} incompatible with nt={nt}.")

    def find_index(t):
        idx = np.where(np.isclose(times, t, atol=atol, rtol=rtol))[0]
        if idx.size == 0:
            return None
        return int(idx[0])

    def upd(lims, arr):
        a = np.asarray(arr)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return lims
        mn, mx = float(a.min()), float(a.max())
        if lims is None:
            return (mn, mx)
        return (min(lims[0], mn), max(lims[1], mx))

    # If no reference_clims provided, compute from the SAME slices you will plot
    computed = None
    if reference_clims is None:
        computed = {"E": {"mag": None, "x": None, "y": None, "z": None},
                    "B": {"mag": None, "x": None, "y": None, "z": None}}

        for t in req:
            k = find_index(float(t))
            if k is None:
                if on_missing == "skip":
                    continue
                raise ValueError(f"Requested time {t} not found in recorded_times.")

            if using_states:
                E_coeffs, B_coeffs = recorded_states[k]
            else:
                E_coeffs = E_mat[:, k]
                B_coeffs = B_mat[:, k]

            mesh.cell_data["E"] = evaluate_at_cell_center(file_hdf5["Eeval"], E_coeffs)
            mesh.cell_data["B"] = evaluate_at_cell_center(file_hdf5["Beval"], B_coeffs)

            ms = mesh.slice(normal=plane_normal, origin=plane_origin)
            E_s = np.asarray(ms.cell_data["E"])
            B_s = np.asarray(ms.cell_data["B"])

            computed["E"]["mag"] = upd(computed["E"]["mag"], np.linalg.norm(E_s, axis=1))
            computed["E"]["x"]   = upd(computed["E"]["x"],   E_s[:, 0])
            computed["E"]["y"]   = upd(computed["E"]["y"],   E_s[:, 1])
            computed["E"]["z"]   = upd(computed["E"]["z"],   E_s[:, 2])

            computed["B"]["mag"] = upd(computed["B"]["mag"], np.linalg.norm(B_s, axis=1))
            computed["B"]["x"]   = upd(computed["B"]["x"],   B_s[:, 0])
            computed["B"]["y"]   = upd(computed["B"]["y"],   B_s[:, 1])
            computed["B"]["z"]   = upd(computed["B"]["z"],   B_s[:, 2])

        # fill any Nones (edge case)
        for fld in ("E", "B"):
            for key in ("mag", "x", "y", "z"):
                if computed[fld][key] is None:
                    computed[fld][key] = (0.0, 1.0)

        reference_clims = computed

    # Now do the actual rendering
    for t in tqdm(req, desc="  ==> saving PNG snapshots"):
        k = find_index(float(t))
        if k is None:
            if on_missing == "skip":
                continue
            raise ValueError(f"Requested time {t} not found in recorded_times (atol={atol}, rtol={rtol}).")

        if using_states:
            E_coeffs, B_coeffs = recorded_states[k]
        else:
            E_coeffs = E_mat[:, k]
            B_coeffs = B_mat[:, k]

        mesh.cell_data["E"] = evaluate_at_cell_center(file_hdf5["Eeval"], E_coeffs)
        mesh.cell_data["B"] = evaluate_at_cell_center(file_hdf5["Beval"], B_coeffs)

        ttag = f"{times[k]:0.6f}"
        outE = f"{prefix}_t{ttag}_E.png"
        outB = f"{prefix}_t{ttag}_B.png"

        _render_field_png_mpl(mesh, "E", outE,
                              plane_origin=plane_origin, plane_normal=plane_normal,
                              cmap=cmap, image_size=image_size,
                              panel_clims=reference_clims["E"],
                              show_colorbar=show_colorbar)
        _render_field_png_mpl(mesh, "B", outB,
                              plane_origin=plane_origin, plane_normal=plane_normal,
                              cmap=cmap, image_size=image_size,
                              panel_clims=reference_clims["B"],
                              show_colorbar=show_colorbar)

    if return_clims:
        return reference_clims


def _render_field_png_mpl(mesh_full, name, out_png,
                          plane_origin=(0.5, 0.5, 0.5),
                          plane_normal=(0, 0, 1),
                          cmap="jet",
                          image_size=(1200, 300),   # (width, height)
                          top_view_axis="+z",
                          panel_clims=None,
                          show_colorbar=True):
    """
    Render a 1x4 panel (|field|, x, y, z) from mesh_full.cell_data[name] and save to PNG.
    Uses PyVista for slicing/triangulation and matplotlib for rendering.

    panel_clims (optional): dict with keys {"mag","x","y","z"} -> (vmin,vmax)
      If a key is missing or None, auto-scales that panel.
    """

    W, H = map(int, image_size)
    panel_clims = panel_clims or {}

    # Use PyVista for slice, triangulation, and cell->point interpolation
    ms = mesh_full.slice(normal=plane_normal, origin=plane_origin)
    ms = ms.triangulate()
    ms = ms.cell_data_to_point_data()

    pts = ms.points                            # (N, 3)

    # Project 3D points to the 2D viewing plane
    _ax = top_view_axis.lstrip("+-")
    if _ax == "z":
        u, v = pts[:, 0], pts[:, 1]
    elif _ax == "y":
        u, v = pts[:, 0], pts[:, 2]
    else:  # x
        u, v = pts[:, 1], pts[:, 2]

    field = ms.point_data[name]               # (N, 3)
    points2d = np.column_stack([u, v])

    # Interpolate each scalar onto a regular grid
    res = 512
    u_grid = np.linspace(u.min(), u.max(), res)
    v_grid = np.linspace(v.min(), v.max(), res)
    uu, vv = np.meshgrid(u_grid, v_grid)

    scalars    = [np.linalg.norm(field, axis=1), field[:, 0], field[:, 1], field[:, 2]]
    titles     = [fr"$|\boldsymbol{{{name}}}|$", fr"$\boldsymbol{{{name}}}_x$",
                  fr"$\boldsymbol{{{name}}}_y$", fr"$\boldsymbol{{{name}}}_z$"]
    clim_keys  = ["mag", "x", "y", "z"]
    grids      = [griddata(points2d, s, (uu, vv), method="linear") for s in scalars]

    extent = [u.min(), u.max(), v.min(), v.max()]

    dpi = 150
    fig, axes = plt.subplots(1, 4, figsize=(W / dpi, H / dpi), dpi=dpi,
                             sharey=True)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.78, bottom=0.12, wspace=0.05)

    for i, (ax, grid, title, ck) in enumerate(zip(axes, grids, titles, clim_keys)):
        clim = panel_clims.get(ck)
        vmin, vmax = clim if clim is not None else (np.nanmin(grid), np.nanmax(grid))

        im = ax.imshow(grid, origin="lower", extent=extent, aspect="equal",
                       cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")

        # Extreme ticks only
        ax.set_xticks([u.min(), u.max()])
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%g"))
        ax.tick_params(axis="x", labelsize=12)
        ax.set_xlabel(r"$x_1$", fontsize=14, labelpad=-6)

        if i == 0:
            ax.set_yticks([v.min(), v.max()])
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%g"))
            ax.tick_params(axis="y", labelsize=12)
            ax.set_ylabel(r"$x_2$", fontsize=14, labelpad=-6)
        else:
            ax.tick_params(axis="y", left=False)

        ax.text(0.97, 0.97, title, transform=ax.transAxes,
                ha="right", va="top", fontsize=13, color="black", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=2.0, alpha=0.8))

        cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                          pad=0.04, fraction=0.046, location="top")
        cb.set_ticks([])
        if show_colorbar:
            cb.ax.text(0.0, 1.6, f"{vmin:.2e}", transform=cb.ax.transAxes,
                       ha="left", va="bottom", fontsize=11)
            cb.ax.text(1.0, 1.6, f"{vmax:.2e}", transform=cb.ax.transAxes,
                       ha="right", va="bottom", fontsize=11)
        else:
            cb.ax.set_visible(False)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


_METHODS = ["mono", "rbf", "lid"]
_COLORS  = {"mono": "C0", "rbf": "C1", "lid": "C2"}
_LABELS  = {"mono": "Monolithic", "rbf": "RBF", "lid": "MO"}
_MARKERS = {"mono": "^", "rbf": "s", "lid": "o"}
_MS      = {"mono": 6, "rbf": 7.5, "lid": 6}
_LW      = {"mono": 2.2, "rbf": 2.6, "lid": 2.2}


def plot_errors(errs, methods=None, xlabel=r"$r$", figsize=None):
    """2×2 error plot for E and B fields with median + IQR band.

    Parameters
    ----------
    errs    : dict-like (e.g. np.load(...))
              Must contain ``r_arr`` and keys ``{m}_romE_train``,
              ``{m}_projE_train``, ``{m}_romE_test``, ``{m}_projE_test``,
              ``{m}_romB_train``, ``{m}_projB_train``, ``{m}_romB_test``,
              ``{m}_projB_test`` for each method ``m`` in *methods*.
    methods : list, optional  (default: ["mono", "rbf", "lid"])
    xlabel  : str, x-axis label
    """
    if methods is None:
        methods = _METHODS

    r_arr = errs["r_arr"]

    def _subplot(ax, title_text, key_fmt_rom, key_fmt_proj):
        for m in methods:
            A   = errs[key_fmt_rom.format(m=m)]
            med = np.median(A, axis=1)
            lo  = np.quantile(A, 0.25, axis=1)
            hi  = np.quantile(A, 0.75, axis=1)
            ax.plot(r_arr, med, color=_COLORS[m], lw=_LW[m],
                    marker=_MARKERS[m], ms=_MS[m])
            ax.fill_between(r_arr, lo, hi, color=_COLORS[m], alpha=0.07)

            P    = errs[key_fmt_proj.format(m=m)]
            pmed = np.median(P, axis=1)
            ax.plot(r_arr, pmed, color=_COLORS[m], lw=_LW[m], ls="--",
                    marker=_MARKERS[m], ms=_MS[m], alpha=0.95)

        ax.set_yscale("log")
        ax.grid(True, which="major", ls="-", alpha=0.35)
        ax.text(0.98, 0.98, title_text, transform=ax.transAxes,
                ha="right", va="top", fontsize=18,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.0))

    if figsize is None: 
        figsize = (15, 9)

    fig, ax = plt.subplots(2, 2, figsize=figsize,
                           constrained_layout=True, sharex=True, sharey=True)

    _subplot(ax[0, 0], r"$\boldsymbol{e}$ Training", "{m}_romE_train", "{m}_projE_train")
    _subplot(ax[0, 1], r"$\boldsymbol{e}$ Testing",  "{m}_romE_test",  "{m}_projE_test")
    _subplot(ax[1, 0], r"$\boldsymbol{b}$ Training", "{m}_romB_train", "{m}_projB_train")
    _subplot(ax[1, 1], r"$\boldsymbol{b}$ Testing",  "{m}_romB_test",  "{m}_projB_test")

    for a in ax[1, :]:
        a.set_xlabel(xlabel)
    for a in ax[:, 0]:
        a.set_ylabel(r"Relative error")

    method_handles = [
        Line2D([0], [0], color=_COLORS[m], lw=_LW[m],
               marker=_MARKERS[m], markersize=_MS[m], label=_LABELS[m])
        for m in methods
    ]
    style_handles = [
        Line2D([0], [0], color="k", lw=2.2, ls="--", label="Projection")
    ]

    fig.subplots_adjust(bottom=0.16)
    fig.legend(method_handles + style_handles,
               [h.get_label() for h in method_handles + style_handles],
               loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.05))
    return fig
