import os
import sys
import numpy as np
import pyvista as pv
from tqdm import tqdm
from pathlib import Path

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
              reference_clims=None,   # <-- NEW: dict or None
              return_clims=False):    # <-- NEW: if True and reference_clims None, return computed clims
    """
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

        _render_field_png(mesh, "E", outE,
                          plane_origin=plane_origin, plane_normal=plane_normal,
                          cmap=cmap, image_size=image_size,
                          panel_clims=reference_clims["E"])
        _render_field_png(mesh, "B", outB,
                          plane_origin=plane_origin, plane_normal=plane_normal,
                          cmap=cmap, image_size=image_size,
                          panel_clims=reference_clims["B"])

    if return_clims:
        return reference_clims
