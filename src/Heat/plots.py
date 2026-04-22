import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import ngsolve as ng


_METHODS = ["rbf", "mono", "lid"]
_COLORS  = {"mono": "C0", "rbf": "C1", "lid": "C2"}
_LABELS  = {"mono": "Monolithic", "rbf": "RBF", "lid": "MO"}
_MARKERS = {"mono": "^", "rbf": "s", "lid": "o"}
_MS      = {"mono": 6, "rbf": 7.5, "lid": 6}
_LW      = {"mono": 2.2, "rbf": 2.6, "lid": 2.2}


def plot_errors(errs, methods=None, xlabel=r"$r$"):
    """Training/Testing error plot with median + IQR band.

    Parameters
    ----------
    errs    : dict-like (e.g. np.load(...))
              Must contain ``r_arr`` and keys ``{m}_rom_train``,
              ``{m}_proj_train``, ``{m}_rom_test``, ``{m}_proj_test``
              for each method ``m`` in *methods*.
    methods : list, optional  (default: ["rbf", "mono", "lid"])
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                             constrained_layout=True, sharex=True, sharey=True)

    _subplot(axes[0], " Training", "{m}_rom_train", "{m}_proj_train")
    _subplot(axes[1], " Testing",  "{m}_rom_test",  "{m}_proj_test")

    for a in axes:
        a.set_xlabel(xlabel)
    axes[0].set_ylabel(r"Relative error")

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
               bbox_to_anchor=(0.5, -0.13))
    return fig


def eval_grid(fom, q_vec, N=80):
    """Evaluate a Heat FOM solution vector (free DOFs only) on an N×N regular grid."""
    gfu  = ng.GridFunction(fom.V)
    free = np.array(list(fom.V.FreeDofs()), dtype=bool)
    full = np.zeros(fom.V.ndof)
    full[free] = q_vec
    gfu.vec.FV().NumPy()[:] = full
    xs = np.linspace(0, fom.L, N)
    ys = np.linspace(0, fom.L, N)
    return np.array([[gfu(fom.mesh(x, y)) for x in xs] for y in ys])


def plot_contours(fom, Z_fom, Z_rbf, Z_mono, Z_lid,
                  sol_label="Solution", err_label="Signed Error",
                  nlevels=20, N=80, clims=None, eclims=None):
    """
    Two-row contour plot: top row = FOM / RBF / Monolithic / MO solutions,
    bottom row = (blank) / RBF error / Mono error / MO error.

    Parameters
    ----------
    fom       : HeatFEM2D instance (for domain length L)
    Z_fom, Z_rbf, Z_mono, Z_lid : (N, N) arrays from eval_grid
    sol_label : colorbar label for the solution row
    err_label : colorbar label for the error row
    nlevels   : number of contour levels
    N         : grid resolution (must match the Z arrays)
    clims     : (vmin, vmax) for the solution row; auto-computed if None
    eclims    : (emin, emax) for the error row; auto-computed if None
    """
    X, Y = np.meshgrid(np.linspace(0, fom.L, N), np.linspace(0, fom.L, N))

    E_rbf  = Z_fom - Z_rbf
    E_mono = Z_fom - Z_mono
    E_lid  = Z_fom - Z_lid

    if clims is not None:
        vmin_sol, vmax_sol = clims
    else:
        vmin_sol = min(Z_fom.min(), Z_rbf.min(), Z_mono.min(), Z_lid.min())
        vmax_sol = max(Z_fom.max(), Z_rbf.max(), Z_mono.max(), Z_lid.max())

    if eclims is not None:
        emin, emax = eclims
    else:
        emax = max(abs(E_rbf).max(), abs(E_mono).max(), abs(E_lid).max())
        emin = -emax

    sol_levels = np.linspace(vmin_sol, vmax_sol, nlevels + 1)
    err_levels = np.linspace(emin, emax, nlevels + 1)
    kw_sol     = dict(levels=sol_levels, cmap="viridis")
    kw_err     = dict(levels=err_levels, cmap="RdBu_r")

    fig = plt.figure(figsize=(18, 10))
    gs  = fig.add_gridspec(2, 4, hspace=0.05, wspace=0.1)

    ax_fom   = fig.add_subplot(gs[0, 0])
    ax_rbf   = fig.add_subplot(gs[0, 1], sharey=ax_fom)
    ax_mono  = fig.add_subplot(gs[0, 2], sharey=ax_fom)
    ax_lid   = fig.add_subplot(gs[0, 3], sharey=ax_fom)
    ax_erbf  = fig.add_subplot(gs[1, 1])
    ax_emono = fig.add_subplot(gs[1, 2], sharey=ax_erbf)
    ax_elid  = fig.add_subplot(gs[1, 3], sharey=ax_erbf)

    for ax in [ax_rbf, ax_mono, ax_lid, ax_emono, ax_elid]:
        ax.tick_params(labelleft=False)

    def _plot(ax, Z, kw, show_ylabel=True):
        Z = np.clip(Z, kw['levels'][0], kw['levels'][-1])
        cf = ax.contourf(X, Y, Z, **kw)
        levels = cf.levels
        pos = levels[levels >= 0]
        neg = levels[levels < 0]
        ckw = dict(colors='k', linewidths=0.5)
        if len(pos):
            ax.contour(X, Y, Z, levels=pos, linestyles='solid', **ckw)
        if len(neg):
            ax.contour(X, Y, Z, levels=neg, linestyles='dashed', **ckw)
        ax.set_xlabel(r"$x_1$")
        if show_ylabel:
            ax.set_ylabel(r"$x_2$")
        ax.set_aspect("equal")
        return cf

    cf_sol = _plot(ax_fom,   Z_fom,  kw_sol, show_ylabel=True)
    _plot(ax_rbf,   Z_rbf,  kw_sol, show_ylabel=False)
    _plot(ax_mono,  Z_mono, kw_sol, show_ylabel=False)
    _plot(ax_lid,   Z_lid,  kw_sol, show_ylabel=False)
    cf_err = _plot(ax_erbf,  E_rbf,  kw_err, show_ylabel=True)
    _plot(ax_emono, E_mono, kw_err, show_ylabel=False)
    _plot(ax_elid,  E_lid,  kw_err, show_ylabel=False)

    for ax, lbl in zip(
        [ax_fom, ax_rbf, ax_mono, ax_lid],
        ["FOM",  "RBF",  "Monolithic", "MO"]
    ):
        ax.set_title(lbl)

    fig.canvas.draw()

    cbar_w   = 0.012
    cbar_gap = 0.02

    pos_lid  = ax_lid.get_position()
    pos_elid = ax_elid.get_position()

    cax_sol = fig.add_axes([pos_lid.x1  + cbar_gap, pos_lid.y0,  cbar_w, pos_lid.height])
    cax_err = fig.add_axes([pos_elid.x1 + cbar_gap, pos_elid.y0, cbar_w, pos_elid.height])

    def _sci(x, _):
        if x == 0:
            return r'$0$'
        s = f'{x:.2e}'
        s = s.replace('e-0', 'e-').replace('e+0', 'e+').replace('e+', 'e')
        return s

    fmt = ticker.FuncFormatter(_sci)

    cb_sol = fig.colorbar(cf_sol, cax=cax_sol)
    cb_sol.set_ticks([sol_levels[0], sol_levels[-1]])
    cb_sol.formatter = fmt
    cb_sol.update_ticks()

    cb_err = fig.colorbar(cf_err, cax=cax_err)
    cb_err.set_ticks([err_levels[0], err_levels[-1]])
    cb_err.formatter = fmt
    cb_err.update_ticks()

    label_x = pos_lid.x1 + cbar_gap + cbar_w + 0.01
    fig.text(label_x, (pos_lid.y0  + pos_lid.y1)  / 2, sol_label,
             rotation=90, va='center', ha='left')
    fig.text(label_x, (pos_elid.y0 + pos_elid.y1) / 2, err_label,
             rotation=90, va='center', ha='left')

    return fig


def plot_singular_values(s_rbf, s_mono, s_lid, methods=None):
    """Normalized singular value decay for each basis type.

    Parameters
    ----------
    s_rbf  : 1-D array  singular values for Radial Basis Functions basis
    s_mono : 1-D array  singular values for Monolithic basis
    s_lid  : 1-D array  singular values for MO basis
    methods : list, optional  subset of ["rbf", "mono", "lid"] to plot
    """
    if methods is None:
        methods = _METHODS

    svs = {"rbf": np.asarray(s_rbf), "mono": np.asarray(s_mono), "lid": np.asarray(s_lid)}

    local_methods = [m for m in methods if m != "mono"]
    n_plot = max(len(svs[m]) for m in local_methods) if local_methods else len(svs["mono"])

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    for m in methods:
        s = svs[m][:n_plot]
        s_norm = s / s[0]
        ax.semilogy(np.arange(1, len(s_norm) + 1), s_norm,
                    color=_COLORS[m], lw=_LW[m],
                    marker=_MARKERS[m], ms=_MS[m], markevery=5, label=_LABELS[m])

    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Normalized singular value")
    ax.grid(True, which="major", ls="-", alpha=0.35)
    ax.grid(True, which="minor", ls=":", alpha=0.2)
    ax.legend(frameon=False)

    return fig
