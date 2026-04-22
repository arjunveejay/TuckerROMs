# FOM.py
__all__ = [
    "WaveFEM",
    "WaveFEM2D",
]

import abc
import numpy as np
import IPython.display
import scipy.sparse as sp
import matplotlib.animation
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla

import ngsolve as ng
from ngsolve.webgui import Draw
from netgen.geom2d import SplineGeometry



# Full-order finite-element-based models ======================================
class WaveFEM(abc.ABC):

    dim = NotImplemented
    dirichlet_BCs = NotImplemented
    parameter_dimension = NotImplemented

    def __init__(self, orderW: int = 1, orderV: int = 1):
        # Initialize the finite element space
        self.mesh = self.create_mesh()
        self.W = ng.L2(self.mesh, order=orderW)
        if self.dim == 1:
            self.V = ng.H1(self.mesh, order=orderV)
        else:
            self.V = ng.HDiv(self.mesh, order=orderV, RT=True)

        # Get the mass matrix associated with space W
        M = ng.BilinearForm(self.W)
        M += self.W.TrialFunction() * self.W.TestFunction() * ng.dx
        self.MW = self._tocoo(M).tocsc()
        self.MWinv = spla.inv(self.MW)

        # Get the S matrix
        self.S = self._tocoo(self.getS()).tocsc()

        self.MVs = self.getMVs()
        MV = sum(self.MVs)
        self.MV_A = spla.spsolve(MV, self.S)
        self.D = self.MWinv @ self.S.T @ self.MV_A
        

        # ICs
        self.q0 = self._asarray(self.initial_condition()[0])
        self.p0 = self._asarray(self.initial_condition()[1])

    @property
    def Nx(self) -> int:
        """Size of the spatial discretization"""
        return self.W.ndof

    @property
    def nodes(self) -> np.ndarray:
        """Nodes in the spatial mesh"""
        return self.getNodes()

    def __str__(self):
        """String representation."""
        interval = f"[0, {self.L:.4f}]"
        domain = " x ".join([interval] * self.dim)
        return "\n".join(
            [
                "Parametric wave equation finite element model",
                f"  Spatial domain ({self.dim:d}D): {domain}",
                f"  Discretization size: {self.Nx}",
            ]
        )

    # Abstract methods --------------------------------------------------------
    @abc.abstractmethod
    def create_mesh(self):
        """Create the spatial finite element mesh."""
        raise NotImplementedError

    @abc.abstractmethod
    def initial_condition(self, asarray: bool = True):
        """Construct the initial condition.

        Parameters
        ----------
        asarray : bool
            If ``True`` (default), return a NumPy array.
            If ``False``, return an ``ngsolve`` object.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def getS(self):
        """Get the coupling matrix corresponding to the mixed bilinear form."""
        raise NotImplementedError

    @abc.abstractmethod
    def getMVs(self) -> list:
        """Assemble the (parameter-independent) matrices (M_V)_1, ..., (M_V)_p
        used in constructing the parameter-dependent matrix A(μ).
        """
        raise NotImplementedError

    # Utilities ---------------------------------------------------------------
    def getNodes(self) -> np.ndarray:
        """Returns the coordinates of the spatial nodes"""
        pnts_x = []
        for e in self.mesh.edges:
            for v in e.vertices:
                pnts_x.append(self.mesh[v].point[0])
        if self.W.globalorder == 0:
            return np.array(pnts_x)[::2] / 2 + np.array(pnts_x)[1::2] / 2
        return pnts_x

    def _tocoo(self, A) -> sp.coo_array:
        """Convert an ngsolve matrix to a SciPy sparse COO array."""
        coo = A.Assemble().mat.COO()
        return sp.coo_array((coo[2], coo[:2])).tocsc()

    def _asarray(self, func):
        """Convert an ngsolve function to a NumPy array."""
        f = ng.GridFunction(self.W)
        f.Set(func)
        return f.vec.FV().NumPy()


    def save_VTK(self, q, filename):
        """Save the solution to a VTK file."""
        gfu = ng.GridFunction(self.W)
        gfu.vec.FV().NumPy()[:] = q
        vtk = ng.VTKOutput(
            ma=self.mesh,
            coefs=[gfu],
            names=["q(T)"],
            filename=filename,
            subdivision=3,
        )
        vtk.Do()

    # Solver ------------------------------------------------------------------
    def solve(self, t, omega, x0, y0):
        """Solve the problem for a given parameter vector."""
        dt = t[1] - t[0]
        # MV = sum(self.MVs)
        # MV_A = spla.spsolve(MV, self.S)

        #D = self.MWinv @ self.S.T @ MV_A
        D = self.D.copy()
    
        D *= dt / 2
        D += 2 / dt * sp.identity(D.shape[0], format="csr")
        D_factor = spla.factorized(D)

        Q = np.zeros((self.Nx, len(t)))
        P = np.zeros((self.Nx, len(t)))

        Q[:, 0] = self.q0
        P[:, 0] = self.p0
        forcing = lambda t_ : self.MWinv @ self.forcing(x0, y0)*np.cos(omega * t_)
        
        # Symplectic time integrato 
        for i in range(1, len(t)):
            fHalf = forcing(t[i-1]/2+t[i]/2)
            qHalf = D_factor(2 / dt * Q[:, i - 1] + P[:, i - 1] + dt/2*fHalf)
            Q[:, i] = 2 * qHalf - Q[:, i - 1]
            P[:, i] = 4 / dt * (qHalf - Q[:, i - 1]) - P[:, i - 1]
        return Q, P
    

    def solve_multi(self, muarr, t):
        """Solve the problem for multiple parameter vectors."""
        Q_list, P_list = [], []

        for mu in muarr:
            if self.dim == 2:
                Q, P = self.solve(t, mu[0], mu[1], mu[2])
            else:
                Q, P = self.solve(t, mu[0], mu[1], 0)
            Q_list.append(Q)
            P_list.append(P)

        return np.array(Q_list), np.array(P_list)



    def Hamiltonian(self, Q, P, mu):
        """Compute the time-dependent Hamiltonian given position snapshots Q
        and momentum snapsots P corresponding to the parameter vector mu.
        """
        H = np.empty(Q.shape[1])
        MV = sum(1 / mui * MVi for mui, MVi in zip(mu, self.MVs))
        MVinv = self.S.T @ spla.inv(MV) @ self.S

        for i in range(Q.shape[1]):
            H[i] = (
                P[:, i].T @ (self.MW @ P[:, i]) + Q[:, i].T @ MVinv @ Q[:, i]
            )

        return 0.5 * H


    def animate(self, Q, skip=5):
        """Animate a single evolution profile in time in Jupyter notebook.

        Parameters
        ----------
        Q : (Nx, Nt) ndarray
            Trajectory to animate.
        skip : int
            Animate every `skip` snapshots, so the total number of
            frames is `Nt // skip`.
        """
        if Q.ndim != 2:
            raise ValueError("two-dimensional data required for animation")
        x = self.nodes

        # Initialize the figure and subplots.
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 4), dpi=200)
        lines = [ax.plot([], [])[0]]

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(index):
            lines[0].set_data(x, Q[:, index * skip])
            ax.set_title(rf"$t = t_{{{index*skip}}}$")
            return lines

        ax.axvline(self.x1, linestyle=":", linewidth=0.5)
        ax.axvline(self.x2, linestyle=":", linewidth=0.5)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(Q.min() * 0.95, Q.max() * 1.05)
        ax.set_title(r"$t = t_{0}$")

        a = matplotlib.animation.FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=Q.shape[1] // skip,
            interval=50,
            blit=True,
        )
        plt.close(fig)
        return IPython.display.HTML(a.to_jshtml())


class WaveFEM2D(WaveFEM):
    r"""

    Parameters
    ----------
    L : float
        Length of one side of the spatial domain.
    h : float
        Maximum mesh spacing.
    orderW : int
        Polynomial order of the L2 finite element space.
    orderV : int
        Polynomial order of the HDiv finite element space.
    """

    dim = 2
    parameter_dimension = 3
    plot_settings = {
        # "camera": {
        #     "transformations": [
        #         {"type": "rotateX", "angle": 0},
        #     ]
        # },
        # "deformation": 3.0,
        "edges": False,
        "mesh": False,
    }

    def __init__(
        self,
        L: float = 2 * np.pi,
        h: float = 0.25,
        orderW: int = 1,
        orderV: int = 1,
        width: float = 0.5
    ):
        self.L, self.h = L, h
        self.width = width
        super().__init__(orderW, orderV)

    # Implement abstract methods ----------------------------------------------
    def create_mesh(self):
        """Create a 2D spatial mesh over [0, L] x [0, L] with max spacing h."""
        L = self.L
        L2 = L / 2

        geo = SplineGeometry()
        geo.AddRectangle(
            (0, 0),
            (L2, L2),
            bcs=["b1", "r1", "t1", "l1"],
            leftdomain=1,
        )
        geo.AddRectangle(
            (L2, 0),
            (L, L2),
            bcs=["b2", "r2", "t2", "l2"],
            leftdomain=2,
        )
        geo.AddRectangle(
            (0, L2),
            (L2, L),
            bcs=["b3", "r3", "t3", "l3"],
            leftdomain=3,
        )
        geo.AddRectangle(
            (L2, L2),
            (L, L),
            bcs=["b4", "r4", "t4", "l4"],
            leftdomain=4,
        )

        geo.SetMaterial(1, "d1")
        geo.SetMaterial(2, "d2")
        geo.SetMaterial(3, "d3")
        geo.SetMaterial(4, "d4")

        return ng.Mesh(geo.GenerateMesh(maxh=self.h))

    def initial_condition(self):
        mid = self.L / 2
        return (
            0*ng.sin(ng.x)
            * ng.sin(ng.y)
        ), 0
    
    def forcing(self, x0, y0):
        f = ng.LinearForm(self.W)
        g = ng.exp(-0.5*((ng.x-x0)**2+(ng.y-y0)**2)/(self.width**2))
        g *= ng.sin(ng.x/2)*ng.sin(ng.y/2)
        f += g*self.W.TestFunction()*ng.dx(bonus_intorder=4)
        f = f.Assemble().vec.FV().NumPy()
        return f


    def getS(self):
        S = ng.BilinearForm(trialspace=self.W, testspace=self.V)
        S += self.W.TrialFunction() * ng.div(self.V.TestFunction()) * ng.dx
        return S

    def getMVs(self) -> list:
        u, v = self.V.TnT()
        materials = self.mesh.GetMaterials()
        out = []

        for i in range(len(materials)):
            Mi = ng.BilinearForm(self.V, check_unused=False)
            Mi += u * v * ng.dx(definedon=self.mesh.Materials(materials[i]))
            out.append(self._tocoo(Mi).tocsc())
        return out

    def plot(self, Q):
        """Plot snapshots of a solution trajectory over the spatial domain.

        Parameters
        ----------
        Q : (Nx, Nt) ndarray
            Trajectory to plot.
        indices : tuple
            Time indices to plot.
        """

        gfu = ng.GridFunction(self.W)
        gfu.vec.FV().NumPy()[:] = Q[:, 0]
        scene = Draw(gfu, settings=self.plot_settings)

        for j in range(1, Q.shape[1]):
            print(f"t = t_{j}", end="\r")
            gfu.vec.FV().NumPy()[:] = Q[:, j]
            scene.Redraw()
