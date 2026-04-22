# heatEq.py
__all__ = [
    "HeatFEM",
    "HeatFEM2D",
]

import abc
import numpy as np
import scipy.integrate
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import ngsolve as ng
from ngsolve.webgui import Draw
from ngsolve.meshes import Make1DMesh
from netgen.geom2d import SplineGeometry
import IPython.display



class HeatFEM(abc.ABC):
    """Base class for finite element models of the heat equation with a
    piecewise constant diffusion coefficient and homogeneous Dirichlet
    boundary conditions. The semi-discrete system has the form

        M dq/dt = -Sq(t) + f(mu, t)

    where f is a parameter dependent forcing term.

    The initial condition and the underlying spatial dimension and geometry are specified in child classes.

    Parameters
    ----------
    t0 : float
        Initial time.
    tf : float
        Final time.
    nt : int
        Number of time steps.
    order : int
        Polynomial order of the finite element space.
    """

    dim = NotImplemented
    dirichlet_BCs = NotImplemented
    parameter_dimension = NotImplemented

    def __init__(self, order: int = 1):
        # Initialize the finite element mesh and space.
        self.mesh = self.create_mesh()
        self.V = ng.H1(self.mesh, order=order, dirichlet=self.dirichlet_BCs)
        self.__x = np.array([self.mesh[v].point for v in self.mesh.vertices])
        if self.parameter_dimension == 1:
            self.__x = np.ravel(self.__x)

        # Assemble the mass matrix.
        u, v = self.V.TnT()
        M = ng.BilinearForm(self.V)
        M += u * v * ng.dx
        M = self._tocoo(M).tocsc()

        # Assemble stiffness matrices and the initial condition.
        As = self.stiffness_matrices()
        q0 = self._asarray(self.initial_condition())

        

        # Get the degrees of freedom not accounted for by boundary conditions.
        self.free = self.V.FreeDofs()
        self.M = M[self.free][:, self.free]
        self.Minv = spla.splu(self.M).solve
        self.As = [A[self.free][:, self.free] for A in As]
        self.q0 = q0[self.free]

    @property
    def Nx(self) -> int:
        """Size of the spatial discretization."""
        return self.V.ndof

    @property
    def Nx_free(self) -> int:
        """Number of degrees of freedom not fixed by boundary conditions."""
        return self.q0.size

    @property
    def nodes(self) -> np.ndarray:
        """Nodes in the spatial mesh."""
        return self.__x

    def __str__(self):
        """String representation."""
        interval = f"[0, {self.L:.4f}]"
        domain = " x ".join([interval] * self.dim)
        return "\n".join(
            [
                "Parametric heat equation finite element model",
                f"  Spatial domain ({self.dim:d}D): {domain}",
                f"  Discretization size: {self.Nx}",
                f"  Degrees of freedom:  {self.Nx_free}",
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
    def stiffness_matrices(self) -> list:
        """Assemble the (parameter-independent) matrices S_1, ..., S_p.
        """
        raise NotImplementedError

    # Utilities ---------------------------------------------------------------
    def _tocoo(self, A) -> sp.coo_array:
        """Convert an ngsolve matrix to a SciPy sparse COO array."""
        coo = A.Assemble().mat.COO()
        return sp.coo_array((coo[2], coo[:2])).tocsc()

    def _asarray(self, func):
        """Convert an ngsolve function to a NumPy array."""
        f = ng.GridFunction(self.V)
        f.Set(func)
        return f.vec.FV().NumPy()

    def pad(self, q_free: np.ndarray) -> np.ndarray:
        """Pad with the homogeneous boundary conditions.

        Parameters
        ----------
        q_free : (N_free, ...) ndarray
            Vector of coefficients for the degrees of freedom.

        Returns
        -------
        q_all : (Nx, ....) ndarray
            Values over the nodes, including the zero boundary conditions.
        """
        shape = list(q_free.shape)
        shape[0] = self.Nx
        q = np.zeros(shape)
        q[self.free] = q_free[:]
        return q

    def sample_parameters(
        self,
        low: list,
        high: list,
        num_samples: int = 10,
        train_ratio: float = 0.8,
        randseed: int = 0,
    ):
        """Sample the parameter space to generate training and testing sets.

        Parameters
        ----------
        low : list of float
            Lower bounds for each component.
        high : list of float
            Upper bounds for each component.
        num_samples : int
            Total number of parameter vectors.
        train_ratio : float
            Proportion of the parameter vectors to be used for training.
        randseed : int
            Random seed for the sampling.

        Returns
        -------
        training_parameters : (N_train, dim) ndarray
            Parameter vectors to train on.
        testing_parameters : (N_test, dim) ndarray
            Parameter vectors to test on.
        """

        np.random.seed(randseed)

        low = np.array(low, dtype=float)
        high = np.array(high, dtype=float)

        if low.shape != high.shape:
            raise ValueError("low and high must have the same length")

        dim = len(low)

        # sample each component independently
        samples = np.random.uniform(
            low=low,
            high=high,
            size=(num_samples, dim),
        )

        split = int(train_ratio * num_samples)
        return samples[:split], samples[split:]

    # Solver ------------------------------------------------------------------
    def derivative(self, mu, q): NotImplementedError
        # """Evaluate the time derivative dq/dt = -M^{-1}S(mu)q."""
        # A = mu*sum(self.As)
        # return self.Minv(A @ q)

    def solve(self, eps, x0, y0, t):
        """Solve the problem for a given parameter vector."""
        
        A =1*sum(self.As)
        jac = self.Minv(A.toarray())

        forcing_ = lambda t_ : self.forcing(eps, x0, y0)*ng.exp(-t_)
        
        def fun(tt, y):
            return self.Minv(A @ y + forcing_(tt))

        # Solve over the non-boundary degrees of freedom.
        return scipy.integrate.solve_ivp(
            fun=fun,
            t_span=[t[0], t[-1]],
            y0=self.q0,
            method="BDF",
            t_eval=t,
            jac=jac,
            vectorized=False,
        ).y

    def solve_multi(self, params, t):
        """Solve the problem for multiple parameter vectors."""
        return np.array([self.solve(param[0], param[1], param[2], t) for param in params])


class HeatFEM2D(HeatFEM):
    """Finite element model for the one-dimensional heat equation with a
    piecewise constant diffusion coefficient and homogeneous Dirichlet
    boundary conditions.

    The governing equation is

        dq/dt = div(c grad(q)] + f(mu; x, t)

    defined over the square, two-dimensional spatial domain [0, L] x [0, L], with boundary conditions q(x,t) = 0 and initial condition

        q(x,0) = 0

    where x = (x1, x2). The semi-discrete system has the form

        M dq/dt = -Sq(t) + f(mu; t),

    where f(mu; t) is the L2-projection of the forcing function

        F(mu, t) = exp(-( (x-mu2)^2+(y-y0)^2)/(mu1^2) ) / mu1^2 / np.pi
    

    Parameters
    ----------
    L : float
        Length of one side of the spatial domain.
    h : float
        Maximum mesh spacing.
    order : int
        Polynomial order of the finite element space.
    """

    dim = 2
    dirichlet_BCs = "b1|b2|l1|l3|t3|t4|r2|r4"
    parameter_dimension = 4
    plot_settings = {
        # "camera": {
        #     "transformations": [
        #         {"type": "rotateY", "angle": 20},
        #         {"type": "rotateZ", "angle": 40},
        #     ]
        # },
        # "deformation": 3.0,
        "edges": False,
        "mesh": False,
    }

    def __init__(self, L: float = 2 * np.pi, h: float = 0.25, order: int = 1):
        self.L, self.h = L, h
        super().__init__(order)

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
        return 0*(
            ng.exp(-((ng.x - mid) ** 2) - ((ng.y - mid) ** 2))
            * ng.sin(ng.x / 2)
            * ng.sin(ng.y / 2)
        )

    def stiffness_matrices(self):
        u, v = self.V.TnT()
        materials = self.mesh.GetMaterials()

        out = []
        for i in range(len(materials)):
            Ai = ng.BilinearForm(self.V, check_unused=False)
            Ai += (
                ng.grad(u)
                * ng.grad(v)
                * ng.dx(definedon=self.mesh.Materials(materials[i]))
            )
            out.append(-self._tocoo(Ai).tocsr())
        return out

    # Visualization -----------------------------------------------------------
    def plot(self, Qfree):
        """Plot snapshots of a solution trajectory over the spatial domain.
        In a notebook, this is displayed as a once-through animation.

        Parameters
        ----------
        Qfree : (Nx_free, Nt) ndarray
            Trajectory to plot.
        indices : tuple
            Time indices to plot.
        """
        Q = self.pad(Qfree)
        gfu = ng.GridFunction(self.V)
        gfu.vec.FV().NumPy()[:] = Q[:, 0]
        scene = Draw(gfu, settings=self.plot_settings)

        for j in range(1, Q.shape[1]):
            print(f"t = t_{j}", end="\r")
            gfu.vec.FV().NumPy()[:] = Q[:, j]
            scene.Redraw()

    def forcing(self, eps, x0, y0):
        # spatial variation only
        f = ng.LinearForm(self.V)
        g = eps*ng.exp(-0.5*((ng.x-x0)**2+(ng.y-y0)**2)/(0.4**2))
        g *= ng.sin(ng.x/2)*ng.sin(ng.y/2)
        f += g*self.V.TestFunction()*ng.dx(bonus_intorder=3)
        F = f.Assemble().vec.FV().NumPy()[self.free]
        return F