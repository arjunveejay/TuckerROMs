import numpy as np
from rbf.basis import get_rbf
from rbf.poly import mvmonos
from rbf.linalg import PosDefSolver, PartitionedPosDefSolver

class RBFWeights:
    """
    Build y(mu) so that f(mu) ≈ sum_k y_k(mu) f(mu_k).
    basis in {"gaussian","imq","mq","phs"}.
    Args:
      mus: (N,d) training sites
      basis: str
      eps: shape parameter (ignored for phs scale-invariance, but accepted)
      order: int polynomial degree, set -1 for none. For PHS use >= CPD-1.
      nugget: float, adds nugget*I to K
    """
    _name_map = {"gaussian": "ga", "imq": "imq", "mq": "mq", "phs": "phs3"}

    def __init__(self, mus, basis="gaussian", eps=1.0, order=-1, nugget=0.0):
        mus = np.asarray(mus, float)
        assert mus.ndim == 2
        self.mus = mus
        self.N, self.d = mus.shape

        # pick rbf basis
        try:
            phi_name = self._name_map[basis.lower()]
        except KeyError:
            raise ValueError("basis must be one of {'gaussian','imq','mq','phs'}")
        self.phi = get_rbf(phi_name)        # e.g. 'ga', 'imq', 'mq', 'phs3'
        self.eps = float(eps)
        self.order = int(order)

        # kernel matrix K = phi(mus, mus)
        K = self.phi(mus, mus, eps=self.eps)
        if nugget and nugget > 0:
            K = K + float(nugget) * np.eye(self.N)

        # with or without polynomial tail
        if self.order >= 0:
            P = mvmonos(mus, self.order)    # (N,M)
            # KKT solve with positive-definite A=K
            self._solver = PartitionedPosDefSolver(K, P, build_inverse=False)
            self._poly = True
        else:
            self._solver = PosDefSolver(K)
            self._poly = False

    def weights(self, mu):
        """Return y(mu) of shape (N,)"""
        mu = np.asarray(mu, float).ravel()
        assert mu.size == self.d
        varphi = self.phi(mu[None, :], self.mus, eps=self.eps).ravel()
        if self._poly:
            b = mvmonos(mu[None, :], self.order).ravel()
            y, _ = self._solver.solve(varphi, b)  # solves KKT, returns (a,lambda)
            return y
        else:
            return self._solver.solve(varphi)

    def weights_many(self, MU):
        """MU: (Q,d) → Y: (N,Q)"""
        MU = np.asarray(MU, float)
        assert MU.ndim == 2 and MU.shape[1] == self.d
        VARPHI = self.phi(MU, self.mus, eps=self.eps).T  # (N,Q)
        if self._poly:
            B = mvmonos(MU, self.order).T                # (M,Q)
            Y, _ = self._solver.solve(VARPHI, B)         # columnwise
            return Y
        else:
            return self._solver.solve(VARPHI)

# -------------------------- example usage --------------------------
if __name__ == "__main__":
    # 1D toy sites
    X = np.linspace(0, 1, 15)[:, None]

    rbfw = RBFWeights(
        mus=X,
        basis="gaussian",  # or "imq", "mq", "phs"
        eps=3.0,
        order=-1,          # set >=0 to add polynomial tail (e.g., 0 for constant)
        nugget=1e-12,
    )

    mu = np.array([0.37])
    y = rbfw.weights(mu)          # (N,)
    MU = np.linspace(0, 1, 50)[:, None]
    Y = rbfw.weights_many(MU)     # (N,50)

    # interpolation check at a node
    e3 = rbfw.weights(X[3])
    print("max offdiag error at node:", np.max(np.abs(e3 - np.eye(len(X))[:, 3])))
