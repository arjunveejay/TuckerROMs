import numpy as np
import tensorly as tl
from scipy.sparse.linalg import eigsh
from scipy.linalg import solve_triangular

def sample_parameters(
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


# ---------------------- Tucker ------------------------------------------

def save_tucker_npz(path, core, factors):
    payload = {"core": tl.to_numpy(core), "n_factors": np.int64(len(factors))}
    for i, U in enumerate(factors):
        payload[f"factor_{i}"] = tl.to_numpy(U)
    np.savez(path, **payload)

def load_tucker_npz(path):
    data = np.load(path, allow_pickle=False)
    core = data["core"]
    n_factors = int(data["n_factors"])
    factors = [data[f"factor_{i}"] for i in range(n_factors)]
    return core, factors


# ----------------------- Norms ---------------------------------------------
def Mnorm(Q: np.ndarray, M, dt: float = 1):
    return np.sqrt(dt * np.sum(Q * (M @ Q)))


def projection_error_M(X, U, M, Mnorm):
    """
    Relative projection error in M-norm for a time history X (N,nT) or vector (N,).
    Projection is M-orthogonal:  alpha = (U^T M U)^{-1} U^T M X
    """
    if X.ndim == 1:
        X = X[:, None]
    # reduced normal equations
    G = U.T @ (M @ U)          # (r,r)
    R = U.T @ (M @ X)          # (r,nT)
    alpha = np.linalg.solve(G, R)
    Xproj = U @ alpha
    return Mnorm(X - Xproj, M) / Mnorm(X, M)


def projection_error_L2(X, U):
    """
    Relative projection error in Euclidean norm for X (N,nT) or (N,).
    Projection is Euclidean: alpha = U^T X (assuming U is orthonormal; if not, uses least squares).
    """
    if X.ndim == 1:
        X = X[:, None]
    # If U is orthonormal, alpha = U^T X; otherwise use least-squares
    UtU = U.T @ U
    if np.allclose(UtU, np.eye(U.shape[1]), rtol=1e-10, atol=1e-12):
        alpha = U.T @ X
    else:
        alpha = np.linalg.solve(UtU, U.T @ X)
    Xproj = U @ alpha
    num = np.linalg.norm(X - Xproj)
    den = np.linalg.norm(X)
    if den == 0.0:
        raise ValueError("||X||_2 is zero; cannot form relative projection error.")
    return num / den

# ---------------------------- Bases ---------------------------------------
def buildParBasis(tucker_tensor, W):
    Cy = tucker_tensor.core@tucker_tensor.factors[-1].T@W
    U, s, Vt = np.linalg.svd(Cy, full_matrices=False)
    Umu = tucker_tensor.factors[0]@U
    VmuT = tucker_tensor.factors[1]@Vt.T
    return Umu, s, VmuT


def m_ortho_basis_svd(X, M, k, tol=1e-12):
    """
    Build a k-dimensional M-orthonormal basis from snapshot data.

    Parameters
    ----------
    X : (m, n) array - snapshot matrix (columns are snapshots)
    M : (m, m) array or callable - SPD matrix, or a function v -> M @ v
    k : int - number of basis vectors
    tol : float - threshold for discarding near-zero singular values

    Returns
    -------
    Psi : (m, k) array - M-orthonormal basis vectors as columns
    sigma : (k,) array - corresponding singular values
    """
    apply_M = M if callable(M) else lambda v: M @ v

    n = X.shape[1]
    MX = np.column_stack([apply_M(X[:, j]) for j in range(n)])
    G = X.T @ MX

    # compute only the k largest eigenvalues/vectors
    eigvals, eigvecs = eigsh(G, k=k, which='LM')
    # eigsh returns ascending order; reverse to descending
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    valid = eigvals > tol
    eigvals = eigvals[valid]
    eigvecs = eigvecs[:, valid]

    sigma = np.sqrt(eigvals)
    Psi = X @ eigvecs / sigma[np.newaxis, :]

    return Psi, sigma

def m_orthonormalize_chol(V, M, tol=1e-12):
    """M-orthonormalize via Cholesky of Gram matrix"""
    G = V.T @ (M @ V)
    
    # Regularize if needed
    G += tol * np.trace(G) / G.shape[0] * np.eye(G.shape[0])
    
    L = np.linalg.cholesky(G)  # G = L L^T
    return solve_triangular(L, V.T, lower=True).T