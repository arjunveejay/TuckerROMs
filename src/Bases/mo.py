import numpy as np

def mo(training_params, K, mu_test, *, eps=1e-12, rcond=1e-12, match_tol=0.0):
    """
    Compute the weight vector e(mu) as described:
      - pick K nearest neighbors of mu among training params
      - D = diag(1 / ||mu - mu_i||)
      - Sbar = [mu_i; 1] columns (shape (d+1, K))
      - c = [mu; 1]
      - a_hat = D @ pinv(Sbar @ D) @ c
      - e is length P with e[i_k] = a_hat[k], others 0

    Special case:
      - if mu matches a training point within match_tol, return a one-hot vector on that index.

    Args:
        training_params: array of shape (P, d) (rows are samples) OR (d, P) (cols are samples).
        K: number of nearest neighbors (should satisfy K > d).
        mu_test: array of shape (d,).
        eps: small number to avoid division by zero in distances (used only when not exact-matching).
        rcond: cutoff for pseudoinverse.
        match_tol: if min distance <= match_tol, return one-hot (exact/near match).

    Returns:
        e: weight vector of shape (P,).
    """
    S = np.asarray(training_params, dtype=float)
    mu = np.asarray(mu_test, dtype=float).reshape(-1)
    if S.ndim != 2:
        raise ValueError("training_params must be a 2D array.")
    d = mu.size

    # Accept either (P,d) or (d,P)
    if S.shape[1] == d:
        points = S              # (P, d)
    elif S.shape[0] == d:
        points = S.T            # (P, d)
    else:
        raise ValueError(f"training_params must have a dimension equal to d={d}.")

    P = points.shape[0]
    if not (1 <= K <= P):
        raise ValueError(f"K must be between 1 and P={P}. Got K={K}.")
    if K <= d:
        raise ValueError(f"Expected K > d (K={K}, d={d}).")

    # Distances to all training points
    dist = np.linalg.norm(points - mu[None, :], axis=1)

    # Exact (or near) match -> one-hot
    j = int(np.argmin(dist))
    if dist[j] <= match_tol:
        e = np.zeros(P, dtype=float)
        e[j] = 1.0
        return e

    # K nearest neighbors (ordered)
    nn_idx = np.argpartition(dist, K - 1)[:K]
    nn_idx = nn_idx[np.argsort(dist[nn_idx])]

    dvec = np.maximum(dist[nn_idx], eps)
    D = np.diag(1.0 / dvec)  # (K, K)

    # Build Sbar with columns [mu_i; 1]
    neigh = points[nn_idx].T                    # (d, K)
    Sbar = np.vstack([neigh, np.ones((1, K))])  # (d+1, K)
    c = np.concatenate([mu, [1.0]])             # (d+1,)

    SD = Sbar @ D
    a_hat = D @ (np.linalg.pinv(SD, rcond=rcond) @ c)  # (K,)

    e = np.zeros(P, dtype=float)
    e[nn_idx] = a_hat
    return e