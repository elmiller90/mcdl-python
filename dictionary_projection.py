import numpy as np


def project_dictionary_nonnegative(G, zero_tol=1e-15):
    """Project base dictionary onto nonnegative unit-norm atoms."""
    G = np.asarray(G, dtype=np.float64)

    if G.ndim != 2:
        raise ValueError("G must have shape (M,K)")

    P = np.maximum(G, 0.0)
    norms = np.linalg.norm(P, axis=0)

    nonzero = norms > zero_tol
    P[:, nonzero] /= norms[nonzero]
    P[:, ~nonzero] = 0.0

    return P
