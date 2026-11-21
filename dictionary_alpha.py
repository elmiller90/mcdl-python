import numpy as np

from multiscale_operator import resample_atom


def compute_dictionary_alpha(
    G,
    scales,
    center,
    N,
    zero_tol=1e-15,
):
    """Compute NormVals and alpha from projected dictionary G."""
    G = np.asarray(G, dtype=np.float64)

    if G.ndim != 2:
        raise ValueError("G must have shape (M,K)")

    M, K = G.shape
    J = len(scales)

    norm_vals = np.zeros((K, J), dtype=np.float64)
    alpha = np.zeros((K, J), dtype=np.float64)

    for k in range(K):
        for j, (c1, c2) in enumerate(scales):
            atom, _ = resample_atom(
                G[:, k],
                c1,
                c2,
                center,
                N,
            )

            norm_val = np.linalg.norm(atom)
            norm_vals[k, j] = norm_val

            if norm_val > zero_tol:
                alpha[k, j] = 1.0 / norm_val

    return alpha, norm_vals