import numpy as np

from dictionary_projection import project_dictionary_nonnegative


def make_banco_boxes(M, K):
    """MATLAB-compatible initXD(..., dictInit='flat') dictionary."""
    D = np.zeros((M, K), dtype=float)

    for k in range(1, K + 1):
        i1 = int(np.floor(M * (0.5 - 0.5 * k / K) + 1 + 0.5))
        i2 = int(np.floor(M * (0.5 + 0.5 * k / K) + 0.5))

        # MATLAB i1:i2 -> zero-based Python [i1-1:i2].
        D[i1 - 1:i2, k - 1] = 1.0

    return project_dictionary_nonnegative(D)