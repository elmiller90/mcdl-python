import numpy as np


def solvedbi_sm_rank1(a, rho, b):
    """
    Solve (rho I + a^H a)x = b.

    a, b: (..., F, T) complex arrays
    channel axis is -2.
    """
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)

    if a.shape[:-1] != b.shape[:-1]:
        raise ValueError("a and b channel dimensions must match")
    if rho <= 0:
        raise ValueError("rho must be positive")

    # A(x) = sum_f a_f x_f
    ab = np.sum(a*b, axis=-2, keepdims=True)

    # ||a||^2 = sum_f |a_f|^2
    aa = np.sum(
        np.abs(a)**2,
        axis=-2,
        keepdims=True,
    )

    return (
        b/rho
        - np.conj(a)*ab/(rho*(rho+aa))
    )