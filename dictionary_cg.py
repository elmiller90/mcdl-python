import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def solve_dictionary_cg(
    op,
    S,
    G,
    H,
    rho,
    D0=None,
    rtol=1e-8,
    maxiter=500,
):
    """Solve (B*B + rho I)D = B*S + rho(G-H)."""

    S = np.asarray(S, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)

    shape = (op.M, op.K)

    if S.shape != (op.N, op.T):
        raise ValueError(
            f"S must have shape {(op.N, op.T)}"
        )
    if G.shape != shape or H.shape != shape:
        raise ValueError(
            f"G and H must have shape {shape}"
        )
    if rho <= 0:
        raise ValueError("rho must be positive")

    rhs = op.adjoint(S) + rho * (G - H)
    n = op.M * op.K

    def matvec(v):
        D = v.reshape(shape)
        return op.normal(D, rho).ravel()

    A = LinearOperator(
        (n, n),
        matvec=matvec,
        dtype=np.float64,
    )

    x0 = None
    if D0 is not None:
        D0 = np.asarray(D0, dtype=np.float64)
        if D0.shape != shape:
            raise ValueError(
                f"D0 must have shape {shape}"
            )
        x0 = D0.ravel()

    x, info = cg(
        A,
        rhs.ravel(),
        x0=x0,
        rtol=rtol,
        atol=0.0,
        maxiter=maxiter,
    )

    return x.reshape(shape), info