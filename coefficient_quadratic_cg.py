import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def solve_coefficient_cg(
    operator, S, Y, U, rho,
    X0=None, rtol=1e-10, maxiter=500,
):
    """Solve (A* A + rho I)X = A* S + rho(Y-U)."""
    S = np.asarray(S, dtype=float)
    Y = np.asarray(Y, dtype=float)
    U = np.asarray(U, dtype=float)

    if Y.shape != U.shape:
        raise ValueError("Y and U must have the same shape")
    if rho <= 0:
        raise ValueError("rho must be positive")

    shape = Y.shape
    rhs = (
        operator.adjoint(S, structured=True)
        + rho*(Y-U)
    )
    n = Y.size

    def matvec(v):
        X = v.reshape(shape)
        AX = operator.forward(X)
        AtAX = operator.adjoint(
            AX, structured=True
        )
        return (AtAX + rho*X).ravel()

    Aop = LinearOperator(
        (n, n),
        matvec=matvec,
        dtype=np.float64,
    )

    x0 = None
    if X0 is not None:
        X0 = np.asarray(X0, dtype=float)
        if X0.shape != shape:
            raise ValueError(
                f"X0 must have shape {shape}"
            )
        x0 = X0.ravel()

    x, info = cg(
        Aop,
        rhs.ravel(),
        x0=x0,
        rtol=rtol,
        atol=0.0,
        maxiter=maxiter,
    )

    return x.reshape(shape), info