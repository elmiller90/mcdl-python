import numpy as np

from coefficient_quadratic_cg import solve_coefficient_cg
from multiscale_convolution import MultiscaleConvolutionOperator


def test_coefficient_cg_against_direct_solve():
    rng = np.random.default_rng(50)

    N, M, K, T = 16, 5, 1, 2
    scales = [(1, 1)]
    center = (M+1)//2

    d = rng.random((M, K))
    d /= np.linalg.norm(d, axis=0)

    A = MultiscaleConvolutionOperator(
        D=d, scales=scales,
        center=center, N=N,
        normalize=True,
    )

    shape = (A.L, K, 1, T)
    Y = rng.standard_normal(shape)
    U = rng.standard_normal(shape)
    S = rng.standard_normal((N, T))
    rho = 0.7

    Xcg, info = solve_coefficient_cg(
        A, S, Y, U, rho,
        rtol=1e-11, maxiter=500,
    )

    n = np.prod(shape)
    H = np.empty((n, n))

    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        X = e.reshape(shape)
        H[:, i] = (
            A.adjoint(
                A.forward(X),
                structured=True,
            )
            + rho*X
        ).ravel()

    rhs = (
        A.adjoint(S, structured=True)
        + rho*(Y-U)
    ).ravel()

    Xdirect = np.linalg.solve(
        H, rhs
    ).reshape(shape)

    relerr = (
        np.linalg.norm(Xcg-Xdirect)
        / np.linalg.norm(Xdirect)
    )

    residual = (
        np.linalg.norm(
            (
                A.adjoint(
                    A.forward(Xcg),
                    structured=True,
                )
                + rho*Xcg
            ).ravel()
            - rhs
        )
        / np.linalg.norm(rhs)
    )

    print(f"CG info: {info}")
    print(
        f"CG/direct relative error: "
        f"{relerr:.3e}"
    )
    print(
        f"CG relative residual: "
        f"{residual:.3e}"
    )

    assert info == 0
    assert relerr < 1e-9
    assert residual < 1e-9
