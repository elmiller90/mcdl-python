import numpy as np

from dictionary_forward_operator import DictionaryForwardOperator
from dictionary_cg import solve_dictionary_cg


def test_dictionary_cg_against_direct_solve():
    rng = np.random.default_rng(4)

    N, M, K, J, T = 32, 9, 2, 3, 4
    L = N + M - 1
    rho = 0.7

    scales = [
        (1, 1),
        (2, 1),
        (1, 2),
    ]

    X = rng.standard_normal((L, K, J, T))
    alpha = 0.5 + rng.random((K, J))

    op = DictionaryForwardOperator(
        X=X,
        alpha=alpha,
        scales=scales,
        center=(M + 1) // 2,
        N=N,
        M=M,
    )

    S = rng.standard_normal((N, T))
    G = rng.standard_normal((M, K))
    H = rng.standard_normal((M, K))

    Dcg, info = solve_dictionary_cg(
        op=op,
        S=S,
        G=G,
        H=H,
        rho=rho,
        rtol=1e-11,
        maxiter=500,
    )

    n = M * K
    A = np.empty((n, n))

    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0

        A[:, i] = op.normal(
            e.reshape(M, K),
            rho,
        ).ravel()

    rhs = (
        op.adjoint(S)
        + rho * (G - H)
    ).ravel()

    Ddirect = np.linalg.solve(
        A,
        rhs,
    ).reshape(M, K)

    relerr = (
        np.linalg.norm(Dcg - Ddirect)
        / np.linalg.norm(Ddirect)
    )

    residual = (
        np.linalg.norm(
            op.normal(Dcg, rho).ravel()
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