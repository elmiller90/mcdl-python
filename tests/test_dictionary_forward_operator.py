import numpy as np

from dictionary_forward_operator import DictionaryForwardOperator


def test_dictionary_forward_linearity():
    rng = np.random.default_rng(0)

    N, M, K, J, T = 32, 9, 2, 3, 4
    L = N + M - 1

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

    D1 = rng.standard_normal((M, K))
    D2 = rng.standard_normal((M, K))

    a = 0.37
    b = -1.21

    lhs = op.forward(a * D1 + b * D2)
    rhs = a * op.forward(D1) + b * op.forward(D2)

    err = np.linalg.norm(lhs - rhs)
    ref = max(np.linalg.norm(lhs), 1e-15)
    relerr = err / ref

    print(f"dictionary forward relative error: {relerr:.3e}")

    assert relerr < 1e-12

def test_dictionary_adjoint():
    rng = np.random.default_rng(1)

    N, M, K, J, T = 32, 9, 2, 3, 4
    L = N + M - 1

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

    D = rng.standard_normal((M, K))
    R = rng.standard_normal((N, T))

    lhs = np.vdot(op.forward(D), R)
    rhs = np.vdot(D, op.adjoint(R))

    relerr = abs(lhs - rhs) / max(
        abs(lhs),
        abs(rhs),
        1e-15,
    )

    print(f"dictionary adjoint relative error: {relerr:.3e}")

    assert relerr < 1e-10
    
def test_dictionary_objective_gradient():
    rng = np.random.default_rng(2)
    N, M, K, J, T = 32, 9, 2, 3, 4
    L = N + M - 1
    scales = [(1, 1), (2, 1), (1, 2)]

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

    D = rng.standard_normal((M, K))
    S = rng.standard_normal((N, T))
    P = rng.standard_normal((M, K))
    P /= np.linalg.norm(P)

    def objective(A):
        R = op.forward(A) - S
        return 0.5 * np.sum(R * R)

    R = op.forward(D) - S
    grad = op.adjoint(R)
    exact = np.vdot(grad, P).real

    eps_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    errors = []

    for eps in eps_values:
        fp = objective(D + eps * P)
        fm = objective(D - eps * P)
        fd = (fp - fm) / (2.0 * eps)

        relerr = abs(fd - exact) / max(
            abs(fd), abs(exact), 1e-15
        )
        errors.append(relerr)

        print(
            f"eps={eps:.0e} "
            f"fd={fd:.8e} "
            f"exact={exact:.8e} "
            f"relerr={relerr:.3e}"
        )

    assert min(errors) < 1e-8\

def test_dictionary_normal_operator():
    rng = np.random.default_rng(3)

    N, M, K, J, T = 32, 9, 2, 3, 4
    L = N + M - 1
    rho = 0.7

    scales = [(1, 1), (2, 1), (1, 2)]

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

    D1 = rng.standard_normal((M, K))
    D2 = rng.standard_normal((M, K))

    ND1 = op.normal(D1, rho)
    ND2 = op.normal(D2, rho)

    lhs = np.vdot(D1, ND2).real
    rhs = np.vdot(ND1, D2).real

    symerr = abs(lhs - rhs) / max(
        abs(lhs), abs(rhs), 1e-15
    )

    quad = np.vdot(D1, ND1).real
    lower_bound = rho * np.linalg.norm(D1) ** 2

    print(
        f"normal symmetry error: {symerr:.3e}"
    )
    print(
        f"quadratic form: {quad:.8e}"
    )
    print(
        f"rho lower bound: {lower_bound:.8e}"
    )

    assert symerr < 1e-10
    assert quad > 0.0
    assert quad >= lower_bound - 1e-10