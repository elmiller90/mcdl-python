import numpy as np

from multiscale_convolution import (
    MultiscaleConvolutionOperator,
)
from x_update import x_objective


def make_dictionary(M, K):
    """
    Construct smooth normalized base dictionary atoms.
    """
    x = np.arange(
        M,
        dtype=np.float64,
    )

    D = np.zeros(
        (M, K),
        dtype=np.float64,
    )

    centers = [24.0, 46.0]
    widths = [4.0, 7.0]

    for k in range(K):
        D[:, k] = np.exp(
            -0.5
            * (
                (x - centers[k])
                / widths[k]
            ) ** 2
        )

        D[:, k] /= np.linalg.norm(
            D[:, k]
        )

    return D


def gradient_test():
    rng = np.random.default_rng(80)

    # -------------------------------------------------
    # Problem dimensions
    # -------------------------------------------------

    M = 71
    N = 201
    K = 2
    T = 4
    center = 36

    # Paper-compatible contraction scales.
    scales = [
        (1, 1),
        (4, 5),
        (3, 4),
        (2, 3),
        (1, 2),
    ]

    J = len(scales)

    # -------------------------------------------------
    # Fixed multiscale dictionary operator
    # -------------------------------------------------

    D = make_dictionary(
        M,
        K,
    )

    operator = MultiscaleConvolutionOperator(
        D=D,
        scales=scales,
        center=center,
        N=N,
        normalize=True,
    )

    # Extended MATLAB coefficient domain.
    L = operator.L

    expected_L = N + M - 1

    if L != expected_L:
        raise RuntimeError(
            f"Expected L={expected_L}, "
            f"got {L}"
        )

    # -------------------------------------------------
    # Random test problem
    #
    # X,Y,U : (L,K,J,T)
    # S     : (N,T)
    # -------------------------------------------------

    shape = (
        L,
        K,
        J,
        T,
    )

    X = rng.normal(
        scale=0.1,
        size=shape,
    )

    Y = rng.normal(
        scale=0.1,
        size=shape,
    )

    U = rng.normal(
        scale=0.1,
        size=shape,
    )

    S = rng.normal(
        scale=0.1,
        size=(
            N,
            T,
        ),
    )

    rho = 1.3
    lambda2 = 0.7
    tau = 1e-2

    # -------------------------------------------------
    # Analytical objective and gradient
    # -------------------------------------------------

    f, grad = x_objective(
        X.ravel(),
        shape,
        S,
        Y,
        U,
        operator,
        rho,
        lambda2,
        tau,
    )

    # -------------------------------------------------
    # Random unit-norm direction
    # -------------------------------------------------

    V = rng.normal(
        size=shape
    )

    V /= np.linalg.norm(V)

    analytic = np.dot(
        grad,
        V.ravel(),
    )

    # -------------------------------------------------
    # Report dimensions
    # -------------------------------------------------

    print()
    print("L-domain X-objective gradient test")
    print("----------------------------------")

    print(f"M = {M}")
    print(f"N = {N}")
    print(f"L = {L}")
    print(f"K = {K}")
    print(f"J = {J}")
    print(f"T = {T}")

    print()

    print(
        "X shape =",
        X.shape,
    )

    print(
        "S shape =",
        S.shape,
    )

    AX = operator.forward(X)

    print(
        "AX shape =",
        AX.shape,
    )

    AtS = operator.adjoint(S)

    print(
        "A*S shape =",
        AtS.shape,
    )

    print()

    print(
        f"f = {f:.12e}"
    )

    print(
        f"analytic = "
        f"{analytic:.12e}"
    )

    # -------------------------------------------------
    # Central finite-difference test
    # -------------------------------------------------

    print()
    print(
        "eps          finite-difference       "
        "relative error"
    )

    eps_values = [
        1e-2,
        1e-3,
        1e-4,
        1e-5,
        1e-6,
    ]

    for eps in eps_values:
        Xp = X + eps * V
        Xm = X - eps * V

        fp, _ = x_objective(
            Xp.ravel(),
            shape,
            S,
            Y,
            U,
            operator,
            rho,
            lambda2,
            tau,
        )

        fm, _ = x_objective(
            Xm.ravel(),
            shape,
            S,
            Y,
            U,
            operator,
            rho,
            lambda2,
            tau,
        )

        finite_difference = (
            fp - fm
        ) / (
            2.0 * eps
        )

        relative_error = abs(
            finite_difference
            - analytic
        ) / max(
            1.0,
            abs(finite_difference),
            abs(analytic),
        )

        print(
            f"{eps:1.0e}    "
            f"{finite_difference: .12e}    "
            f"{relative_error:.3e}"
        )


if __name__ == "__main__":
    gradient_test()
