import numpy as np

from multiscale_convolution import MultiscaleConvolutionOperator
from x_update import x_objective


def make_dictionary(M, K):
    x = np.arange(M, dtype=np.float64)
    D = np.zeros((M, K))

    centers = [24.0, 46.0]
    widths = [4.0, 7.0]

    for k in range(K):
        D[:, k] = np.exp(
            -0.5 * ((x - centers[k]) / widths[k]) ** 2
        )
        D[:, k] /= np.linalg.norm(D[:, k])

    return D


def gradient_test():
    rng = np.random.default_rng(40)

    # Problem dimensions
    M = 71
    N = 201
    K = 2
    T = 4
    center = 36

    scales = [
        (1, 1),
        (1, 2),
        (2, 3),
        (3, 2),
        (2, 1),
    ]

    J = len(scales)

    # Fixed multiscale dictionary
    D = make_dictionary(M, K)

    operator = MultiscaleConvolutionOperator(
        D=D,
        scales=scales,
        center=center,
        N=N,
        normalize=True,
    )

    # Canonical layout:
    #
    # X, Y, U : (N, K, J, T)
    # S       : (N, T)
    shape = (N, K, J, T)

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
        size=(N, T),
    )

    rho = 1.3
    lambda2 = 0.7
    tau = 1e-2

    # Analytical objective and gradient
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

    # Random unit direction
    V = rng.normal(size=shape)
    V /= np.linalg.norm(V)

    analytic = np.dot(
        grad,
        V.ravel(),
    )

    print()
    print("Composite X-objective gradient test")
    print("-----------------------------------")
    print(f"X shape = {X.shape}")
    print(f"S shape = {S.shape}")
    print(f"f = {f:.12e}")
    print(f"analytic = {analytic:.12e}")

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

        fd = (fp - fm) / (2.0 * eps)

        error = abs(fd - analytic) / max(
            1.0,
            abs(fd),
            abs(analytic),
        )

        print(
            f"{eps:1.0e}    "
            f"{fd: .12e}    "
            f"{error:.3e}"
        )


if __name__ == "__main__":
    gradient_test()
