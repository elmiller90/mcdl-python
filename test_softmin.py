import numpy as np

from softmin import softmin


def directional_derivative_test(
        K=2,
        J=4,
        N=17,
        T=5,
        tau=1e-2,
        seed=1):

    rng = np.random.default_rng(seed)

    X = rng.normal(
        size=(1, N, K * J, T)
    )

    V = rng.normal(size=X.shape)
    V /= np.linalg.norm(V)

    R, grad = softmin(
        X,
        K,
        J,
        tau=tau,
        pad_mode="zero",
        normalize=False
    )

    analytic = np.sum(grad * V)

    print(f"R = {R:.12e}")
    print(f"analytic = {analytic:.12e}")
    print()
    print(
        "eps          finite-difference       "
        "relative error"
    )

    for eps in [
        1e-2,
        1e-3,
        1e-4,
        1e-5,
        1e-6,
        1e-7,
    ]:

        Rp = softmin(
            X + eps * V,
            K,
            J,
            tau=tau,
            return_gradient=False
        )

        Rm = softmin(
            X - eps * V,
            K,
            J,
            tau=tau,
            return_gradient=False
        )

        fd = (Rp - Rm) / (2 * eps)

        relerr = abs(
            fd - analytic
        ) / max(
            1.0,
            abs(fd),
            abs(analytic)
        )

        print(
            f"{eps:1.0e}    "
            f"{fd: .12e}    "
            f"{relerr:.3e}"
        )

def basic_tests():
    # Single atom, 3 scales, 8 positions, 3 times.
    K = 1
    J = 3
    N = 8
    T = 3

    X = np.zeros((1, N, K * J, T))

    # One stationary coefficient.
    X[0, 4, 1, 0] = 1.0
    X[0, 4, 1, 1] = 1.0
    X[0, 4, 1, 2] = 1.0

    R, g = softmin(
        X,
        K,
        J,
        tau=1e-2
    )

    print()
    print("Basic tests")
    print("-----------")
    print("R:", R)
    print("gradient norm:", np.linalg.norm(g))

    assert np.isfinite(R)
    assert np.all(np.isfinite(g))
    assert g.shape == X.shape

    print("PASS")        

def moving_peak_test():
    K = 1
    J = 5
    N = 20
    T = 6
    tau = 1e-2

    X = np.zeros((1, N, K * J, T))

    j = 2
    n0 = 5

    for t in range(T):
        X[0, n0 + t, j, t] = 1.0

    R, grad = softmin(
        X,
        K,
        J,
        tau=tau,
        pad_mode="zero",
        normalize=False,
        debug=False,
    )

    print()
    print("Moving peak test")
    print("----------------")
    print("R:", R)
    print("gradient norm:", np.linalg.norm(grad))

def jumping_peak_test():
    K = 1
    J = 5
    N = 20
    T = 6
    tau = 1e-2

    X = np.zeros((1, N, K * J, T))

    j = 2
    n0 = 2

    for t in range(T):
        X[0, n0 + 3 * t, j, t] = 1.0

    R, grad = softmin(
        X,
        K,
        J,
        tau=tau,
        pad_mode="zero",
        normalize=False,
    )

    print()
    print("Jumping peak test")
    print("-----------------")
    print("R:", R)
    print("gradient norm:", np.linalg.norm(grad))

def diagonal_peak_test():
    K = 1
    J = 6
    N = 20
    T = 4
    tau = 1e-2

    X = np.zeros((1, N, K * J, T))

    j0 = 1
    n0 = 5

    for t in range(T):
        X[0, n0 + t, j0 + t, t] = 1.0

    R, grad = softmin(
        X,
        K,
        J,
        tau=tau,
        pad_mode="zero",
        normalize=False,
        debug=True,
    )

    print()
    print("Diagonal peak test")
    print("------------------")
    print("R:", R)
    print("gradient norm:", np.linalg.norm(grad))


if __name__ == "__main__":
    basic_tests()
    moving_peak_test()
    jumping_peak_test()
    diagonal_peak_test()    
    directional_derivative_test()