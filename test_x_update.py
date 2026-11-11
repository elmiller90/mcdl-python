import numpy as np

from x_update import x_objective, solve_x


def gradient_test():

    rng = np.random.default_rng(2)

    K = 1
    J = 4
    N = 12
    T = 4

    shape = (1, N, K * J, T)

    X = rng.normal(size=shape)
    S = rng.normal(size=shape)
    Y = rng.normal(size=shape)
    U = rng.normal(size=shape)

    rho = 1.3
    lambda2 = 0.7
    tau = 1e-2

    f, grad = x_objective(
        X.ravel(),
        shape,
        S,
        Y,
        U,
        K,
        J,
        rho,
        lambda2,
        tau,
    )

    V = rng.normal(size=shape)
    V /= np.linalg.norm(V)

    analytic = np.dot(
        grad,
        V.ravel(),
    )

    print("X-objective gradient test")
    print("-------------------------")
    print(f"f = {f:.12e}")
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
    ]:

        fp, _ = x_objective(
            (X + eps * V).ravel(),
            shape,
            S,
            Y,
            U,
            K,
            J,
            rho,
            lambda2,
            tau,
        )

        fm, _ = x_objective(
            (X - eps * V).ravel(),
            shape,
            S,
            Y,
            U,
            K,
            J,
            rho,
            lambda2,
            tau,
        )

        fd = (fp - fm) / (2 * eps)

        error = abs(
            fd - analytic
        ) / max(
            1.0,
            abs(fd),
            abs(analytic),
        )

        print(
            f"{eps:1.0e}    "
            f"{fd: .12e}    "
            f"{error:.3e}"
        )

def optimization_test():

    rng = np.random.default_rng(3)

    K = 1
    J = 5
    N = 25
    T = 6
    shape = (1, N, K * J, T)

    # Desired data: mostly coherent trajectory,
    # with one deliberately displaced observation.
    S = np.zeros(shape)

    j = 2
    n0 = 6

    for t in range(T):
        n = n0 + t

        if t == 3:
            n += 3

        S[0, n, j, t] = 1.0

    # Start optimization from observations.
    X0 = S.copy()

    # For this isolated experiment, remove the influence
    # of the ADMM target by setting rho very small.
    Y = np.zeros_like(X0)
    U = np.zeros_like(X0)

    X, result = solve_x(
        X0,
        S,
        Y,
        U,
        K,
        J,
        rho=1e-6,
        lambda2=1.0,
        tau=1e-2,
        maxiter=200,
    )

    print()
    print("L-BFGS optimization test")
    print("------------------------")
    print("success:", result.success)
    print("message:", result.message)
    print("iterations:", result.nit)
    print("function evaluations:", result.nfev)
    print("initial objective:",
          x_objective(
              X0.ravel(),
              shape, S, Y, U,
              K, J,
              1e-6, 1.0, 1e-2
          )[0])
    print("final objective:", result.fun)

    print()
    print("Frame  observed   recovered")

    for t in range(T):
        observed = np.argmax(S[0, :, j, t])
        recovered = np.argmax(X[0, :, j, t])

        print(
            f"{t:5d}"
            f"{observed:10d}"
            f"{recovered:12d}"
        )

def lambda_sweep_test():

    K = 1
    J = 5
    N = 25
    T = 6
    shape = (1, N, K * J, T)

    S = np.zeros(shape)

    j = 2
    n0 = 6

    for t in range(T):
        n = n0 + t

        if t == 3:
            n += 3

        S[0, n, j, t] = 1.0

    X0 = S.copy()
    Y = np.zeros_like(X0)
    U = np.zeros_like(X0)

    lambda_values = [
        0.0,
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
        3.0,
    ]

    print()
    print("lambda2 sweep")
    print("-------------")
    print(
        "lambda2    objective     iterations    trajectory"
    )

    for lambda2 in lambda_values:

        X, result = solve_x(
            X0,
            S,
            Y,
            U,
            K,
            J,
            rho=1e-6,
            lambda2=lambda2,
            tau=1e-2,
            maxiter=200,
        )

        trajectory = [
            int(np.argmax(X[0, :, j, t]))
            for t in range(T)
        ]

        print(
            f"{lambda2:7.2f}"
            f"    {result.fun:10.6f}"
            f"    {result.nit:10d}"
            f"    {trajectory}"
        )

if __name__ == "__main__":
    # optimization_test()
    # gradient_test()
    lambda_sweep_test()