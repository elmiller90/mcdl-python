import numpy as np

from multiscale_convolution import (
    MultiscaleConvolutionOperator,
)
from x_update import (
    objective_components,
    solve_x,
)


def make_dictionary(M):
    """
    Create one normalized Gaussian base atom.
    """
    x = np.arange(M, dtype=np.float64)

    atom_center = (M - 1) / 2.0
    atom_width = 4.0

    d = np.exp(
        -0.5
        * ((x - atom_center) / atom_width) ** 2
    )

    d /= np.linalg.norm(d)

    return d[:, None]


def make_true_coefficients(N, K, J, T):
    """
    Create a coefficient peak following an allowed diagonal
    trajectory through position-scale space.
    """
    X_true = np.zeros(
        (N, K, J, T),
        dtype=np.float64,
    )

    positions = [36, 37, 38, 39, 40]
    scales = [0, 1, 2, 1, 0]

    for t in range(T):
        n = positions[t]
        j = scales[t]

        X_true[n, 0, j, t] = 1.0

    return X_true, positions, scales


def trajectory_from_coefficients(X):
    """
    Return the largest-magnitude coefficient location
    (position, scale) in each frame.
    """
    N, K, J, T = X.shape

    if K != 1:
        raise ValueError(
            "This helper assumes K=1"
        )

    positions = []
    scales = []
    amplitudes = []

    for t in range(T):
        frame = np.abs(X[:, 0, :, t])

        flat_index = int(np.argmax(frame))

        n, j = np.unravel_index(
            flat_index,
            frame.shape,
        )

        positions.append(int(n))
        scales.append(int(j))
        amplitudes.append(
            float(X[n, 0, j, t])
        )

    return positions, scales, amplitudes


def relative_error(x, y):
    return np.linalg.norm(x - y) / max(
        1e-15,
        np.linalg.norm(y),
    )


def recovery_test():
    rng = np.random.default_rng(50)

    # Problem dimensions
    M = 41
    N = 96
    K = 1
    T = 5
    center = 21

    scales = [
        (1, 2),
        (1, 1),
        (2, 1),
    ]

    J = len(scales)

    D = make_dictionary(M)

    operator = MultiscaleConvolutionOperator(
        D=D,
        scales=scales,
        center=center,
        N=N,
        normalize=True,
    )

    X_true, true_positions, true_scales = (
        make_true_coefficients(
            N=N,
            K=K,
            J=J,
            T=T,
        )
    )

    # Generate noiseless measurements.
    S_clean = operator.forward(X_true)

    # Add Gaussian measurement noise.
    noise_level = 0.02

    noise = rng.normal(
        size=S_clean.shape
    )

    noise *= (
        noise_level
        * np.linalg.norm(S_clean)
        / np.linalg.norm(noise)
    )

    S_noisy = S_clean + noise

    measured_noise = relative_error(
        S_noisy,
        S_clean,
    )

    # Start from the adjoint/backprojection.
    X0 = operator.adjoint(
        S_noisy,
        structured=True,
    )

    # Scale the initialization to avoid beginning with
    # excessively large correlated coefficients.
    max_abs = np.max(np.abs(X0))

    if max_abs > 0:
        X0 = X0 / max_abs

    # This is still an isolated coefficient-subproblem test.
    # Y=0 and U=0 make the ADMM term a small L2 penalty.
    Y = np.zeros_like(X0)
    U = np.zeros_like(X0)

    rho = 1e-3
    tau = 1e-2

    print()
    print("End-to-end coefficient recovery")
    print("-------------------------------")
    print("X shape:", X_true.shape)
    print("S shape:", S_noisy.shape)
    print(f"relative noise level: {measured_noise:.6e}")

    print()
    print(
        "lambda2    success    iterations    "
        "signal error    coefficient error"
    )

    results = {}

    for lambda2 in [0.0, 0.1, 1.0]:
        X_hat, result = solve_x(
            X0=X0,
            S=S_noisy,
            Y=Y,
            U=U,
            operator=operator,
            rho=rho,
            lambda2=lambda2,
            tau=tau,
            maxiter=300,
            ftol=1e-12,
            gtol=1e-8,
            maxls=50,
        )

        S_hat = operator.forward(X_hat)

        signal_error = relative_error(
            S_hat,
            S_clean,
        )

        coefficient_error = relative_error(
            X_hat,
            X_true,
        )

        results[lambda2] = {
            "X": X_hat,
            "result": result,
            "signal_error": signal_error,
            "coefficient_error": coefficient_error,
        }

        print(
            f"{lambda2:7.2f}"
            f"    {str(result.success):7s}"
            f"    {result.nit:10d}"
            f"    {signal_error:12.6e}"
            f"    {coefficient_error:17.6e}"
        )

    print()
    print("True trajectory")
    print("---------------")
    print("positions:", true_positions)
    print("scales:   ", true_scales)

    for lambda2 in [0.0, 0.1, 1.0]:
        X_hat = results[lambda2]["X"]

        positions, scale_indices, amplitudes = (
            trajectory_from_coefficients(X_hat)
        )

        print()
        print(f"Recovered trajectory, lambda2={lambda2}")
        print("-" * 36)
        print("positions: ", positions)
        print("scales:    ", scale_indices)
        print(
            "amplitudes: ",
            [
                round(value, 4)
                for value in amplitudes
            ],
        )

        components = objective_components(
            X=X_hat,
            S=S_noisy,
            Y=Y,
            U=U,
            operator=operator,
            rho=rho,
            lambda2=lambda2,
            tau=tau,
        )

        print("objective components:")
        print(
            f"  data:             "
            f"{components['data']:.6e}"
        )
        print(
            f"  ADMM quadratic:   "
            f"{components['admm']:.6e}"
        )
        print(
            f"  raw softmin:      "
            f"{components['softmin']:.6e}"
        )
        print(
            f"  weighted softmin: "
            f"{components['weighted_softmin']:.6e}"
        )
        print(
            f"  total:            "
            f"{components['total']:.6e}"
        )


if __name__ == "__main__":
    recovery_test()
    