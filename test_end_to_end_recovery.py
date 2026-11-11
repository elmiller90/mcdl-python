import numpy as np

from multiscale_convolution import MultiscaleConvolutionOperator
from sparse_admm import solve_l1_admm


def make_dictionary(M):
    """
    Create one normalized Gaussian base atom.
    """
    x = np.arange(M, dtype=np.float64)

    center = (M - 1) / 2.0
    width = 4.0

    d = np.exp(
        -0.5 * ((x - center) / width) ** 2
    )

    d /= np.linalg.norm(d)

    return d[:, None]


def make_true_coefficients(N, K, J, T):
    """
    Create a known coefficient trajectory through
    position-scale space.
    """
    X_true = np.zeros(
        (N, K, J, T),
        dtype=np.float64,
    )

    positions = [36, 37, 38, 39, 40]
    scale_indices = [0, 1, 2, 1, 0]

    for t in range(T):
        n = positions[t]
        j = scale_indices[t]

        X_true[n, 0, j, t] = 1.0

    return X_true, positions, scale_indices


def trajectory_from_coefficients(X):
    """
    Find the largest coefficient in each frame.

    This test assumes K=1.
    """
    N, K, J, T = X.shape

    if K != 1:
        raise ValueError(
            "trajectory_from_coefficients assumes K=1"
        )

    positions = []
    scale_indices = []
    amplitudes = []

    for t in range(T):
        frame = X[:, 0, :, t]

        flat_index = int(
            np.argmax(np.abs(frame))
        )

        n, j = np.unravel_index(
            flat_index,
            frame.shape,
        )

        positions.append(int(n))
        scale_indices.append(int(j))
        amplitudes.append(
            float(X[n, 0, j, t])
        )

    return positions, scale_indices, amplitudes


def relative_error(x, y):
    """
    Relative L2/Frobenius error.
    """
    return np.linalg.norm(
        x - y
    ) / max(
        1e-15,
        np.linalg.norm(y),
    )


def count_significant(X, threshold=1e-3):
    """
    Number of coefficients whose magnitude exceeds
    a reporting threshold.
    """
    return int(
        np.count_nonzero(
            np.abs(X) > threshold
        )
    )


def recovery_test():
    rng = np.random.default_rng(50)

    # -------------------------------------------------
    # Problem dimensions
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Fixed dictionary and forward operator
    # -------------------------------------------------

    D = make_dictionary(M)

    operator = MultiscaleConvolutionOperator(
        D=D,
        scales=scales,
        center=center,
        N=N,
        normalize=True,
    )

    # -------------------------------------------------
    # Ground-truth sparse coefficients
    # -------------------------------------------------

    (
        X_true,
        true_positions,
        true_scales,
    ) = make_true_coefficients(
        N=N,
        K=K,
        J=J,
        T=T,
    )

    # -------------------------------------------------
    # Synthetic measurements
    # -------------------------------------------------

    S_clean = operator.forward(
        X_true
    )

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

    actual_noise = relative_error(
        S_noisy,
        S_clean,
    )

    # -------------------------------------------------
    # Initial coefficient estimate
    # -------------------------------------------------

    X0 = operator.adjoint(
        S_noisy,
        structured=True,
    )

    max_abs = np.max(
        np.abs(X0)
    )

    if max_abs > 0:
        X0 /= max_abs

    # -------------------------------------------------
    # Report setup
    # -------------------------------------------------

    print()
    print("End-to-end L1-ADMM recovery")
    print("---------------------------")
    print("X shape:", X_true.shape)
    print("S shape:", S_noisy.shape)

    print(
        f"relative noise level: "
        f"{actual_noise:.6e}"
    )

    print()
    print("True trajectory")
    print("---------------")
    print(
        "positions:",
        true_positions,
    )
    print(
        "scales:   ",
        true_scales,
    )

    print()
    print(
        "Initial significant coefficients:",
        count_significant(X0),
    )

    # -------------------------------------------------
    # ADMM parameters
    # -------------------------------------------------

    tau = 1e-2

    lambda2 = 0.1

    rho = 0.1

    lambda1_values = [
        0.001,
        0.003,
        0.01,
        0.03,
    ]

    lambda1_values = [0.03]

    # -------------------------------------------------
    # Run the L1 sweep
    # -------------------------------------------------

    print()
    print("L1-ADMM sweep")
    print("-------------")

    summary = []

    for lambda1 in lambda1_values:

        print()
        print("=" * 60)
        print(
            f"lambda1 = {lambda1:.4g}, "
            f"lambda2 = {lambda2:.4g}, "
            f"rho = {rho:.4g}"
        )
        print("=" * 60)

        (
            X_admm,
            Y_admm,
            U_admm,
            info,
        ) = solve_l1_admm(
            X0=X0,
            S=S_noisy,
            operator=operator,
            lambda1=lambda1,
            lambda2=lambda2,
            tau=tau,
            rho=rho,
            rho_mu=10.0,
            rho_tau=2.0,
            max_admm_iter=200,
            x_maxiter=50,
            abs_tol=1e-4,
            rel_tol=1e-3,
            nonnegative=True,
            adaptive_rho=True,
            verbose=True,
        )

        # Y is the sparse coefficient estimate.
        X_hat = Y_admm

        S_hat = operator.forward(
            X_hat
        )

        signal_error = relative_error(
            S_hat,
            S_clean,
        )

        noisy_signal_error = relative_error(
            S_hat,
            S_noisy,
        )

        coefficient_error = relative_error(
            X_hat,
            X_true,
        )

        (
            positions,
            scale_indices,
            amplitudes,
        ) = trajectory_from_coefficients(
            X_hat
        )

        significant = count_significant(
            X_hat
        )

        exact_zeros = (
            X_hat.size
            - np.count_nonzero(X_hat)
        )

        print(
            "converged:          ",
            info["converged"],
        )

        print(
            "ADMM iterations:    ",
            info["iterations"],
        )

        print(
            "final rho:          ",
            info["rho"],
        )

        print(
            "exact nonzeros:      ",
            info["nonzero_count"],
        )

        print(
            "significant (>1e-3):",
            significant,
        )

        print(
            "exact zeros:         ",
            exact_zeros,
        )

        print(
            f"signal error clean:  "
            f"{signal_error:.6e}"
        )

        print(
            f"signal error noisy:  "
            f"{noisy_signal_error:.6e}"
        )

        print(
            f"coefficient error:   "
            f"{coefficient_error:.6e}"
        )

        print(
            f"primal residual:     "
            f"{info['final_primal_residual']:.6e}"
        )

        print(
            f"dual residual:       "
            f"{info['final_dual_residual']:.6e}"
        )

        print(
            "positions:           ",
            positions,
        )

        print(
            "scales:              ",
            scale_indices,
        )

        print(
            "amplitudes:          ",
            [
                round(value, 4)
                for value in amplitudes
            ],
        )

        summary.append(
            {
                "lambda1": lambda1,
                "converged": info[
                    "converged"
                ],
                "iterations": info[
                    "iterations"
                ],
                "nonzeros": info[
                    "nonzero_count"
                ],
                "significant": significant,
                "signal_error": signal_error,
                "coefficient_error":
                    coefficient_error,
                "positions": positions,
                "scales": scale_indices,
            }
        )

    # -------------------------------------------------
    # Compact comparison
    # -------------------------------------------------

    print()
    print()
    print("Summary")
    print("-------")

    print(
        "lambda1   conv   iter   "
        "nz    sig    signal err    coeff err"
    )

    for row in summary:
        print(
            f"{row['lambda1']:7.3f}"
            f"   {str(row['converged']):5s}"
            f"   {row['iterations']:4d}"
            f"   {row['nonzeros']:4d}"
            f"   {row['significant']:4d}"
            f"   {row['signal_error']:10.4e}"
            f"   {row['coefficient_error']:10.4e}"
        )

    print()
    print("Trajectory comparison")
    print("---------------------")

    print(
        "true positions:",
        true_positions,
    )

    print(
        "true scales:   ",
        true_scales,
    )

    for row in summary:
        print()

        print(
            f"lambda1={row['lambda1']:.3f}"
        )

        print(
            "  positions:",
            row["positions"],
        )

        print(
            "  scales:   ",
            row["scales"],
        )


if __name__ == "__main__":
    recovery_test()
