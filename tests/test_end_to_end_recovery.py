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


def physical_to_coefficient_position(n, M):
    """
    Convert a physical signal position to the MATLAB
    extended coefficient coordinate.

        n_coeff = (M - 1) + n_physical
    """
    return (M - 1) + n


def coefficient_to_physical_position(n, M):
    """
    Convert MATLAB extended coefficient coordinate
    back to physical signal position.
    """
    return n - (M - 1)


def make_true_coefficients(
    L,
    K,
    J,
    T,
    M,
):
    """
    Construct a known sparse trajectory in the extended
    MATLAB coefficient domain.
    """
    X_true = np.zeros(
        (L, K, J, T),
        dtype=np.float64,
    )

    physical_positions = [
        36,
        37,
        38,
        39,
        40,
    ]

    scale_indices = [
        0,
        1,
        2,
        1,
        0,
    ]

    coefficient_positions = [
        physical_to_coefficient_position(
            n,
            M,
        )
        for n in physical_positions
    ]

    for t in range(T):
        n = coefficient_positions[t]
        j = scale_indices[t]

        X_true[
            n,
            0,
            j,
            t,
        ] = 1.0

    return (
        X_true,
        physical_positions,
        coefficient_positions,
        scale_indices,
    )


def trajectory_from_coefficients(
    X,
    M,
):
    """
    Extract the dominant coefficient in each frame.

    Returns both L-domain coefficient positions and the
    corresponding physical signal positions.

    Assumes K=1.
    """
    L, K, J, T = X.shape

    if K != 1:
        raise ValueError(
            "This trajectory helper assumes K=1"
        )

    coefficient_positions = []
    physical_positions = []
    scale_indices = []
    amplitudes = []

    for t in range(T):
        frame = np.abs(
            X[:, 0, :, t]
        )

        flat_index = int(
            np.argmax(frame)
        )

        n_coeff, j = np.unravel_index(
            flat_index,
            frame.shape,
        )

        n_physical = (
            coefficient_to_physical_position(
                n_coeff,
                M,
            )
        )

        coefficient_positions.append(
            int(n_coeff)
        )

        physical_positions.append(
            int(n_physical)
        )

        scale_indices.append(
            int(j)
        )

        amplitudes.append(
            float(
                X[
                    n_coeff,
                    0,
                    j,
                    t,
                ]
            )
        )

    return (
        coefficient_positions,
        physical_positions,
        scale_indices,
        amplitudes,
    )


def relative_error(x, y):
    """
    Relative Frobenius/L2 error.
    """
    return np.linalg.norm(
        x - y
    ) / max(
        1e-15,
        np.linalg.norm(y),
    )


def count_significant(
    X,
    threshold=1e-3,
):
    """
    Count coefficients above a reporting threshold.
    """
    return int(
        np.count_nonzero(
            np.abs(X) > threshold
        )
    )


def recovery_test():
    rng = np.random.default_rng(90)

    # -------------------------------------------------
    # MATLAB-compatible dimensions
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
    # Fixed multiscale dictionary operator
    # -------------------------------------------------

    D = make_dictionary(M)

    operator = MultiscaleConvolutionOperator(
        D=D,
        scales=scales,
        center=center,
        N=N,
        normalize=True,
    )

    L = operator.L

    expected_L = N + M - 1

    if L != expected_L:
        raise RuntimeError(
            f"Expected L={expected_L}, got {L}"
        )

    # -------------------------------------------------
    # Ground-truth L-domain coefficients
    # -------------------------------------------------

    (
        X_true,
        true_physical_positions,
        true_coefficient_positions,
        true_scales,
    ) = make_true_coefficients(
        L=L,
        K=K,
        J=J,
        T=T,
        M=M,
    )

    # -------------------------------------------------
    # Generate physical N-domain measurements
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

    S_noisy = (
        S_clean
        + noise
    )

    actual_noise = relative_error(
        S_noisy,
        S_clean,
    )

    # -------------------------------------------------
    # L-domain adjoint initialization
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
    # Report geometry
    # -------------------------------------------------

    print()
    print("L-domain end-to-end L1-ADMM recovery")
    print("------------------------------------")

    print(f"M = {M}")
    print(f"N = {N}")
    print(f"L = {L}")
    print(f"K = {K}")
    print(f"J = {J}")
    print(f"T = {T}")

    print()

    print(
        "X shape:",
        X_true.shape,
    )

    print(
        "S shape:",
        S_noisy.shape,
    )

    print(
        f"relative noise level: "
        f"{actual_noise:.6e}"
    )

    print()
    print("True trajectory")
    print("---------------")

    print(
        "physical positions:   ",
        true_physical_positions,
    )

    print(
        "coefficient positions:",
        true_coefficient_positions,
    )

    print(
        "scales:               ",
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

    # -------------------------------------------------
    # L1 sweep
    # -------------------------------------------------

    print()
    print("L-domain L1-ADMM sweep")
    print("----------------------")

    summary = []

    for lambda1 in lambda1_values:
        print()
        print("=" * 64)

        print(
            f"lambda1 = {lambda1:.4g}, "
            f"lambda2 = {lambda2:.4g}, "
            f"rho = {rho:.4g}"
        )

        print("=" * 64)

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
            max_admm_iter=200,
            x_maxiter=50,
            abs_tol=1e-4,
            rel_tol=1e-3,
            nonnegative=True,
            adaptive_rho=True,
            rho_mu=10.0,
            rho_tau=2.0,
            verbose=False,
        )

        # Y is the sparse ADMM coefficient estimate.
        X_hat = Y_admm

        S_hat = operator.forward(
            X_hat
        )

        signal_error_clean = relative_error(
            S_hat,
            S_clean,
        )

        signal_error_noisy = relative_error(
            S_hat,
            S_noisy,
        )

        coefficient_error = relative_error(
            X_hat,
            X_true,
        )

        (
            coefficient_positions,
            physical_positions,
            scale_indices,
            amplitudes,
        ) = trajectory_from_coefficients(
            X_hat,
            M=M,
        )

        significant = count_significant(
            X_hat
        )

        exact_nonzeros = int(
            np.count_nonzero(
                X_hat
            )
        )

        exact_zeros = (
            X_hat.size
            - exact_nonzeros
        )

        print(
            "converged:              ",
            info["converged"],
        )

        print(
            "ADMM iterations:        ",
            info["iterations"],
        )

        print(
            "final rho:              ",
            info["rho"],
        )

        print(
            "exact nonzeros:         ",
            exact_nonzeros,
        )

        print(
            "significant (>1e-3):    ",
            significant,
        )

        print(
            "exact zeros:            ",
            exact_zeros,
        )

        print(
            f"signal error clean:      "
            f"{signal_error_clean:.6e}"
        )

        print(
            f"signal error noisy:      "
            f"{signal_error_noisy:.6e}"
        )

        print(
            f"coefficient error:       "
            f"{coefficient_error:.6e}"
        )

        print(
            f"primal residual:         "
            f"{info['final_primal_residual']:.6e}"
        )

        print(
            f"dual residual:           "
            f"{info['final_dual_residual']:.6e}"
        )

        print(
            "coefficient positions:  ",
            coefficient_positions,
        )

        print(
            "physical positions:     ",
            physical_positions,
        )

        print(
            "scales:                 ",
            scale_indices,
        )

        print(
            "amplitudes:             ",
            [
                round(value, 4)
                for value in amplitudes
            ],
        )

        summary.append(
            {
                "lambda1":
                    lambda1,

                "converged":
                    info["converged"],

                "iterations":
                    info["iterations"],

                "rho":
                    info["rho"],

                "nonzeros":
                    exact_nonzeros,

                "significant":
                    significant,

                "signal_error":
                    signal_error_clean,

                "coefficient_error":
                    coefficient_error,

                "coefficient_positions":
                    coefficient_positions,

                "physical_positions":
                    physical_positions,

                "scales":
                    scale_indices,
            }
        )

    # -------------------------------------------------
    # Compact summary
    # -------------------------------------------------

    print()
    print()
    print("Summary")
    print("-------")

    print(
        "lambda1  conv  iter   rho"
        "      nz    sig    "
        "signal err    coeff err"
    )

    for row in summary:
        print(
            f"{row['lambda1']:7.3f}"
            f"  {str(row['converged']):5s}"
            f"  {row['iterations']:4d}"
            f"  {row['rho']:7.3f}"
            f"  {row['nonzeros']:5d}"
            f"  {row['significant']:5d}"
            f"  {row['signal_error']:10.4e}"
            f"  {row['coefficient_error']:10.4e}"
        )

    # -------------------------------------------------
    # Trajectory comparison
    # -------------------------------------------------

    print()
    print("Trajectory comparison")
    print("---------------------")

    print(
        "true coefficient positions:",
        true_coefficient_positions,
    )

    print(
        "true physical positions:   ",
        true_physical_positions,
    )

    print(
        "true scales:               ",
        true_scales,
    )

    for row in summary:
        print()

        print(
            f"lambda1="
            f"{row['lambda1']:.3f}"
        )

        print(
            "  coefficient positions:",
            row[
                "coefficient_positions"
            ],
        )

        print(
            "  physical positions:   ",
            row[
                "physical_positions"
            ],
        )

        print(
            "  scales:               ",
            row["scales"],
        )


if __name__ == "__main__":
    recovery_test()
