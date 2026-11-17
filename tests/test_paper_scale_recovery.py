import numpy as np

from multiscale_convolution import MultiscaleConvolutionOperator
from sparse_admm import solve_l1_admm


def make_dictionary(M):
    """
    Single centered Gaussian base atom.

    The base atom represents scale 1.
    """
    x = np.arange(M, dtype=np.float64)

    center = (M - 1) / 2.0
    width = 4.0

    d = np.exp(
        -0.5 * ((x - center) / width) ** 2
    )

    d /= np.linalg.norm(d)

    return d[:, None]


def physical_to_coefficient(n, M):
    return (M - 1) + n


def coefficient_to_physical(n, M):
    return n - (M - 1)


def make_true_coefficients(
    L,
    M,
    scales,
    T,
):
    """
    Create a trajectory through the paper-compatible
    position-scale domain.

    The scale indices move only by one between frames,
    so all transitions are permitted by softmin.
    """
    J = len(scales)

    X = np.zeros(
        (L, 1, J, T),
        dtype=np.float64,
    )

    physical_positions = [
        28,
        29,
        30,
        31,
        32,
        33,
        34,
    ]

    # Move from scale 1 down through progressively
    # smaller atoms and then back upward.
    scale_indices = [
        0,
        1,
        2,
        3,
        2,
        1,
        0,
    ]

    if T != len(physical_positions):
        raise ValueError(
            "This experiment expects T=7"
        )

    coefficient_positions = [
        physical_to_coefficient(n, M)
        for n in physical_positions
    ]

    for t in range(T):
        X[
            coefficient_positions[t],
            0,
            scale_indices[t],
            t,
        ] = 1.0

    return (
        X,
        physical_positions,
        coefficient_positions,
        scale_indices,
    )


def dominant_trajectory(X, M):
    """
    Extract dominant position and scale in every frame.
    """
    L, K, J, T = X.shape

    if K != 1:
        raise ValueError(
            "This test assumes K=1"
        )

    coefficient_positions = []
    physical_positions = []
    scale_indices = []
    amplitudes = []

    for t in range(T):
        frame = np.abs(
            X[:, 0, :, t]
        )

        flat = int(
            np.argmax(frame)
        )

        n_coeff, j = np.unravel_index(
            flat,
            frame.shape,
        )

        coefficient_positions.append(
            int(n_coeff)
        )

        physical_positions.append(
            int(
                coefficient_to_physical(
                    n_coeff,
                    M,
                )
            )
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
    return int(
        np.count_nonzero(
            np.abs(X) > threshold
        )
    )


def run_experiment():
    rng = np.random.default_rng(100)

    # -------------------------------------------------
    # Dimensions
    # -------------------------------------------------

    M = 41
    N = 96
    K = 1
    T = 7
    center = 21

    # -------------------------------------------------
    # Paper-compatible contraction scales
    #
    # scale = p/q <= 1
    # -------------------------------------------------

    scales = [
        (1, 1),
        (4, 5),
        (3, 4),
        (2, 3),
        (1, 2),
    ]

    scale_values = [
        p / q
        for p, q in scales
    ]

    J = len(scales)

    # -------------------------------------------------
    # Dictionary/operator
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

    # -------------------------------------------------
    # Ground truth
    # -------------------------------------------------

    (
        X_true,
        true_physical,
        true_coeff,
        true_scale_indices,
    ) = make_true_coefficients(
        L=L,
        M=M,
        scales=scales,
        T=T,
    )

    true_scale_values = [
        scale_values[j]
        for j in true_scale_indices
    ]

    # -------------------------------------------------
    # Synthetic physical measurements
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

    # -------------------------------------------------
    # Adjoint initialization
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
    # Geometry report
    # -------------------------------------------------

    print()
    print("Paper-compatible scale-set experiment")
    print("-------------------------------------")

    print(f"M = {M}")
    print(f"N = {N}")
    print(f"L = {L}")
    print(f"K = {K}")
    print(f"J = {J}")
    print(f"T = {T}")

    print()
    print("Scale set")
    print("---------")

    for j, ((p, q), value) in enumerate(
        zip(scales, scale_values)
    ):
        print(
            f"j={j}: "
            f"{p}/{q} = {value:.6f}"
        )

    print()
    print("True trajectory")
    print("---------------")

    print(
        "physical positions:   ",
        true_physical,
    )

    print(
        "coefficient positions:",
        true_coeff,
    )

    print(
        "scale indices:        ",
        true_scale_indices,
    )

    print(
        "scale values:         ",
        [
            round(v, 6)
            for v in true_scale_values
        ],
    )

    print()
    print(
        "Initial significant coefficients:",
        count_significant(X0),
    )

    # -------------------------------------------------
    # Solver parameters
    # -------------------------------------------------

    tau = 1e-2
    lambda2 = 0.1
    rho = 0.1

    lambda1_values = [
        0.01,
        0.03,
        0.05,
    ]

    summary = []

    # -------------------------------------------------
    # Sparse/softmin recovery sweep
    # -------------------------------------------------

    for lambda1 in lambda1_values:
        print()
        print("=" * 64)

        print(
            f"lambda1={lambda1:.4f}, "
            f"lambda2={lambda2:.4f}, "
            f"tau={tau:.3e}"
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

        X_hat = Y_admm

        S_hat = operator.forward(
            X_hat
        )

        signal_error = relative_error(
            S_hat,
            S_clean,
        )

        coefficient_error = relative_error(
            X_hat,
            X_true,
        )

        (
            coeff_positions,
            physical_positions,
            recovered_scale_indices,
            amplitudes,
        ) = dominant_trajectory(
            X_hat,
            M=M,
        )

        recovered_scale_values = [
            scale_values[j]
            for j in recovered_scale_indices
        ]

        significant = count_significant(
            X_hat
        )

        position_matches = sum(
            a == b
            for a, b in zip(
                physical_positions,
                true_physical,
            )
        )

        scale_matches = sum(
            a == b
            for a, b in zip(
                recovered_scale_indices,
                true_scale_indices,
            )
        )

        exact_path = (
            physical_positions
            == true_physical
            and recovered_scale_indices
            == true_scale_indices
        )

        print(
            "converged:             ",
            info["converged"],
        )

        print(
            "ADMM iterations:       ",
            info["iterations"],
        )

        print(
            "final rho:             ",
            info["rho"],
        )

        print(
            "significant (>1e-3):   ",
            significant,
        )

        print(
            f"signal error:           "
            f"{signal_error:.6e}"
        )

        print(
            f"coefficient error:      "
            f"{coefficient_error:.6e}"
        )

        print(
            "physical positions:    ",
            physical_positions,
        )

        print(
            "scale indices:         ",
            recovered_scale_indices,
        )

        print(
            "scale values:          ",
            [
                round(v, 6)
                for v in recovered_scale_values
            ],
        )

        print(
            f"position matches:       "
            f"{position_matches}/{T}"
        )

        print(
            f"scale matches:          "
            f"{scale_matches}/{T}"
        )

        print(
            "exact trajectory:      ",
            exact_path,
        )

        print(
            "amplitudes:            ",
            [
                round(v, 4)
                for v in amplitudes
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

                "significant":
                    significant,

                "signal_error":
                    signal_error,

                "coefficient_error":
                    coefficient_error,

                "position_matches":
                    position_matches,

                "scale_matches":
                    scale_matches,

                "exact":
                    exact_path,

                "physical":
                    physical_positions,

                "scales":
                    recovered_scale_indices,
            }
        )

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    print()
    print()
    print("Summary")
    print("-------")

    print(
        "lambda1  conv  iter   rho"
        "   sig     signal err"
        "    coeff err   pos   scale   exact"
    )

    for row in summary:
        print(
            f"{row['lambda1']:7.3f}"
            f"  {str(row['converged']):5s}"
            f"  {row['iterations']:4d}"
            f"  {row['rho']:5.2f}"
            f"  {row['significant']:4d}"
            f"  {row['signal_error']:11.4e}"
            f"  {row['coefficient_error']:11.4e}"
            f"   {row['position_matches']}/{T}"
            f"   {row['scale_matches']}/{T}"
            f"   {row['exact']}"
        )


if __name__ == "__main__":
    run_experiment()