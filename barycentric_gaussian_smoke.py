import time
import numpy as np

from gaussian_experiment import (
    SCALES,
    TRUE_PHYSICAL_POSITIONS,
    TRUE_SCALE_INDICES,
    add_noise_snr,
    dominant_trajectory,
    make_dictionary,
    make_true_coefficients,
    reconstruction_snr_db,
    relative_error,
)
from barycentric_transition import (
    barycentric_transition_value_grad,
    make_barycentric_transition_regularizer,
)
from local_graph import physical_scale_values
from log_admm import solve_log_admm
from multiscale_convolution import (
    MultiscaleConvolutionOperator,
)


# ---------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------

SNR_DB = 14.0
SEED = 100

LAMBDA_S = 0.125
LAMBDA_T = 1e-3

GATE_MASS = 0.05
MASS_FLOOR = 1e-10

SHIFT_VELOCITY = 1.0

TAU = 1e-2
RHO = 0.10

MAX_ADMM_ITER = 600
X_MAXITER = 50


def trajectory_metrics(Y, M):
    tr = dominant_trajectory(Y, M)

    T = len(TRUE_SCALE_INDICES)

    pos_acc = sum(
        a == b
        for a, b in zip(
            tr["physical_positions"],
            TRUE_PHYSICAL_POSITIONS,
        )
    ) / T

    scale_acc = sum(
        a == b
        for a, b in zip(
            tr["scale_indices"],
            TRUE_SCALE_INDICES,
        )
    ) / T

    joint_acc = sum(
        p == pt and s == st
        for p, s, pt, st in zip(
            tr["physical_positions"],
            tr["scale_indices"],
            TRUE_PHYSICAL_POSITIONS,
            TRUE_SCALE_INDICES,
        )
    ) / T

    return tr, pos_acc, scale_acc, joint_acc


def main():
    rng = np.random.default_rng(SEED)

    M, N, T = 41, 96, 7

    # -----------------------------------------------------
    # Forward model
    # -----------------------------------------------------

    D = make_dictionary(M, 4.0)

    A = MultiscaleConvolutionOperator(
        D=D,
        scales=SCALES,
        center=(M + 1) // 2,
        N=N,
        normalize=True,
    )

    X_true, _ = make_true_coefficients(
        L=A.L,
        M=M,
        J=len(SCALES),
        T=T,
    )

    S_clean = A.forward(X_true)

    S_noisy, noise, input_snr = add_noise_snr(
        S_clean,
        SNR_DB,
        rng,
    )

    noise_target = 0.5 * np.sum(
        noise * noise
    )

    # -----------------------------------------------------
    # Barycentric temporal geometry
    # -----------------------------------------------------

    transport_scales = physical_scale_values(
        SCALES
    )

    shift_coordinates = np.arange(
        A.L,
        dtype=np.float64,
    )

    temporal_regularizer = (
        make_barycentric_transition_regularizer(
            shift_coordinates=shift_coordinates,
            scale_coordinates=transport_scales,
            gate_mass=GATE_MASS,
            mass_floor=MASS_FLOOR,
            shift_velocity=SHIFT_VELOCITY,
        )
    )

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    X0 = A.adjoint(
        S_noisy,
        structured=True,
    )

    X0 = np.maximum(
        X0,
        0.0,
    )

    if X0.max() > 0:
        X0 /= X0.max()

    # -----------------------------------------------------
    # Run
    # -----------------------------------------------------

    print()
    print("Barycentric Gaussian smoke test")
    print("-------------------------------")
    print(f"input SNR:       {input_snr:.2f} dB")
    print(f"lambda_s:        {LAMBDA_S}")
    print(f"lambda_t:        {LAMBDA_T}")
    print(f"gate mass:       {GATE_MASS}")
    print(f"mass floor:      {MASS_FLOOR}")
    print(f"shift velocity:  {SHIFT_VELOCITY}")
    print(
        "transport scales:",
        transport_scales,
    )

    start = time.perf_counter()

    _, Y, _, info = solve_log_admm(
        X0=X0.copy(),
        S=S_noisy,
        operator=A,
        lambda_s=LAMBDA_S,
        lambda2=LAMBDA_T,
        tau=TAU,
        rho=RHO,
        max_admm_iter=MAX_ADMM_ITER,
        x_maxiter=X_MAXITER,
        abs_tol=1e-4,
        rel_tol=1e-3,
        adaptive_rho=True,
        rho_mu=10,
        rho_tau=2,
        a_min=1e-4,
        continuation_iters=100,
        a_fraction=.95,
        verbose=False,
        temporal_regularizer=temporal_regularizer,
    )

    elapsed = time.perf_counter() - start

    # -----------------------------------------------------
    # Reconstruction diagnostics
    # -----------------------------------------------------

    S_hat = A.forward(Y)

    x_error = relative_error(
        Y,
        X_true,
    )

    s_error = relative_error(
        S_hat,
        S_clean,
    )

    output_snr = reconstruction_snr_db(
        S_clean,
        S_hat,
    )

    tr, pos_acc, scale_acc, joint_acc = (
        trajectory_metrics(
            Y,
            M,
        )
    )

    n_sig = int(
        np.count_nonzero(
            np.abs(Y) > 1e-3
        )
    )

    # -----------------------------------------------------
    # Noise-consistency diagnostics
    # -----------------------------------------------------

    residual = S_hat - S_noisy

    data_fidelity = 0.5 * np.sum(
        residual * residual
    )

    discrepancy_ratio = (
        data_fidelity
        / noise_target
    )

    # -----------------------------------------------------
    # Barycentric temporal diagnostics
    # -----------------------------------------------------

    temporal_value, _, td = (
        barycentric_transition_value_grad(
            Y,
            shift_coordinates=shift_coordinates,
            scale_coordinates=transport_scales,
            gate_mass=GATE_MASS,
            mass_floor=MASS_FLOOR,
            shift_velocity=SHIFT_VELOCITY,
        )
    )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    print()
    print("Solver")
    print("------")
    print(
        f"converged:        "
        f"{info['converged']}"
    )
    print(
        f"iterations:       "
        f"{info['iterations']}"
    )
    print(
        f"final rho:        "
        f"{info['rho']:.6g}"
    )
    print(
        f"final a:          "
        f"{info['a']:.6g}"
    )
    print(
        f"runtime:          "
        f"{elapsed:.2f} s"
    )

    print()
    print("Recovery")
    print("--------")
    print(
        f"coefficient err:  "
        f"{x_error:.6f}"
    )
    print(
        f"signal err:       "
        f"{s_error:.6f}"
    )
    print(
        f"output SNR:       "
        f"{output_snr:.2f} dB"
    )
    print(
        f"significant nz:   "
        f"{n_sig}"
    )
    print(
        f"position acc:     "
        f"{100 * pos_acc:.1f}%"
    )
    print(
        f"scale acc:        "
        f"{100 * scale_acc:.1f}%"
    )
    print(
        f"joint acc:        "
        f"{100 * joint_acc:.1f}%"
    )

    print()
    print("Noise consistency")
    print("-----------------")
    print(
        f"data fidelity:    "
        f"{data_fidelity:.6e}"
    )
    print(
        f"noise target:     "
        f"{noise_target:.6e}"
    )
    print(
        f"ratio D/Dnoise:   "
        f"{discrepancy_ratio:.4f}"
    )

    print()
    print("Barycentric temporal")
    print("--------------------")
    print(
        f"raw value:        "
        f"{temporal_value:.6e}"
    )
    print(
        f"weighted value:   "
        f"{LAMBDA_T * temporal_value:.6e}"
    )

    print(
        "frame masses:     ",
        np.array2string(
            td["masses"][0],
            precision=4,
        ),
    )

    print(
        "frame gates:      ",
        np.array2string(
            td["frame_gates"][0],
            precision=6,
        ),
    )

    print(
        "position centers: ",
        np.array2string(
            td["position_centroids"][0],
            precision=3,
        ),
    )

    print(
        "log-scale centers:",
        np.array2string(
            td["log_scale_centroids"][0],
            precision=4,
        ),
    )

    print(
        "pair motion:      ",
        np.array2string(
            td["pair_values"][0],
            precision=4,
        ),
    )

    print(
        "pair weights:     ",
        np.array2string(
            td["pair_weights"][0],
            precision=6,
        ),
    )

    print()
    print("Dominant trajectory")
    print("-------------------")
    print(
        "positions:",
        tr["physical_positions"],
    )
    print(
        "scales:   ",
        tr["scale_indices"],
    )

    idx = np.argwhere(np.abs(Y) > 1e-3)

    print()
    print("Significant coefficients")
    print("------------------------")

    for n, k, j, t in idx:
        print(
            f"t={t} "
            f"n={n} "
            f"k={k} "
            f"j={j} "
            f"amp={Y[n,k,j,t]:.6f}"
        )


if __name__ == "__main__":
    main()

