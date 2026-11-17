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
from gated_transition import (
    make_gated_transition_regularizer,
    gated_transition_value_grad,
)
from local_graph import (
    build_local_transport_graph,
    physical_scale_values,
)
from local_transition import build_transition_kernel
from log_admm import solve_log_admm
from multiscale_convolution import MultiscaleConvolutionOperator


SNR_DB = 14.0
SEED = 100

LAMBDA_S = 0.125
LAMBDA_T = 0.0

EPSILON = 0.5
GATE_MASS = 0.05
MASS_FLOOR = 1e-10

TAU = 1e-2
RHO = 0.10

MAX_ADMM_ITER = 600
X_MAXITER = 50


def trajectory_metrics(Y, M):
    tr = dominant_trajectory(Y, M)

    pos = sum(
        a == b
        for a, b in zip(
            tr["physical_positions"],
            TRUE_PHYSICAL_POSITIONS,
        )
    )

    scale = sum(
        a == b
        for a, b in zip(
            tr["scale_indices"],
            TRUE_SCALE_INDICES,
        )
    )

    joint = sum(
        p == pt and s == st
        for p, s, pt, st in zip(
            tr["physical_positions"],
            tr["scale_indices"],
            TRUE_PHYSICAL_POSITIONS,
            TRUE_SCALE_INDICES,
        )
    )

    T = len(TRUE_SCALE_INDICES)

    return (
        tr,
        pos / T,
        scale / T,
        joint / T,
    )


def main():
    rng = np.random.default_rng(SEED)

    M, N, T = 41, 96, 7

    # --------------------------------------
    # Forward model
    # --------------------------------------

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

    # --------------------------------------
    # Gated local-transition geometry
    # --------------------------------------

    transport_scales = physical_scale_values(
        SCALES
    )

    graph = build_local_transport_graph(
        L=A.L,
        scales=transport_scales,
        shift_radius=3,
        scale_radius=1,
        shift_velocity=1.0,
    )

    prob = build_transition_kernel(
        graph,
        epsilon=EPSILON,
    )

    temporal_regularizer = (
        make_gated_transition_regularizer(
            graph,
            prob,
            gate_mass=GATE_MASS,
            mass_floor=MASS_FLOOR,
        )
    )

    # --------------------------------------
    # Initialization
    # --------------------------------------

    X0 = A.adjoint(
        S_noisy,
        structured=True,
    )

    X0 /= np.max(
        np.abs(X0)
    )

    # --------------------------------------
    # Solve
    # --------------------------------------

    print()
    print("Gated-transition Gaussian smoke test")
    print("------------------------------------")
    print(f"input SNR:       {input_snr:.2f} dB")
    print(f"lambda_s:        {LAMBDA_S}")
    print(f"lambda_t:        {LAMBDA_T}")
    print(f"epsilon:         {EPSILON}")
    print(f"gate mass:       {GATE_MASS}")
    print(f"mass floor:      {MASS_FLOOR}")
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

    # --------------------------------------
    # Recovery diagnostics
    # --------------------------------------

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

    # --------------------------------------
    # Gated-temporal diagnostics
    # --------------------------------------

    temporal_value, _, td = (
        gated_transition_value_grad(
            Y,
            graph,
            prob,
            gate_mass=GATE_MASS,
            mass_floor=MASS_FLOOR,
        )
    )

    discrepancy_target = (
        0.5
        * np.sum(noise * noise)
    )

    residual = A.forward(Y) - S_noisy

    data_fidelity = (
        0.5
        * np.sum(residual * residual)
    )

    discrepancy_ratio = (
        data_fidelity
        / discrepancy_target
    )

    # --------------------------------------
    # Report
    # --------------------------------------

    print()
    print("Solver")
    print("------")
    print(f"converged:        {info['converged']}")
    print(f"iterations:       {info['iterations']}")
    print(f"final rho:        {info['rho']:.6g}")
    print(f"final a:          {info['a']:.6g}")
    print(f"runtime:          {elapsed:.2f} s")

    print()
    print("Recovery")
    print("--------")
    print(f"coefficient err:  {x_error:.6f}")
    print(f"signal err:       {s_error:.6f}")
    print(f"output SNR:       {output_snr:.2f} dB")
    print(f"significant nz:   {n_sig}")
    print(f"position acc:     {100*pos_acc:.1f}%")
    print(f"scale acc:        {100*scale_acc:.1f}%")
    print(f"joint acc:        {100*joint_acc:.1f}%")

    print()
    print("Noise consistency")
    print("-----------------")
    print(
        f"data fidelity:    "
        f"{data_fidelity:.6e}"
    )
    print(
        f"noise target:     "
        f"{discrepancy_target:.6e}"
    )
    print(
        f"ratio D/Dnoise:   "
        f"{discrepancy_ratio:.4f}"
    )

    print()
    print("Gated temporal")
    print("--------------")
    print(
        f"raw value:        "
        f"{temporal_value:.6e}"
    )
    print(
        f"weighted value:   "
        f"{LAMBDA_T*temporal_value:.6e}"
    )
    print(
        "frame masses:",
        np.array2string(
            td["masses"][0],
            precision=4,
        ),
    )
    print(
        "frame gates: ",
        np.array2string(
            td["frame_gates"][0],
            precision=6,
        ),
    )
    print(
        "pair weights:",
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


if __name__ == "__main__":
    main()
