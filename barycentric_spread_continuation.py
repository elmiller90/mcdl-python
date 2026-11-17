import time
import numpy as np

from gaussian_experiment import (
    SCALES,
    add_noise_snr,
    make_dictionary,
    reconstruction_snr_db,
    relative_error,
)
from barycentric_transition import (
    barycentric_transition_value_grad,
    make_barycentric_transition_regularizer,
)
from local_graph import physical_scale_values
from log_admm import solve_log_admm
from multiscale_convolution import MultiscaleConvolutionOperator


BETA_FORWARD = [
    0.00, 0.03, 0.05, 0.075,
    0.10, 0.15, 0.20, 0.30,
]

BETA_REVERSE = list(
    reversed(BETA_FORWARD)
)

SNR_DB = 14.0
SEED = 100

LAMBDA_S = 0.15
LAMBDA_T = 1e-5

GATE_MASS = 0.05
MASS_FLOOR = 1e-10
SHIFT_VELOCITY = 1.0

TAU = 1e-2
RHO = 0.10
MAX_ADMM_ITER = 600
X_MAXITER = 50

COEFF_THRESHOLD = 1e-3

M, N, T = 41, 96, 12

ACTIVE_FRAMES = np.arange(2, 10)

TRUE_POSITIONS = np.array([
    28, 29, 30, 31,
    32, 33, 34, 35,
])

TRUE_SCALES = np.array([
    0, 1, 2, 3,
    3, 2, 1, 0,
])


def make_truth(A):
    X = np.zeros(
        (A.L, A.K, A.J, T),
        dtype=np.float64,
    )

    for t, pos, j in zip(
        ACTIVE_FRAMES,
        TRUE_POSITIONS,
        TRUE_SCALES,
    ):
        n = int(pos + M - 1)
        X[n, 0, j, t] = 1.0

    return X


def dominant_state(Y, t):
    frame = np.abs(
        Y[:, :, :, t]
    )

    n, k, j = np.unravel_index(
        np.argmax(frame),
        frame.shape,
    )

    return (
        int(n - (M - 1)),
        int(j),
    )


def trajectory_metrics(Y):
    states = [
        dominant_state(Y, int(t))
        for t in ACTIVE_FRAMES
    ]

    pos = np.array(
        [z[0] for z in states]
    )

    scale = np.array(
        [z[1] for z in states]
    )

    joint = np.mean(
        (pos == TRUE_POSITIONS)
        & (scale == TRUE_SCALES)
    )

    return pos, scale, float(joint)


def support_per_frame(Y):
    return np.count_nonzero(
        np.abs(Y) > COEFF_THRESHOLD,
        axis=(0, 1, 2),
    )


def frame_support(Y, t):
    idx = np.argwhere(
        np.abs(
            Y[:, :, :, t]
        ) > COEFF_THRESHOLD
    )

    out = []

    for n, k, j in idx:
        out.append(
            (
                int(n),
                int(j),
                float(Y[n, k, j, t]),
            )
        )

    return out


def make_callback(
    beta,
    shift_coordinates,
    scales,
):
    return (
        make_barycentric_transition_regularizer(
            shift_coordinates=
                shift_coordinates,
            scale_coordinates=scales,
            gate_mass=GATE_MASS,
            mass_floor=MASS_FLOOR,
            shift_velocity=
                SHIFT_VELOCITY,
            spread_weight=beta,
        )
    )


def solve_one(
    beta,
    Xstart,
    A,
    Snoisy,
    Sclean,
    Xtrue,
    noise_target,
    shift_coordinates,
    scales,
):
    callback = make_callback(
        beta,
        shift_coordinates,
        scales,
    )

    start = time.perf_counter()

    _, Y, _, info = solve_log_admm(
        X0=Xstart.copy(),
        S=Snoisy,
        operator=A,
        lambda_s=LAMBDA_S,
        lambda2=LAMBDA_T,
        tau=TAU,
        rho=RHO,
        max_admm_iter=
            MAX_ADMM_ITER,
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
        temporal_regularizer=
            callback,
    )

    elapsed = (
        time.perf_counter() - start
    )

    Shat = A.forward(Y)
    residual = Shat - Snoisy

    data = float(
        0.5
        * np.sum(
            residual * residual
        )
    )

    rt, _, td = (
        barycentric_transition_value_grad(
            Y,
            shift_coordinates=
                shift_coordinates,
            scale_coordinates=scales,
            gate_mass=GATE_MASS,
            mass_floor=MASS_FLOOR,
            shift_velocity=
                SHIFT_VELOCITY,
            spread_weight=beta,
        )
    )

    pos, scale, joint = (
        trajectory_metrics(Y)
    )

    nz_frame = support_per_frame(Y)

    return {
        "beta": float(beta),
        "Y": Y,
        "converged":
            bool(info["converged"]),
        "iterations":
            int(info["iterations"]),
        "nz":
            int(nz_frame.sum()),
        "nz_per_frame":
            nz_frame.tolist(),
        "discrepancy":
            data / noise_target,
        "xerr":
            float(
                relative_error(
                    Y, Xtrue
                )
            ),
        "snr":
            float(
                reconstruction_snr_db(
                    Sclean, Shat
                )
            ),
        "joint": joint,
        "positions":
            pos.tolist(),
        "scales":
            scale.tolist(),
        "total_temporal":
            float(rt),
        "motion":
            float(
                td["motion_value"]
            ),
        "spread":
            float(
                td["spread_value"]
            ),
        "frame4":
            frame_support(Y, 4),
        "runtime":
            float(elapsed),
    }


def run_path(
    name,
    betas,
    Xstart,
    A,
    Snoisy,
    Sclean,
    Xtrue,
    noise_target,
    shift_coordinates,
    scales,
):
    print()
    print(name)
    print("-" * len(name))

    rows = []
    Xcurrent = Xstart.copy()

    for beta in betas:
        r = solve_one(
            beta,
            Xcurrent,
            A,
            Snoisy,
            Sclean,
            Xtrue,
            noise_target,
            shift_coordinates,
            scales,
        )

        rows.append(r)

        # Continuation step.
        Xcurrent = r["Y"].copy()

        print(
            f"beta={beta:6.3f} "
            f"conv="
            f"{str(r['converged']):5s} "
            f"iter="
            f"{r['iterations']:3d} "
            f"nz={r['nz']:2d} "
            f"D/Dn="
            f"{r['discrepancy']:.3f} "
            f"Xerr="
            f"{r['xerr']:.4f} "
            f"SNR="
            f"{r['snr']:5.2f} "
            f"joint="
            f"{100*r['joint']:5.1f}% "
            f"spread="
            f"{r['spread']:.4f}"
        )

    return rows


def print_path_details(
    name,
    rows,
):
    print()
    print(name)
    print("=" * len(name))

    for r in rows:
        print()
        print(
            f"beta={r['beta']:.3f}"
        )
        print(
            "nz/frame:",
            r["nz_per_frame"],
        )
        print(
            "scales:  ",
            r["scales"],
        )
        print(
            "t=4:"
        )

        for n, j, amp in r["frame4"]:
            print(
                f"  n={n:3d} "
                f"j={j} "
                f"amp={amp:.6f}"
            )


def compare_paths(
    forward,
    reverse,
):
    by_forward = {
        round(r["beta"], 6): r
        for r in forward
    }

    by_reverse = {
        round(r["beta"], 6): r
        for r in reverse
    }

    common = sorted(
        set(by_forward)
        & set(by_reverse)
    )

    print()
    print("Forward / reverse comparison")
    print("----------------------------")
    print(
        "beta   nz_f nz_r "
        "Xerr_f Xerr_r "
        "joint_f joint_r"
    )

    for beta in common:
        f = by_forward[beta]
        r = by_reverse[beta]

        print(
            f"{beta:5.3f} "
            f"{f['nz']:4d} "
            f"{r['nz']:4d} "
            f"{f['xerr']:6.3f} "
            f"{r['xerr']:6.3f} "
            f"{100*f['joint']:7.1f}% "
            f"{100*r['joint']:7.1f}%"
        )


def main():
    rng = np.random.default_rng(
        SEED
    )

    D = make_dictionary(
        M, 4.0
    )

    A = MultiscaleConvolutionOperator(
        D=D,
        scales=SCALES,
        center=(M + 1) // 2,
        N=N,
        normalize=True,
    )

    Xtrue = make_truth(A)
    Sclean = A.forward(Xtrue)

    Snoisy, noise, achieved_snr = (
        add_noise_snr(
            Sclean,
            SNR_DB,
            rng,
        )
    )

    noise_target = float(
        0.5
        * np.sum(
            noise * noise
        )
    )

    scales = physical_scale_values(
        SCALES
    )

    shift_coordinates = np.arange(
        A.L,
        dtype=np.float64,
    )

    Xbase = A.adjoint(
        Snoisy,
        structured=True,
    )

    Xbase = np.maximum(
        Xbase,
        0.0,
    )

    if Xbase.max() > 0:
        Xbase /= Xbase.max()

    print()
    print("Spread continuation experiment")
    print("------------------------------")
    print(
        f"SNR:      "
        f"{achieved_snr:.2f} dB"
    )
    print(
        f"lambda_s: {LAMBDA_S}"
    )
    print(
        f"lambda_t: {LAMBDA_T}"
    )

    # Forward path begins from common adjoint initialization.
    forward = run_path(
        "Forward continuation",
        BETA_FORWARD,
        Xbase,
        A,
        Snoisy,
        Sclean,
        Xtrue,
        noise_target,
        shift_coordinates,
        scales,
    )

    # Reverse path deliberately begins from the converged
    # beta=0.30 endpoint of the forward path.
    reverse_start = (
        forward[-1]["Y"].copy()
    )

    reverse = run_path(
        "Reverse continuation",
        BETA_REVERSE,
        reverse_start,
        A,
        Snoisy,
        Sclean,
        Xtrue,
        noise_target,
        shift_coordinates,
        scales,
    )

    compare_paths(
        forward,
        reverse,
    )

    print_path_details(
        "Forward path details",
        forward,
    )

    print_path_details(
        "Reverse path details",
        reverse,
    )


if __name__ == "__main__":
    main()