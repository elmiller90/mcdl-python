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


SPREAD_WEIGHTS = np.array([
    0.0, 0.1, 0.3, 1.0, 3.0
])

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

ACTIVITY_THRESHOLD = 0.05
COEFF_THRESHOLD = 1e-3

M, N, T = 41, 96, 12

ACTIVE_FRAMES = np.arange(2, 10)

TRUE_POSITIONS = np.array([
    28, 29, 30, 31, 32, 33, 34, 35
])

TRUE_SCALES = np.array([
    0, 1, 2, 3, 3, 2, 1, 0
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
    frame = np.abs(Y[:, :, :, t])

    n, k, j = np.unravel_index(
        np.argmax(frame),
        frame.shape,
    )

    return (
        int(n - (M - 1)),
        int(j),
    )


def trajectory_metrics(Y):
    state = [
        dominant_state(Y, int(t))
        for t in ACTIVE_FRAMES
    ]

    pos = np.array([
        z[0] for z in state
    ])

    scale = np.array([
        z[1] for z in state
    ])

    return {
        "positions": pos,
        "scales": scale,
        "position_accuracy": float(
            np.mean(
                pos == TRUE_POSITIONS
            )
        ),
        "scale_accuracy": float(
            np.mean(
                scale == TRUE_SCALES
            )
        ),
        "joint_accuracy": float(
            np.mean(
                (pos == TRUE_POSITIONS)
                & (scale == TRUE_SCALES)
            )
        ),
    }


def activity_metrics(masses):
    truth = np.zeros(
        T,
        dtype=bool,
    )

    truth[ACTIVE_FRAMES] = True

    recovered = (
        np.asarray(masses)
        > ACTIVITY_THRESHOLD
    )

    tp = np.count_nonzero(
        truth & recovered
    )

    fp = np.count_nonzero(
        ~truth & recovered
    )

    fn = np.count_nonzero(
        truth & ~recovered
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 1.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 1.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return recovered, float(f1)


def frame4_support(Y):
    idx = np.argwhere(
        np.abs(
            Y[:, :, :, 4]
        ) > COEFF_THRESHOLD
    )

    return [
        (
            int(n),
            int(j),
            float(Y[n, k, j, 4]),
        )
        for n, k, j in idx
    ]


def main():
    rng = np.random.default_rng(
        SEED
    )

    D = make_dictionary(
        M,
        4.0,
    )

    A = MultiscaleConvolutionOperator(
        D=D,
        scales=SCALES,
        center=(M + 1) // 2,
        N=N,
        normalize=True,
    )

    Xtrue = make_truth(A)

    Sclean = A.forward(
        Xtrue
    )

    Snoisy, noise, achieved_snr = (
        add_noise_snr(
            Sclean,
            SNR_DB,
            rng,
        )
    )

    noise_target_expected = float(
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

    X0 = A.adjoint(
        Snoisy,
        structured=True,
    )

    X0 = np.maximum(
        X0,
        0.0,
    )

    if X0.max() > 0:
        X0 /= X0.max()

    rows = []

    print()
    print("Barycentric spread-weight sweep")
    print("-------------------------------")
    print(
        f"SNR:           "
        f"{achieved_snr:.2f} dB"
    )
    print(
        f"lambda_s:      "
        f"{LAMBDA_S}"
    )
    print(
        f"lambda_t:      "
        f"{LAMBDA_T}"
    )

    for beta in SPREAD_WEIGHTS:
        beta = float(beta)

        temporal_regularizer = (
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

        start = time.perf_counter()

        _, Y, _, info = solve_log_admm(
            X0=X0.copy(),
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
                temporal_regularizer,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        Shat = A.forward(Y)

        residual = (
            Shat - Snoisy
        )

        data = float(
            0.5
            * np.sum(
                residual * residual
            )
        )

        total_rt, _, td = (
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

        tr = trajectory_metrics(Y)

        masses = (
            td["masses"][0]
        )

        active, f1 = activity_metrics(
            masses
        )

        nz_per_frame = (
            np.count_nonzero(
                np.abs(Y)
                > COEFF_THRESHOLD,
                axis=(0, 1, 2),
            )
        )

        row = {
            "spread_weight": beta,
            "converged":
                bool(info["converged"]),
            "iterations":
                int(info["iterations"]),
            "nz":
                int(nz_per_frame.sum()),
            "nz_per_frame":
                nz_per_frame.tolist(),
            "discrepancy":
                data / noise_target_expected,
            "xerr":
                float(
                    relative_error(
                        Y,
                        Xtrue,
                    )
                ),
            "snr":
                float(
                    reconstruction_snr_db(
                        Sclean,
                        Shat,
                    )
                ),
            "joint":
                tr["joint_accuracy"],
            "scale_accuracy":
                tr["scale_accuracy"],
            "positions":
                tr["positions"].tolist(),
            "scales":
                tr["scales"].tolist(),
            "activity":
                active.astype(int).tolist(),
            "activity_f1":
                f1,
            "total_temporal":
                float(total_rt),
            "motion_value":
                float(
                    td["motion_value"]
                ),
            "spread_value":
                float(
                    td["spread_value"]
                ),
            "weighted_spread":
                float(
                    td[
                        "weighted_spread_value"
                    ]
                ),
            "frame_spread":
                td[
                    "frame_spread"
                ][0].tolist(),
            "frame4":
                frame4_support(Y),
            "runtime":
                float(elapsed),
        }

        rows.append(row)

        print(
            f"beta={beta:4.1f} "
            f"conv="
            f"{str(row['converged']):5s} "
            f"iter="
            f"{row['iterations']:3d} "
            f"nz="
            f"{row['nz']:3d} "
            f"D/Dn="
            f"{row['discrepancy']:.3f} "
            f"Xerr="
            f"{row['xerr']:.4f} "
            f"SNR="
            f"{row['snr']:5.2f} "
            f"joint="
            f"{100*row['joint']:5.1f}% "
            f"F1="
            f"{row['activity_f1']:.2f} "
            f"spread="
            f"{row['spread_value']:.3f} "
            f"time="
            f"{row['runtime']:5.1f}s"
        )

    print()
    print("Summary")
    print("-------")
    print(
        "beta conv iter nz D/Dn "
        "Xerr   SNR  joint F1 "
        "motion spread"
    )

    for r in rows:
        print(
            f"{r['spread_weight']:4.1f} "
            f"{str(r['converged']):5s} "
            f"{r['iterations']:4d} "
            f"{r['nz']:3d} "
            f"{r['discrepancy']:5.3f} "
            f"{r['xerr']:6.4f} "
            f"{r['snr']:5.2f} "
            f"{100*r['joint']:5.1f}% "
            f"{r['activity_f1']:.2f} "
            f"{r['motion_value']:7.3f} "
            f"{r['spread_value']:7.3f}"
        )

    print()
    print("Support by frame")
    print("----------------")

    for r in rows:
        print(
            f"beta="
            f"{r['spread_weight']:4.1f} "
            f"{r['nz_per_frame']}"
        )

    print()
    print("Ambiguous frame t=4")
    print("-------------------")

    for r in rows:
        print()
        print(
            f"spread_weight="
            f"{r['spread_weight']:.1f}"
        )

        for n, j, amp in r["frame4"]:
            print(
                f"  n={n:3d} "
                f"j={j} "
                f"amp={amp:.6f}"
            )

    print()
    print("Recovered scales")
    print("----------------")

    for r in rows:
        print(
            f"beta="
            f"{r['spread_weight']:4.1f} "
            f"{r['scales']}"
        )

    print()
    print("Frame spread")
    print("------------")

    for r in rows:
        v = np.asarray(
            r["frame_spread"]
        )

        print(
            f"beta="
            f"{r['spread_weight']:4.1f} "
            f"{np.array2string(v, precision=4)}"
        )


if __name__ == "__main__":
    main()