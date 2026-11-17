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


LAMBDA_T_VALUES = np.array([
    0.0,
    1e-5,
    3e-5,
    1e-4,
    2e-4,
    3e-4,
    5e-4,
    7.5e-4,
    1e-3,
])

SNR_DB = 14.0
SEED = 100
LAMBDA_S = 0.15

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

    return int(n - (M - 1)), int(j)


def trajectory_metrics(Y):
    states = [
        dominant_state(Y, int(t))
        for t in ACTIVE_FRAMES
    ]

    pos = np.array([
        x[0] for x in states
    ])

    scale = np.array([
        x[1] for x in states
    ])

    joint = (
        (pos == TRUE_POSITIONS)
        & (scale == TRUE_SCALES)
    )

    return {
        "positions": pos,
        "scales": scale,
        "position_accuracy": float(
            np.mean(pos == TRUE_POSITIONS)
        ),
        "scale_accuracy": float(
            np.mean(scale == TRUE_SCALES)
        ),
        "joint_accuracy": float(
            np.mean(joint)
        ),
    }


def activity_metrics(masses):
    truth = np.zeros(T, dtype=bool)
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
        if tp + fp else 1.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn else 1.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return recovered, float(f1)


def main():
    rng = np.random.default_rng(SEED)

    D = make_dictionary(M, 4.0)

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

    noise_target_expected = float(0.5 * Snoisy.size * sigma**2)
    noise_target_realized = float(0.5 * np.sum(noise * noise))

    scales = physical_scale_values(
        SCALES
    )

    shift_coordinates = np.arange(
        A.L,
        dtype=np.float64,
    )

    temporal_regularizer = (
        make_barycentric_transition_regularizer(
            shift_coordinates=
                shift_coordinates,
            scale_coordinates=scales,
            gate_mass=GATE_MASS,
            mass_floor=MASS_FLOOR,
            shift_velocity=SHIFT_VELOCITY,
        )
    )

    X0 = A.adjoint(
        Snoisy,
        structured=True,
    )

    X0 = np.maximum(X0, 0.0)

    if X0.max() > 0:
        X0 /= X0.max()

    rows = []

    print()
    print("Finite-track barycentric lambda_t sweep")
    print("---------------------------------------")
    print(f"SNR:       {achieved_snr:.2f} dB")
    print(f"lambda_s:  {LAMBDA_S}")
    print()

    for lt in LAMBDA_T_VALUES:
        start = time.perf_counter()

        _, Y, _, info = solve_log_admm(
            X0=X0.copy(),
            S=Snoisy,
            operator=A,
            lambda_s=LAMBDA_S,
            lambda2=float(lt),
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
            temporal_regularizer=
                temporal_regularizer,
        )

        elapsed = (
            time.perf_counter() - start
        )

        Shat = A.forward(Y)
        residual = Shat - Snoisy

        data = float(
            0.5 * np.sum(
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
            )
        )

        tr = trajectory_metrics(Y)

        masses = td["masses"][0]

        recovered_active, f1 = (
            activity_metrics(masses)
        )

        nz_per_frame = np.count_nonzero(
            np.abs(Y) > COEFF_THRESHOLD,
            axis=(0, 1, 2),
        )

        nz = int(
            np.sum(nz_per_frame)
        )

        # Detailed support specifically at ambiguous frame t=4.
        idx4 = np.argwhere(
            np.abs(Y[:, :, :, 4])
            > COEFF_THRESHOLD
        )

        frame4 = []

        for n, k, j in idx4:
            frame4.append(
                (
                    int(n),
                    int(j),
                    float(Y[n, k, j, 4]),
                )
            )

        row = {
            "lambda_t": float(lt),
            "converged":
                bool(info["converged"]),
            "iterations":
                int(info["iterations"]),
            "nz": nz,
            "nz_per_frame":
                nz_per_frame.tolist(),
            "discrepancy":
                data / noise_target_expected,
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
            "joint":
                tr["joint_accuracy"],
            "scale":
                tr["scale_accuracy"],
            "positions":
                tr["positions"].tolist(),
            "scales":
                tr["scales"].tolist(),
            "activity_f1":
                f1,
            "activity":
                recovered_active
                .astype(int)
                .tolist(),
            "rt":
                float(rt),
            "weighted_rt":
                float(lt * rt),
            "frame4":
                frame4,
            "runtime":
                float(elapsed),
        }

        rows.append(row)

        print(
            f"lt={lt:8.1e} "
            f"conv={str(row['converged']):5s} "
            f"iter={row['iterations']:3d} "
            f"nz={row['nz']:3d} "
            f"D/Dn={row['discrepancy']:.3f} "
            f"Xerr={row['xerr']:.4f} "
            f"SNR={row['snr']:5.2f} "
            f"joint={100*row['joint']:5.1f}% "
            f"F1={row['activity_f1']:.2f} "
            f"time={row['runtime']:5.1f}s"
        )

    print()
    print("Summary")
    print("-------")
    print(
        "lambda_t conv iter  nz D/Dn "
        "Xerr   SNR   joint F1"
    )

    for r in rows:
        print(
            f"{r['lambda_t']:8.1e} "
            f"{str(r['converged']):5s} "
            f"{r['iterations']:4d} "
            f"{r['nz']:3d} "
            f"{r['discrepancy']:5.3f} "
            f"{r['xerr']:6.4f} "
            f"{r['snr']:5.2f} "
            f"{100*r['joint']:5.1f}% "
            f"{r['activity_f1']:.2f}"
        )

    print()
    print("Support by frame")
    print("----------------")

    for r in rows:
        print(
            f"lt={r['lambda_t']:8.1e} "
            f"{r['nz_per_frame']}"
        )

    print()
    print("Ambiguous frame t=4")
    print("-------------------")

    for r in rows:
        print()
        print(
            f"lambda_t="
            f"{r['lambda_t']:.1e}"
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
            f"lt={r['lambda_t']:8.1e} "
            f"{r['scales']}"
        )


if __name__ == "__main__":
    main()

    