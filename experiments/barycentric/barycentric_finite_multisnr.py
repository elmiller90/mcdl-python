import time
import numpy as np

from gaussian_experiment import (
    SCALES, add_noise_snr, make_dictionary,
    reconstruction_snr_db, relative_error,
)
from barycentric_transition import (
    barycentric_transition_value_grad,
    make_barycentric_transition_regularizer,
)
from local_graph import physical_scale_values
from log_admm_stateful_compat import solve_log_admm_stateful
from multiscale_convolution import MultiscaleConvolutionOperator


SNR_VALUES = [20, 17, 14, 11, 8]
BETA_PATH = [0.075, 0.0875, 0.10]

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
        dtype=float,
    )

    for t, pos, j in zip(
        ACTIVE_FRAMES,
        TRUE_POSITIONS,
        TRUE_SCALES,
    ):
        X[int(pos + M - 1), 0, j, t] = 1.0

    return X


def dominant_trajectory(Y):
    pos, scales = [], []

    for t in ACTIVE_FRAMES:
        F = np.abs(Y[:, :, :, t])

        n, _, j = np.unravel_index(
            np.argmax(F),
            F.shape,
        )

        pos.append(
            int(n - (M - 1))
        )

        scales.append(int(j))

    pos = np.asarray(pos)
    scales = np.asarray(scales)

    pos_acc = np.mean(
        pos == TRUE_POSITIONS
    )

    scale_acc = np.mean(
        scales == TRUE_SCALES
    )

    joint_acc = np.mean(
        (pos == TRUE_POSITIONS)
        & (scales == TRUE_SCALES)
    )

    return (
        pos,
        scales,
        float(pos_acc),
        float(scale_acc),
        float(joint_acc),
    )


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

    true_idx = np.flatnonzero(
        truth
    )

    rec_idx = np.flatnonzero(
        recovered
    )

    onset = (
        int(
            rec_idx[0]
            - true_idx[0]
        )
        if rec_idx.size
        else None
    )

    offset = (
        int(
            rec_idx[-1]
            - true_idx[-1]
        )
        if rec_idx.size
        else None
    )

    return (
        recovered,
        float(f1),
        onset,
        offset,
    )


def support_per_frame(Y):
    return np.count_nonzero(
        np.abs(Y)
        > COEFF_THRESHOLD,
        axis=(0, 1, 2),
    )


def frame_support(Y, t):
    idx = np.argwhere(
        np.abs(
            Y[:, :, :, t]
        ) > COEFF_THRESHOLD
    )

    return [
        (
            int(n),
            int(j),
            float(Y[n, k, j, t]),
        )
        for n, k, j in idx
    ]


def make_callback(
    beta,
    shifts,
    scales,
):
    return (
        make_barycentric_transition_regularizer(
            shift_coordinates=shifts,
            scale_coordinates=scales,
            gate_mass=GATE_MASS,
            mass_floor=MASS_FLOOR,
            shift_velocity=
                SHIFT_VELOCITY,
            spread_weight=beta,
        )
    )


def run_snr(
    snr_db,
    A,
    Xtrue,
    Sclean,
    shifts,
    scales,
):
    # Reset RNG so each SNR uses the same
    # underlying normalized noise realization.
    rng = np.random.default_rng(
        SEED
    )

    ###
    Snoisy, noise, achieved_snr, sigma = (
        add_noise_snr(
            Sclean,
            snr_db,
            rng,
            return_sigma=True,
        )
    )

    noise_target_expected = (
        0.5
        * Snoisy.size
        * sigma**2
    )

    noise_target_realized = (
        0.5
        * np.sum(noise * noise)
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

    state = None
    stages = []

    for beta in BETA_PATH:
        callback = make_callback(
            beta,
            shifts,
            scales,
        )

        start = time.perf_counter()

        X, Y, U, info = (
            solve_log_admm_stateful(
                X0=(
                    Xbase
                    if state is None
                    else state["X"]
                ),
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
                state0=state,
                continue_iteration=False,
            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        state = info["state"]

        nz_frame = support_per_frame(
            Y
        )

        stages.append({
            "beta": beta,
            "converged":
                bool(info["converged"]),
            "iterations":
                int(info["iterations"]),
            "nz":
                int(nz_frame.sum()),
            "nz_per_frame":
                nz_frame.tolist(),
            "runtime":
                float(elapsed),
        })

    # Final beta=0.10 solution.
    Y = state["Y"]
    beta = BETA_PATH[-1]

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

    _, _, td = (
        barycentric_transition_value_grad(
            Y,
            shift_coordinates=shifts,
            scale_coordinates=scales,
            gate_mass=GATE_MASS,
            mass_floor=MASS_FLOOR,
            shift_velocity=
                SHIFT_VELOCITY,
            spread_weight=beta,
        )
    )

    (
        positions,
        recovered_scales,
        pos_acc,
        scale_acc,
        joint_acc,
    ) = dominant_trajectory(Y)

    (
        recovered_activity,
        activity_f1,
        onset_error,
        offset_error,
    ) = activity_metrics(
        td["masses"][0]
    )

    nz_frame = support_per_frame(Y)

    result = {
        "requested_snr":
            float(snr_db),
        "achieved_snr":
            float(achieved_snr),
        "converged":
            bool(stages[-1]["converged"]),
        "iterations":
            int(stages[-1]["iterations"]),
        "nz":
            int(nz_frame.sum()),
        "nz_per_frame":
            nz_frame.tolist(),
        "discrepancy":
            float(data / noise_target_realized),
        "discrepancy_expected":
            float(data / noise_target_expected),
        "discrepancy_realized": float(
            data / noise_target_realized
        ),
        "sigma": float(sigma),
        "noise_target_expected": float(
            noise_target_expected
        ),
        "noise_target_realized": float(
            noise_target_realized
        ),
        "xerr":
            float(
                relative_error(
                    Y,
                    Xtrue,
                )
            ),
        "output_snr":
            float(
                reconstruction_snr_db(
                    Sclean,
                    Shat,
                )
            ),
        "position_accuracy":
            pos_acc,
        "scale_accuracy":
            scale_acc,
        "joint_accuracy":
            joint_acc,
        "activity_f1":
            activity_f1,
        "onset_error":
            onset_error,
        "offset_error":
            offset_error,
        "activity":
            recovered_activity
            .astype(int)
            .tolist(),
        "positions":
            positions.tolist(),
        "scales":
            recovered_scales.tolist(),
        "spread":
            float(
                td["spread_value"]
            ),
        "motion":
            float(
                td["motion_value"]
            ),
        "frame4":
            frame_support(
                Y,
                4,
            ),
        "stages":
            stages,
    }

    return result


def main():
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
    Sclean = A.forward(Xtrue)

    scales = physical_scale_values(
        SCALES
    )

    shifts = np.arange(
        A.L,
        dtype=float,
    )

    rows = []

    print()
    print("Finite-track multi-SNR robustness")
    print("---------------------------------")
    print(
        f"lambda_s: {LAMBDA_S}"
    )
    print(
        f"lambda_t: {LAMBDA_T}"
    )
    print(
        f"beta path: {BETA_PATH}"
    )

    for snr in SNR_VALUES:
        r = run_snr(
            snr,
            A,
            Xtrue,
            Sclean,
            shifts,
            scales,
        )

        rows.append(r)

        print()
        print(
            f"SNR={snr:2d} dB "
            f"conv={str(r['converged']):5s} "
            f"nz={r['nz']:2d} "
            f"D/Dexp={r['discrepancy']:.3f} "
            f"Xerr={r['xerr']:.4f} "
            f"outSNR={r['output_snr']:5.2f} "
            f"joint="
            f"{100*r['joint_accuracy']:5.1f}% "
            f"F1={r['activity_f1']:.2f}"
        )

    print()
    print("Summary")
    print("-------")
    print(
        "SNR conv nz D/Dexp "
        "Xerr outSNR pos scale joint F1 "
        "on off"
    )

    for r in rows:
        print(
            f"{r['requested_snr']:3.0f} "
            f"{str(r['converged']):5s} "
            f"{r['nz']:2d} "
            f"{r['discrepancy']:5.3f} "
            f"{r['xerr']:6.4f} "
            f"{r['output_snr']:6.2f} "
            f"{100*r['position_accuracy']:5.1f}% "
            f"{100*r['scale_accuracy']:5.1f}% "
            f"{100*r['joint_accuracy']:5.1f}% "
            f"{r['activity_f1']:.2f} "
            f"{str(r['onset_error']):>3s} "
            f"{str(r['offset_error']):>3s}"
        )

    print()
    print("Support per frame")
    print("-----------------")

    for r in rows:
        print(
            f"SNR={r['requested_snr']:3.0f}: "
            f"{r['nz_per_frame']}"
        )

    print()
    print("Recovered trajectories")
    print("----------------------")

    for r in rows:
        print(
            f"SNR={r['requested_snr']:3.0f}: "
            f"pos={r['positions']} "
            f"scale={r['scales']}"
        )

    print()
    print("Continuation stages")
    print("-------------------")

    for r in rows:
        print()
        print(
            f"SNR={r['requested_snr']:.0f}"
        )

        for s in r["stages"]:
            print(
                f"  beta="
                f"{s['beta']:.4f} "
                f"conv="
                f"{str(s['converged']):5s} "
                f"iter="
                f"{s['iterations']:3d} "
                f"nz="
                f"{s['nz']:2d} "
                f"{s['nz_per_frame']}"
            )

    print()
    print("Ambiguous frame t=4")
    print("-------------------")

    for r in rows:
        print()
        print(
            f"SNR="
            f"{r['requested_snr']:.0f}"
        )

        for n, j, amp in r["frame4"]:
            print(
                f"  n={n:3d} "
                f"j={j} "
                f"amp={amp:.6f}"
            )


if __name__ == "__main__":
    main()