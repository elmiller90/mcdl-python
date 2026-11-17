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


# ---------------------------------------------------------
# Experiment
# ---------------------------------------------------------

SNR_DB = 14.0
SEED = 100

LAMBDA_S = 0.15
LAMBDA_T = 0.000

GATE_MASS = 0.05
MASS_FLOOR = 1e-10
SHIFT_VELOCITY = 1.0

TAU = 1e-2
RHO = 0.10

MAX_ADMM_ITER = 600
X_MAXITER = 50

ACTIVITY_THRESHOLD = 0.05
COEFF_THRESHOLD = 1e-3

M = 41
N = 96
T = 12

ACTIVE_FRAMES = np.arange(2, 10)

TRUE_PHYSICAL_POSITIONS = np.array([
    28, 29, 30, 31, 32, 33, 34, 35
])

TRUE_SCALE_INDICES = np.array([
    0, 1, 2, 3, 3, 2, 1, 0
])


# ---------------------------------------------------------
# Truth
# ---------------------------------------------------------

def physical_to_coefficient_position(position, M):
    """
    Match the canonical Gaussian position convention:
        physical 28 -> coefficient index 68 for M=41.
    """
    return int(position + M - 1)


def make_finite_track_coefficients(A):
    X = np.zeros(
        (A.L, A.K, A.J, T),
        dtype=np.float64,
    )

    if A.K != 1:
        raise ValueError(
            "Finite-track test expects one dictionary atom"
        )

    for t, pos, j in zip(
        ACTIVE_FRAMES,
        TRUE_PHYSICAL_POSITIONS,
        TRUE_SCALE_INDICES,
    ):
        n = physical_to_coefficient_position(
            pos, M
        )

        X[n, 0, j, t] = 1.0

    return X


# ---------------------------------------------------------
# Activity metrics
# ---------------------------------------------------------

def activity_metrics(masses):
    true_active = np.zeros(T, dtype=bool)
    true_active[ACTIVE_FRAMES] = True

    recovered = (
        np.asarray(masses)
        > ACTIVITY_THRESHOLD
    )

    tp = int(np.count_nonzero(
        recovered & true_active
    ))

    fp = int(np.count_nonzero(
        recovered & ~true_active
    ))

    fn = int(np.count_nonzero(
        ~recovered & true_active
    ))

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
        if precision + recall else 0.0
    )

    true_idx = np.flatnonzero(true_active)
    rec_idx = np.flatnonzero(recovered)

    onset_error = (
        int(rec_idx[0] - true_idx[0])
        if rec_idx.size else None
    )

    offset_error = (
        int(rec_idx[-1] - true_idx[-1])
        if rec_idx.size else None
    )

    # Interior inactive frames between first and last
    # recovered active frames.
    if rec_idx.size >= 2:
        interior = recovered[
            rec_idx[0]:rec_idx[-1] + 1
        ]

        interior_gaps = int(
            np.count_nonzero(~interior)
        )
    else:
        interior_gaps = 0

    return {
        "true": true_active,
        "recovered": recovered,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "onset_error": onset_error,
        "offset_error": offset_error,
        "interior_gaps": interior_gaps,
    }


# ---------------------------------------------------------
# Active-frame trajectory
# ---------------------------------------------------------

def dominant_frame_state(Y, t):
    """
    Return dominant coefficient location for one frame.
    """
    frame = np.abs(Y[:, :, :, t])

    index = np.unravel_index(
        np.argmax(frame),
        frame.shape,
    )

    n, k, j = index

    physical_position = n - (M - 1)

    return {
        "n": int(n),
        "k": int(k),
        "j": int(j),
        "physical_position":
            int(physical_position),
        "amplitude":
            float(Y[n, k, j, t]),
    }


def active_trajectory_metrics(Y):
    states = [
        dominant_frame_state(Y, int(t))
        for t in ACTIVE_FRAMES
    ]

    positions = np.array([
        s["physical_position"]
        for s in states
    ])

    scales = np.array([
        s["j"]
        for s in states
    ])

    position_matches = (
        positions
        == TRUE_PHYSICAL_POSITIONS
    )

    scale_matches = (
        scales
        == TRUE_SCALE_INDICES
    )

    joint = (
        position_matches
        & scale_matches
    )

    return {
        "states": states,
        "positions": positions,
        "scales": scales,
        "position_accuracy":
            float(position_matches.mean()),
        "scale_accuracy":
            float(scale_matches.mean()),
        "joint_accuracy":
            float(joint.mean()),
        "mean_position_error":
            float(np.mean(
                np.abs(
                    positions
                    - TRUE_PHYSICAL_POSITIONS
                )
            )),
    }


# ---------------------------------------------------------
# Support metrics
# ---------------------------------------------------------

def support_metrics(Y):
    active_per_frame = np.count_nonzero(
        np.abs(Y) > COEFF_THRESHOLD,
        axis=(0, 1, 2),
    )

    significant = np.argwhere(
        np.abs(Y) > COEFF_THRESHOLD
    )

    return {
        "per_frame": active_per_frame,
        "total": int(len(significant)),
        "indices": significant,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

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

    X_true = make_finite_track_coefficients(A)

    S_clean = A.forward(X_true)

    S_noisy, noise, achieved_snr, sigma = add_noise_snr(
        S_clean,
        SNR_DB,
        rng,
        return_sigma=True,
    )

    noise_target_expected = float(
        0.5 * S_noisy.size * sigma**2
    )

    noise_target_realized = float(
        0.5 * np.sum(noise * noise)
    )

    # -----------------------------------------------------
    # Barycentric temporal callback
    # -----------------------------------------------------

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
    # Solve
    # -----------------------------------------------------

    print()
    print("Barycentric finite-track experiment")
    print("-----------------------------------")
    print(
        f"input SNR:       "
        f"{achieved_snr:.2f} dB"
    )
    print(f"lambda_s:        {LAMBDA_S}")
    print(f"lambda_t:        {LAMBDA_T}")
    print(f"gate mass:       {GATE_MASS}")
    print(f"mass floor:      {MASS_FLOOR}")
    print(
        "true activity:   ",
        [
            int(t in ACTIVE_FRAMES)
            for t in range(T)
        ],
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
        temporal_regularizer=
            temporal_regularizer,
    )

    runtime = (
        time.perf_counter() - start
    )

    # -----------------------------------------------------
    # Reconstruction
    # -----------------------------------------------------

    S_hat = A.forward(Y)

    residual = S_hat - S_noisy

    data_fidelity = float(
        0.5 * np.sum(
            residual * residual
        )
    )

    discrepancy = (
        data_fidelity / noise_target_expected
    )

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

    temporal_value, _, td = (
        barycentric_transition_value_grad(
            Y,
            shift_coordinates=
                shift_coordinates,
            scale_coordinates=scales,
            gate_mass=GATE_MASS,
            mass_floor=MASS_FLOOR,
            shift_velocity=SHIFT_VELOCITY,
        )
    )

    masses = td["masses"][0]

    activity = activity_metrics(
        masses
    )

    trajectory = active_trajectory_metrics(
        Y
    )

    support = support_metrics(
        Y
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
        f"runtime:          "
        f"{runtime:.2f} s"
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
        f"{support['total']}"
    )
    print(
        f"D/Dexpected:         "
        f"{discrepancy:.4f}"
    )

    print()
    print("Activity")
    print("--------")
    print(
        "frame masses:     ",
        np.array2string(
            masses,
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
        "recovered active: ",
        activity[
            "recovered"
        ].astype(int).tolist(),
    )
    print(
        f"precision:        "
        f"{activity['precision']:.3f}"
    )
    print(
        f"recall:           "
        f"{activity['recall']:.3f}"
    )
    print(
        f"F1:               "
        f"{activity['f1']:.3f}"
    )
    print(
        f"onset error:      "
        f"{activity['onset_error']}"
    )
    print(
        f"offset error:     "
        f"{activity['offset_error']}"
    )
    print(
        f"interior gaps:    "
        f"{activity['interior_gaps']}"
    )

    print()
    print("Active trajectory")
    print("-----------------")
    print(
        "true position:    ",
        TRUE_PHYSICAL_POSITIONS.tolist(),
    )
    print(
        "recovered pos:    ",
        trajectory[
            "positions"
        ].tolist(),
    )
    print(
        "true scale:       ",
        TRUE_SCALE_INDICES.tolist(),
    )
    print(
        "recovered scale:  ",
        trajectory[
            "scales"
        ].tolist(),
    )
    print(
        f"position accuracy:"
        f" {100*trajectory['position_accuracy']:.1f}%"
    )
    print(
        f"scale accuracy:   "
        f"{100*trajectory['scale_accuracy']:.1f}%"
    )
    print(
        f"joint accuracy:   "
        f"{100*trajectory['joint_accuracy']:.1f}%"
    )

    print()
    print("Support")
    print("-------")
    print(
        "nz per frame:     ",
        support["per_frame"].tolist(),
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
        f"{LAMBDA_T*temporal_value:.6e}"
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
        "pair weights:     ",
        np.array2string(
            td["pair_weights"][0],
            precision=6,
        ),
    )

    print()
    print("Significant coefficients")
    print("------------------------")

    for n, k, j, t in support["indices"]:
        print(
            f"t={t:2d} "
            f"n={n:3d} "
            f"j={j} "
            f"amp="
            f"{Y[n,k,j,t]:.6f}"
        )


if __name__ == "__main__":
    main()
