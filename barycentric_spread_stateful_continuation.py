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

BETAS = [0.075, 0.0875, 0.10, 0.1125, 0.125, 0.15]
SNR_DB, SEED = 14.0, 100
LAMBDA_S, LAMBDA_T = 0.15, 1e-5
GATE_MASS, MASS_FLOOR, SHIFT_VELOCITY = 0.05, 1e-10, 1.0
TAU, RHO = 1e-2, 0.10
MAX_ADMM_ITER, X_MAXITER = 600, 50
COEFF_THRESHOLD = 1e-3
M, N, T = 41, 96, 12
ACTIVE_FRAMES = np.arange(2, 10)
TRUE_POSITIONS = np.array([28, 29, 30, 31, 32, 33, 34, 35])
TRUE_SCALES = np.array([0, 1, 2, 3, 3, 2, 1, 0])


def make_truth(A):
    X = np.zeros((A.L, A.K, A.J, T))
    for t, pos, j in zip(ACTIVE_FRAMES, TRUE_POSITIONS, TRUE_SCALES):
        X[int(pos + M - 1), 0, j, t] = 1.0
    return X


def trajectory(Y):
    pos, scales = [], []
    for t in ACTIVE_FRAMES:
        frame = np.abs(Y[:, :, :, t])
        n, _, j = np.unravel_index(np.argmax(frame), frame.shape)
        pos.append(int(n - (M - 1)))
        scales.append(int(j))
    pos, scales = np.array(pos), np.array(scales)
    joint = np.mean((pos == TRUE_POSITIONS) & (scales == TRUE_SCALES))
    return pos.tolist(), scales.tolist(), float(joint)


def support(Y, t=None):
    Z = Y if t is None else Y[:, :, :, t]
    idx = np.argwhere(np.abs(Z) > COEFF_THRESHOLD)
    if t is None:
        return int(len(idx)), np.count_nonzero(
            np.abs(Y) > COEFF_THRESHOLD, axis=(0, 1, 2)
        ).tolist()
    return [(int(n), int(j), float(Y[n, k, j, t])) for n, k, j in idx]


def callback(beta, shifts, scales):
    return make_barycentric_transition_regularizer(
        shift_coordinates=shifts, scale_coordinates=scales,
        gate_mass=GATE_MASS, mass_floor=MASS_FLOOR,
        shift_velocity=SHIFT_VELOCITY, spread_weight=beta,
    )


def solve_beta(beta, state0, Xbase, A, Snoisy, Sclean, Xtrue,
               noise_target, shifts, scales):
    reg = callback(beta, shifts, scales)
    start = time.perf_counter()
    X, Y, U, info = solve_log_admm_stateful(
        X0=Xbase if state0 is None else state0["X"],
        S=Snoisy, operator=A, lambda_s=LAMBDA_S,
        lambda2=LAMBDA_T, tau=TAU, rho=RHO,
        max_admm_iter=MAX_ADMM_ITER, x_maxiter=X_MAXITER,
        abs_tol=1e-4, rel_tol=1e-3, adaptive_rho=True,
        rho_mu=10, rho_tau=2, a_min=1e-4,
        continuation_iters=100, a_fraction=.95, verbose=False,
        temporal_regularizer=reg, state0=state0,
        continue_iteration=False,
    )
    elapsed = time.perf_counter() - start
    Shat = A.forward(Y)
    data = 0.5 * np.sum((Shat - Snoisy) ** 2)
    _, _, td = barycentric_transition_value_grad(
        Y, shift_coordinates=shifts, scale_coordinates=scales,
        gate_mass=GATE_MASS, mass_floor=MASS_FLOOR,
        shift_velocity=SHIFT_VELOCITY, spread_weight=beta,
    )
    pos, sc, joint = trajectory(Y)
    nz, nz_frame = support(Y)
    return {
        "beta": beta, "X": X, "Y": Y, "U": U,
        "state": info["state"], "converged": info["converged"],
        "iterations": info["iterations"], "nz": nz,
        "nz_per_frame": nz_frame, "discrepancy": data / noise_target,
        "xerr": relative_error(Y, Xtrue),
        "snr": reconstruction_snr_db(Sclean, Shat),
        "joint": joint, "positions": pos, "scales": sc,
        "frame4": support(Y, 4),
        "motion": td["motion_value"], "spread": td["spread_value"],
        "runtime": elapsed,
    }


def run_path(name, betas, initial_state, Xbase, *args):
    print(f"\n{name}\n{'-' * len(name)}")
    rows, state = [], initial_state
    for beta in betas:
        r = solve_beta(beta, state, Xbase, *args)
        rows.append(r)
        state = r["state"]
        print(
            f"beta={beta:7.4f} conv={str(r['converged']):5s} "
            f"iter={r['iterations']:3d} nz={r['nz']:2d} "
            f"D/Dn={r['discrepancy']:.3f} Xerr={r['xerr']:.4f} "
            f"SNR={r['snr']:5.2f} joint={100*r['joint']:5.1f}% "
            f"spread={r['spread']:.4f}"
        )
    return rows


def print_details(name, rows):
    print(f"\n{name}\n{'=' * len(name)}")
    for r in rows:
        print(f"\nbeta={r['beta']:.4f}")
        print("nz/frame:", r["nz_per_frame"])
        print("scales:  ", r["scales"])
        print("t=4:")
        for n, j, amp in r["frame4"]:
            print(f"  n={n:3d} j={j} amp={amp:.6f}")


def main():
    rng = np.random.default_rng(SEED)
    D = make_dictionary(M, 4.0)
    A = MultiscaleConvolutionOperator(
        D=D, scales=SCALES, center=(M + 1) // 2, N=N, normalize=True
    )
    Xtrue = make_truth(A)
    Sclean = A.forward(Xtrue)
    Snoisy, noise, snr = add_noise_snr(Sclean, SNR_DB, rng)
    noise_target = 0.5 * np.sum(noise * noise)
    scales = physical_scale_values(SCALES)
    shifts = np.arange(A.L, dtype=float)
    Xbase = np.maximum(A.adjoint(Snoisy, structured=True), 0.0)
    if Xbase.max() > 0:
        Xbase /= Xbase.max()

    print("\nFull-state spread continuation")
    print("------------------------------")
    print(f"SNR: {snr:.2f} dB  lambda_s: {LAMBDA_S}  lambda_t: {LAMBDA_T}")

    common = (A, Snoisy, Sclean, Xtrue, noise_target, shifts, scales)
    forward = run_path("Forward path", BETAS, None, Xbase, *common)
    reverse = run_path(
        "Reverse path", list(reversed(BETAS)),
        forward[-1]["state"], Xbase, *common,
    )
    print_details("Forward details", forward)
    print_details("Reverse details", reverse)


if __name__ == "__main__":
    main()
