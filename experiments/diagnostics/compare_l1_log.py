import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gaussian_experiment import (
    SCALES, TRUE_PHYSICAL_POSITIONS, TRUE_SCALE_INDICES,
    add_noise_snr, dominant_trajectory, make_dictionary,
    make_true_coefficients, reconstruction_snr_db, relative_error,
)
from log_admm import solve_log_admm
from multiscale_convolution import MultiscaleConvolutionOperator
from sparse_admm import solve_l1_admm


SNR_DB = 14.0
LAMBDA_S = 0.03
LAMBDA_T = 0.10
TAU = 1e-2
RHO = 0.10
SEED = 100
OUTPUT_DIR = Path("results/l1_vs_log")


def metrics(name, Y, S_clean, S_noisy, operator, info):
    S_hat = operator.forward(Y)
    traj = dominant_trajectory(Y, operator.M)

    pos = sum(a == b for a, b in zip(
        traj["physical_positions"], TRUE_PHYSICAL_POSITIONS))
    scale = sum(a == b for a, b in zip(
        traj["scale_indices"], TRUE_SCALE_INDICES))
    joint = sum(
        p == pt and s == st
        for p, s, pt, st in zip(
            traj["physical_positions"], traj["scale_indices"],
            TRUE_PHYSICAL_POSITIONS, TRUE_SCALE_INDICES
        )
    )

    return {
        "method": name,
        "output_snr_db": reconstruction_snr_db(S_clean, S_hat),
        "signal_error": relative_error(S_hat, S_clean),
        "significant_coefficients": int(np.count_nonzero(np.abs(Y) > 1e-3)),
        "exact_nonzeros": int(np.count_nonzero(Y)),
        "position_accuracy": pos / len(TRUE_PHYSICAL_POSITIONS),
        "scale_accuracy": scale / len(TRUE_SCALE_INDICES),
        "trajectory_accuracy": joint / len(TRUE_SCALE_INDICES),
        "converged": bool(info["converged"]),
        "iterations": int(info["iterations"]),
        "final_rho": float(info["rho"]),
        "trajectory": traj,
        "S_recovered": S_hat,
    }


def plot_data(S_clean, S_noisy, l1, log):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharex=True, sharey=True)
    data = [
        ("Exact", S_clean),
        ("Noisy", S_noisy),
        ("L1", l1["S_recovered"]),
        ("Log", log["S_recovered"]),
    ]
    vmin = min(x.min() for _, x in data)
    vmax = max(x.max() for _, x in data)

    for ax, (title, x) in zip(axes, data):
        im = ax.imshow(x, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Time frame")

    axes[0].set_ylabel("Signal position")
    fig.colorbar(im, ax=axes, shrink=.8)
    fig.subplots_adjust(left=.06, right=.92, bottom=.15, top=.88, wspace=.12)
    fig.savefig(OUTPUT_DIR / "data_comparison.png", dpi=180)
    plt.close(fig)


def plot_trajectories(l1, log):
    t = np.arange(len(TRUE_PHYSICAL_POSITIONS))
    scale_values = [p/q for p, q in SCALES]
    true_scales = [scale_values[j] for j in TRUE_SCALE_INDICES]

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(t, TRUE_PHYSICAL_POSITIONS, "ko-", label="True")
    axes[0].plot(t, l1["trajectory"]["physical_positions"], "s--", label="L1")
    axes[0].plot(t, log["trajectory"]["physical_positions"], "^--", label="Log")
    axes[0].set_ylabel("Position")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, true_scales, "ko-", label="True")
    axes[1].plot(t, [scale_values[j] for j in l1["trajectory"]["scale_indices"]],
                 "s--", label="L1")
    axes[1].plot(t, [scale_values[j] for j in log["trajectory"]["scale_indices"]],
                 "^--", label="Log")
    axes[1].set(xlabel="Time frame", ylabel="Scale")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "trajectory_comparison.png", dpi=180)
    plt.close(fig)


def plot_amplitudes(l1, log):
    t = np.arange(len(TRUE_PHYSICAL_POSITIONS))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(1.0, color="k", linestyle="--", label="True amplitude")
    ax.plot(t, l1["trajectory"]["amplitudes"], "s-", label="L1")
    ax.plot(t, log["trajectory"]["amplitudes"], "^-", label="Log")
    ax.set(xlabel="Time frame", ylabel="Dominant coefficient amplitude",
           title="Amplitude bias")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "amplitude_comparison.png", dpi=180)
    plt.close(fig)


def strip_arrays(result):
    result = dict(result)
    result.pop("S_recovered")
    return result


def run_comparison():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    M, N, T = 41, 96, 7
    D = make_dictionary(M, 4.0)
    operator = MultiscaleConvolutionOperator(
        D=D, scales=SCALES, center=(M + 1)//2, N=N, normalize=True)

    X_true, _ = make_true_coefficients(
        L=operator.L, M=M, J=len(SCALES), T=T)

    S_clean = operator.forward(X_true)
    S_noisy, _, input_snr = add_noise_snr(S_clean, SNR_DB, rng)

    # Identical initialization for both methods.
    X0 = operator.adjoint(S_noisy, structured=True)
    X0 /= np.max(np.abs(X0))

    print("\nRunning L1 ADMM...")
    _, Y_l1, _, info_l1 = solve_l1_admm(
        X0=X0.copy(), S=S_noisy, operator=operator,
        lambda1=LAMBDA_S, lambda2=LAMBDA_T, tau=TAU, rho=RHO,
        max_admm_iter=200, x_maxiter=50,
        nonnegative=True, adaptive_rho=True, verbose=False)

    print("Running log-penalty ADMM...")
    _, Y_log, _, info_log = solve_log_admm(
        X0=X0.copy(), S=S_noisy, operator=operator,
        lambda_s=LAMBDA_S, lambda2=LAMBDA_T, tau=TAU, rho=RHO,
        max_admm_iter=200, x_maxiter=50,
        adaptive_rho=True, a_min=1e-4, continuation_iters=100,
        a_fraction=0.95, verbose=False)

    l1 = metrics("L1", Y_l1, S_clean, S_noisy, operator, info_l1)
    log = metrics("Log", Y_log, S_clean, S_noisy, operator, info_log)

    summary = {
        "snr_db": float(input_snr),
        "lambda_s": LAMBDA_S,
        "lambda_t": LAMBDA_T,
        "tau": TAU,
        "l1": strip_arrays(l1),
        "log": strip_arrays(log),
        "log_final_a": float(info_log["a"]),
    }

    with (OUTPUT_DIR / "comparison.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_data(S_clean, S_noisy, l1, log)
    plot_trajectories(l1, log)
    plot_amplitudes(l1, log)

    print("\nL1 vs log comparison")
    print("--------------------")
    print("method  outSNR  sig.err   nz   pos  scale joint iter")
    for r in [l1, log]:
        print(
            f"{r['method']:6s} "
            f"{r['output_snr_db']:6.2f}  "
            f"{r['signal_error']:7.4f}  "
            f"{r['significant_coefficients']:3d}  "
            f"{100*r['position_accuracy']:3.0f}%  "
            f"{100*r['scale_accuracy']:4.0f}%  "
            f"{100*r['trajectory_accuracy']:4.0f}%  "
            f"{r['iterations']:4d}"
        )

    print("\nDominant amplitudes")
    print("L1: ", [round(x, 4) for x in l1["trajectory"]["amplitudes"]])
    print("Log:", [round(x, 4) for x in log["trajectory"]["amplitudes"]])
    print(f"Final log a: {info_log['a']:.6g}")


if __name__ == "__main__":
    run_comparison()
    