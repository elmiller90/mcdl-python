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


LAMBDAS = np.arange(0.04, 0.1601, 0.005)

SNR_DB = 14.0
LAMBDA_T = 0.0
TAU = 1e-2
RHO = 0.10
SEED = 100

OUT = Path("results/lambda_s_oracle_fine")


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    M, N, T = 41, 96, 7

    D = make_dictionary(M, 4.0)

    A = MultiscaleConvolutionOperator(
        D=D,
        scales=SCALES,
        center=(M + 1)//2,
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

    S_noisy, _, input_snr = add_noise_snr(
        S_clean,
        SNR_DB,
        rng,
    )

    X0 = A.adjoint(
        S_noisy,
        structured=True,
    )
    X0 /= np.max(np.abs(X0))

    rows = []

    for lam in LAMBDAS:
        lam = float(lam)

        print(f"\n{'='*58}")
        print(f"lambda_s={lam:.4f}, lambda_t=0")
        print("="*58)

        _, Y, _, info = solve_log_admm(
            X0=X0.copy(),
            S=S_noisy,
            operator=A,
            lambda_s=lam,
            lambda2=LAMBDA_T,
            tau=TAU,
            rho=RHO,
            max_admm_iter=400,
            x_maxiter=50,
            abs_tol=1e-4,
            rel_tol=1e-3,
            adaptive_rho=True,
            rho_mu=10.0,
            rho_tau=2.0,
            a_min=1e-4,
            continuation_iters=100,
            a_fraction=0.95,
            verbose=False,
        )

        S_hat = A.forward(Y)

        x_abs_error = np.linalg.norm(Y - X_true)
        x_rel_error = relative_error(Y, X_true)
        s_error = relative_error(S_hat, S_clean)
        output_snr = reconstruction_snr_db(S_clean, S_hat)

        traj = dominant_trajectory(Y, M)

        pos_matches = sum(
            a == b
            for a, b in zip(
                traj["physical_positions"],
                TRUE_PHYSICAL_POSITIONS,
            )
        )

        scale_matches = sum(
            a == b
            for a, b in zip(
                traj["scale_indices"],
                TRUE_SCALE_INDICES,
            )
        )

        joint_matches = sum(
            p == pt and s == st
            for p, s, pt, st in zip(
                traj["physical_positions"],
                traj["scale_indices"],
                TRUE_PHYSICAL_POSITIONS,
                TRUE_SCALE_INDICES,
            )
        )

        row = {
            "lambda_s": lam,
            "coefficient_error": float(x_rel_error),
            "coefficient_l2_error": float(x_abs_error),
            "signal_error": float(s_error),
            "output_snr_db": float(output_snr),
            "significant_coefficients":
                int(np.count_nonzero(np.abs(Y) > 1e-3)),
            "exact_nonzeros": int(np.count_nonzero(Y)),
            "position_accuracy": pos_matches / T,
            "scale_accuracy": scale_matches / T,
            "trajectory_accuracy": joint_matches / T,
            "converged": bool(info["converged"]),
            "iterations": int(info["iterations"]),
            "final_rho": float(info["rho"]),
            "final_a": float(info["a"]),
            "trajectory": traj,
        }

        rows.append(row)

        print(
            f"Xerr={x_rel_error:.5f}  "
            f"Serr={s_error:.5f}  "
            f"SNRout={output_snr:.2f}  "
            f"nz={row['significant_coefficients']:3d}  "
            f"path={joint_matches}/{T}  "
            f"iter={info['iterations']:3d}  "
            f"conv={info['converged']}"
        )

    best = min(
        rows,
        key=lambda r: r["coefficient_error"],
    )

    converged_rows = [
        r for r in rows if r["converged"]
    ]

    best_converged = (
        min(
            converged_rows,
            key=lambda r: r["coefficient_error"],
        )
        if converged_rows else None
    )

    summary = {
        "input_snr_db": float(input_snr),
        "lambda_t": LAMBDA_T,
        "tau": TAU,
        "lambda_range": [
            float(LAMBDAS[0]),
            float(LAMBDAS[-1]),
        ],
        "lambda_step": 0.005,
        "selection_rule":
            "Minimum relative Frobenius coefficient error",
        "selected_lambda_s": best["lambda_s"],
        "selected_result": best,
        "selected_converged_lambda_s":
            None if best_converged is None
            else best_converged["lambda_s"],
        "selected_converged_result":
            best_converged,
        "results": rows,
    }

    with (OUT / "oracle_fine_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    plot_results(rows, best)

    print("\nFine oracle selection")
    print("---------------------")
    print(f"lambda_t:          {LAMBDA_T}")
    print(f"selected lambda_s: {best['lambda_s']:.4f}")
    print(f"X relative error:  {best['coefficient_error']:.6f}")
    print(f"X L2 error:        {best['coefficient_l2_error']:.6f}")
    print(f"signal error:      {best['signal_error']:.6f}")
    print(f"output SNR:        {best['output_snr_db']:.2f} dB")
    print(f"nonzeros:          {best['exact_nonzeros']}")
    print(f"trajectory acc.:   {100*best['trajectory_accuracy']:.1f}%")
    print(f"converged:         {best['converged']}")


def plot_results(rows, best):
    lam = np.array([r["lambda_s"] for r in rows])
    xe = np.array([r["coefficient_error"] for r in rows])
    se = np.array([r["signal_error"] for r in rows])
    snr = np.array([r["output_snr_db"] for r in rows])
    nz = np.array([r["significant_coefficients"] for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(lam, xe, "o-")
    axes[0, 0].axvline(best["lambda_s"], color="k", ls="--")
    axes[0, 0].set(
        xlabel="lambda_s",
        ylabel="Relative coefficient error",
        title="Oracle coefficient error",
    )

    axes[0, 1].plot(lam, se, "o-")
    axes[0, 1].axvline(best["lambda_s"], color="k", ls="--")
    axes[0, 1].set(
        xlabel="lambda_s",
        ylabel="Relative signal error",
        title="Signal reconstruction error",
    )

    axes[1, 0].plot(lam, snr, "o-")
    axes[1, 0].axvline(best["lambda_s"], color="k", ls="--")
    axes[1, 0].set(
        xlabel="lambda_s",
        ylabel="Output SNR (dB)",
        title="Output SNR",
    )

    axes[1, 1].plot(lam, nz, "o-")
    axes[1, 1].axvline(best["lambda_s"], color="k", ls="--")
    axes[1, 1].set(
        xlabel="lambda_s",
        ylabel="Significant coefficients",
        title="Sparsity",
    )

    for ax in axes.flat:
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(
        OUT / "fine_oracle_summary.png",
        dpi=180,
    )
    plt.close(fig)


if __name__ == "__main__":
    run()