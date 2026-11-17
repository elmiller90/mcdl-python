import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gaussian_experiment import (
    SCALES, add_noise_snr, make_dictionary,
    make_true_coefficients, dominant_trajectory,
    reconstruction_snr_db, relative_error,
    TRUE_PHYSICAL_POSITIONS, TRUE_SCALE_INDICES,
)
from log_admm import solve_log_admm
from multiscale_convolution import MultiscaleConvolutionOperator


LAMBDAS = np.array([
    0.0005, 0.001, 0.002, 0.003, 0.005,
    0.0075, 0.010, 0.015, 0.020, 0.030,
    0.050, 0.075, 0.100,
])

SNR_DB = 14.0
LAMBDA_T = 0.10
TAU = 1e-2
RHO = 0.10
SEED = 100
OUT = Path("results/lambda_s_oracle")


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    M, N, T = 41, 96, 7
    D = make_dictionary(M, 4.0)

    A = MultiscaleConvolutionOperator(
        D=D, scales=SCALES, center=(M + 1)//2,
        N=N, normalize=True)

    X_true, _ = make_true_coefficients(
        L=A.L, M=M, J=len(SCALES), T=T)

    S_clean = A.forward(X_true)
    S_noisy, _, input_snr = add_noise_snr(
        S_clean, SNR_DB, rng)

    # Same initialization for every candidate lambda_s.
    X0 = A.adjoint(S_noisy, structured=True)
    X0 /= np.max(np.abs(X0))

    rows = []

    for lam in LAMBDAS:
        print(f"\n{'='*60}")
        print(f"lambda_s = {lam:.5g}")
        print("="*60)

        X, Y, U, info = solve_log_admm(
            X0=X0.copy(),
            S=S_noisy,
            operator=A,
            lambda_s=float(lam),
            lambda2=LAMBDA_T,
            tau=TAU,
            rho=RHO,
            max_admm_iter=300,
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

        X_hat = Y
        S_hat = A.forward(X_hat)

        x_error = relative_error(X_hat, X_true)
        s_error = relative_error(S_hat, S_clean)
        output_snr = reconstruction_snr_db(
            S_clean, S_hat)

        traj = dominant_trajectory(X_hat, M)

        pos_matches = sum(
            a == b for a, b in zip(
                traj["physical_positions"],
                TRUE_PHYSICAL_POSITIONS))

        scale_matches = sum(
            a == b for a, b in zip(
                traj["scale_indices"],
                TRUE_SCALE_INDICES))

        joint_matches = sum(
            p == pt and s == st
            for p, s, pt, st in zip(
                traj["physical_positions"],
                traj["scale_indices"],
                TRUE_PHYSICAL_POSITIONS,
                TRUE_SCALE_INDICES))

        row = {
            "lambda_s": float(lam),
            "coefficient_error": float(x_error),
            "coefficient_l2_error":
                float(np.linalg.norm(X_hat - X_true)),
            "signal_error": float(s_error),
            "output_snr_db": float(output_snr),
            "significant_coefficients":
                int(np.count_nonzero(np.abs(X_hat) > 1e-3)),
            "exact_nonzeros": int(np.count_nonzero(X_hat)),
            "position_accuracy":
                pos_matches / T,
            "scale_accuracy":
                scale_matches / T,
            "trajectory_accuracy":
                joint_matches / T,
            "converged": bool(info["converged"]),
            "iterations": int(info["iterations"]),
            "final_rho": float(info["rho"]),
            "final_a": float(info["a"]),
            "trajectory": traj,
        }

        rows.append(row)

        print(
            f"X error={x_error:.6f}, "
            f"S error={s_error:.6f}, "
            f"out SNR={output_snr:.2f} dB, "
            f"nz={row['significant_coefficients']}, "
            f"joint={joint_matches}/{T}, "
            f"conv={info['converged']}"
        )

    # Oracle selection: minimize relative Frobenius error in X.
    best = min(rows, key=lambda r: r["coefficient_error"])

    summary = {
        "input_snr_db": float(input_snr),
        "lambda_t": LAMBDA_T,
        "tau": TAU,
        "selection_rule":
            "Minimum relative Frobenius error ||Xhat-Xtrue||/||Xtrue||",
        "selected_lambda_s": best["lambda_s"],
        "selected_result": best,
        "results": rows,
    }

    with (OUT / "oracle_summary.json").open(
            "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_results(rows, best)

    print("\nOracle lambda_s selection")
    print("-------------------------")
    print(f"selected lambda_s: {best['lambda_s']:.6g}")
    print(f"coefficient error: {best['coefficient_error']:.6f}")
    print(f"signal error:      {best['signal_error']:.6f}")
    print(f"output SNR:        {best['output_snr_db']:.2f} dB")
    print(f"significant coeffs:{best['significant_coefficients']}")
    print(
        "trajectory accuracy:"
        f" {100*best['trajectory_accuracy']:.1f}%"
    )


def plot_results(rows, best):
    lam = np.array([r["lambda_s"] for r in rows])
    xe = np.array([r["coefficient_error"] for r in rows])
    se = np.array([r["signal_error"] for r in rows])
    nz = np.array([r["significant_coefficients"] for r in rows])
    snr = np.array([r["output_snr_db"] for r in rows])

    fig, ax = plt.subplots()
    ax.semilogx(lam, xe, "o-", label="Coefficient error")
    ax.axvline(best["lambda_s"], color="k", linestyle="--",
               label=f"Selected = {best['lambda_s']:.4g}")
    ax.set(xlabel="lambda_s", ylabel="Relative Frobenius error",
           title="Oracle coefficient-error selection")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "coefficient_error_vs_lambda.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.semilogx(lam, se, "o-", label="Signal error")
    ax.semilogx(lam, xe, "s-", label="Coefficient error")
    ax.set(xlabel="lambda_s", ylabel="Relative error",
           title="Recovery errors")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "errors_vs_lambda.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.semilogx(lam, snr, "o-")
    ax.axvline(best["lambda_s"], color="k", linestyle="--")
    ax.set(xlabel="lambda_s", ylabel="Output SNR (dB)",
           title="Output SNR")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUT / "output_snr_vs_lambda.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.semilogx(lam, nz, "o-")
    ax.axvline(best["lambda_s"], color="k", linestyle="--")
    ax.set(xlabel="lambda_s", ylabel="Significant coefficients",
           title="Recovered sparsity")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUT / "sparsity_vs_lambda.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    run()
