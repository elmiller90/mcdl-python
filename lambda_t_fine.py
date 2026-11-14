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
from softmin import softmin
from x_update import structured_to_softmin


LAMBDA_S = .135
LAMBDA_T_VALUES = np.concatenate(([0.0], np.arange(.005, .0751, .005)))

SNR_DB, TAU, RHO, SEED = 14.0, 1e-2, .10, 100

OUT = Path("results/lambda_t_fine")
CACHE = OUT / "cache"
CACHE_VERSION = "v1_warm_desc_ls0135_snr14_tau1e-2_seed100"


def cache_path(lam):
    return CACHE / f"{CACHE_VERSION}_lt_{lam:.6f}.npz"


def save_cache(lam, Y, info):
    np.savez_compressed(
        cache_path(lam), Y=Y,
        converged=np.array(info["converged"]),
        iterations=np.array(info["iterations"]),
        rho=np.array(info["rho"]),
        a=np.array(info["a"]),
    )


def load_cache(lam):
    p = cache_path(lam)
    if not p.exists():
        return None

    z = np.load(p)
    return z["Y"], {
        "converged": bool(z["converged"].item()),
        "iterations": int(z["iterations"].item()),
        "rho": float(z["rho"].item()),
        "a": float(z["a"].item()),
    }


def data_fidelity(Y, S, A):
    r = A.forward(Y) - S
    return float(.5*np.sum(r*r))


def temporal_energy(Y, A):
    X = structured_to_softmin(Y)

    raw = softmin(
        X, K=A.K, J=A.J, tau=TAU,
        pad_mode="zero", normalize=False,
        return_gradient=False,
    )

    # Softmin lower bound: -K(T-1) tau log(9).
    correction = A.K*(Y.shape[-1]-1)*TAU*np.log(9.0)
    return float(raw), float(max(raw + correction, 0.0))


def trajectory_accuracy(Y, M):
    t = dominant_trajectory(Y, M)

    joint = sum(
        p == pt and s == st
        for p, s, pt, st in zip(
            t["physical_positions"], t["scale_indices"],
            TRUE_PHYSICAL_POSITIONS, TRUE_SCALE_INDICES)
    )

    return joint / len(TRUE_SCALE_INDICES)


def normalize_log(v):
    v = np.log10(np.maximum(np.asarray(v, float), 1e-15))
    lo, hi = v.min(), v.max()
    return np.zeros_like(v) if hi == lo else (v-lo)/(hi-lo)


def smooth3(x):
    y = np.array(x, float, copy=True)
    if len(y) >= 3:
        y[1:-1] = (x[:-2] + x[1:-1] + x[2:])/3
    return y


def curvature(x, y):
    k = np.full(len(x), np.nan)

    for i in range(1, len(x)-1):
        xp, yp = (x[i+1]-x[i-1])/2, (y[i+1]-y[i-1])/2
        xpp = x[i+1] - 2*x[i] + x[i-1]
        ypp = y[i+1] - 2*y[i] + y[i-1]
        den = (xp*xp + yp*yp)**1.5

        if den > 0:
            k[i] = abs(xp*ypp - yp*xpp)/den

    return k


def lcurve_select(rows):
    curve = [
        r for r in rows
        if r["converged"]
        and r["lambda_t"] > 0
        and r["data_fidelity"] > 0
        and r["temporal_energy"] > 0
    ]

    curve.sort(key=lambda r: r["lambda_t"])

    if len(curve) < 3:
        return None, curve

    x = normalize_log([r["data_fidelity"] for r in curve])
    y = normalize_log([r["temporal_energy"] for r in curve])

    xs, ys = smooth3(x), smooth3(y)
    k = curvature(xs, ys)

    for i, r in enumerate(curve):
        r["lcurve_x"] = float(x[i])
        r["lcurve_y"] = float(y[i])
        r["lcurve_x_smoothed"] = float(xs[i])
        r["lcurve_y_smoothed"] = float(ys[i])
        r["curvature"] = None if not np.isfinite(k[i]) else float(k[i])

    valid = [r for r in curve if r["curvature"] is not None]
    return (
        max(valid, key=lambda r: r["curvature"]) if valid else None,
        curve,
    )


def plot_results(rows, oracle, selected):
    positive = [r for r in rows if r["lambda_t"] > 0]

    lam = np.array([r["lambda_t"] for r in positive])
    xe = np.array([r["coefficient_error"] for r in positive])
    snr = np.array([r["output_snr_db"] for r in positive])
    acc = 100*np.array([r["trajectory_accuracy"] for r in positive])

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(lam, xe, "o-")
    axes[0].set_ylabel("Coefficient error")

    axes[1].plot(lam, snr, "o-")
    axes[1].set_ylabel("Output SNR (dB)")

    axes[2].plot(lam, acc, "o-")
    axes[2].set(xlabel="lambda_t", ylabel="Trajectory accuracy (%)")
    axes[2].set_ylim(-5, 105)

    for ax in axes:
        ax.grid(True)
        ax.axvline(oracle["lambda_t"], color="k", ls="--", label="Oracle")

        if selected:
            ax.axvline(
                selected["lambda_t"], color="r", ls=":", label="L-curve")

    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "lambda_t_fine_summary.png", dpi=180)
    plt.close(fig)


def plot_lcurve(curve, selected):
    if selected is None:
        return

    x = [r["lcurve_x_smoothed"] for r in curve]
    y = [r["lcurve_y_smoothed"] for r in curve]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x, y, "o-")

    for xi, yi, r in zip(x, y, curve):
        ax.annotate(
            f"{r['lambda_t']:.3g}", (xi, yi),
            xytext=(4, 4), textcoords="offset points", fontsize=8)

    i = next(
        i for i, r in enumerate(curve)
        if r["lambda_t"] == selected["lambda_t"])

    ax.plot(x[i], y[i], "r*", ms=16,
            label=f"Selected={selected['lambda_t']:.3g}")

    ax.set(
        xlabel="Normalized log data fidelity",
        ylabel="Normalized log temporal energy",
        title="Refined Stage-2 L-curve")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "lambda_t_fine_lcurve.png", dpi=180)
    plt.close(fig)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    M, N, T = 41, 96, 7

    D = make_dictionary(M, 4.0)

    A = MultiscaleConvolutionOperator(
        D=D, scales=SCALES, center=(M+1)//2,
        N=N, normalize=True)

    X_true, _ = make_true_coefficients(
        L=A.L, M=M, J=len(SCALES), T=T)

    S_clean = A.forward(X_true)
    S_noisy, _, input_snr = add_noise_snr(S_clean, SNR_DB, rng)

    X0 = A.adjoint(S_noisy, structured=True)
    X0 /= np.max(np.abs(X0))

    X_start = X0.copy()
    previous_lambda = None
    rows = []

    # Strong-to-weak temporal continuation.
    for lam in np.sort(LAMBDA_T_VALUES)[::-1]:
        lam = float(lam)

        print(f"\n{'='*58}")
        print(f"lambda_s={LAMBDA_S:.3f}, lambda_t={lam:.4f}")
        print("="*58)

        cached = load_cache(lam)

        if cached:
            Y, info = cached
            print(
                f"Loaded cache: iter={info['iterations']}, "
                f"conv={info['converged']}")
        else:
            _, Y, _, info = solve_log_admm(
                X0=X_start.copy(), S=S_noisy, operator=A,
                lambda_s=LAMBDA_S, lambda2=lam,
                tau=TAU, rho=RHO,
                max_admm_iter=600, x_maxiter=50,
                abs_tol=1e-4, rel_tol=1e-3,
                adaptive_rho=True, rho_mu=10, rho_tau=2,
                a_min=1e-4, continuation_iters=100,
                a_fraction=.95, verbose=False)

            # Cache immediately after expensive solve.
            save_cache(lam, Y, info)

        raw, temp = temporal_energy(Y, A)
        S_hat = A.forward(Y)

        row = {
            "lambda_t": lam,
            "warm_start_from_lambda": previous_lambda,
            "converged": bool(info["converged"]),
            "iterations": int(info["iterations"]),
            "final_rho": float(info["rho"]),
            "data_fidelity": data_fidelity(Y, S_noisy, A),
            "softmin_raw": raw,
            "temporal_energy": temp,
            "coefficient_error": relative_error(Y, X_true),
            "signal_error": relative_error(S_hat, S_clean),
            "output_snr_db": reconstruction_snr_db(S_clean, S_hat),
            "trajectory_accuracy": trajectory_accuracy(Y, M),
            "significant_coefficients":
                int(np.count_nonzero(np.abs(Y) > 1e-3)),
        }
        rows.append(row)

        print(
            f"D={row['data_fidelity']:.4e}  "
            f"T={temp:.4e}  "
            f"Xerr={row['coefficient_error']:.5f}  "
            f"path={100*row['trajectory_accuracy']:.0f}%  "
            f"iter={info['iterations']:3d}  "
            f"conv={info['converged']}"
        )

        if info["converged"]:
            X_start = Y.copy()
            previous_lambda = lam
        else:
            print("Unconverged solution not propagated.")

    rows.sort(key=lambda r: r["lambda_t"])

    valid = [r for r in rows if r["converged"]]

    if not valid:
        raise RuntimeError("No converged solutions.")

    oracle = min(valid, key=lambda r: r["coefficient_error"])
    selected, curve = lcurve_select(rows)

    summary = {
        "input_snr_db": float(input_snr),
        "lambda_s": LAMBDA_S,
        "tau": TAU,
        "oracle_lambda_t": oracle["lambda_t"],
        "oracle_result": oracle,
        "lcurve_lambda_t":
            None if selected is None else selected["lambda_t"],
        "lcurve_result": selected,
        "results": rows,
        "curve_results": curve,
    }

    with (OUT / "lambda_t_fine_summary.json").open(
            "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_results(rows, oracle, selected)
    plot_lcurve(curve, selected)

    print("\nRefined Stage-2 selection")
    print("-------------------------")
    print(f"oracle lambda_T: {oracle['lambda_t']:.4f}")
    print(f"oracle X error:  {oracle['coefficient_error']:.6f}")

    if selected:
        print(f"L-curve lambda_T: {selected['lambda_t']:.4f}")
        print(f"curvature:        {selected['curvature']:.6f}")


if __name__ == "__main__":
    run()
