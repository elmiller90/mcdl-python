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
LAMBDA_T_VALUES = np.array([
    0, .001, .002, .005, .01, .02, .03, .05,
    .075, .10, .15, .20, .30, .50, .75, 1.0,
])

SNR_DB, TAU, RHO, SEED = 14.0, 1e-2, .10, 100
OUT = Path("results/lambda_t_sweep")
CACHE = OUT / "cache"
CACHE_VERSION = "v1_warm_desc_ls0135_snr14_tau1e-2_seed100"


def cache_path(lam):
    return CACHE / f"{CACHE_VERSION}_lambda_t_{lam:.6f}.npz"


def save_cache(lam, Y, info):
    np.savez_compressed(
        cache_path(lam), Y=Y,
        converged=np.array(info["converged"]),
        iterations=np.array(info["iterations"]),
        rho=np.array(info["rho"]),
        a=np.array(info["a"]),
    )


def load_cache(lam):
    path = cache_path(lam)
    if not path.exists():
        return None

    z = np.load(path)
    return z["Y"], {
        "converged": bool(z["converged"].item()),
        "iterations": int(z["iterations"].item()),
        "rho": float(z["rho"].item()),
        "a": float(z["a"].item()),
    }


def data_fidelity(Y, S, A):
    r = A.forward(Y) - S
    return float(.5*np.sum(r*r))


def temporal_energy(Y, A, tau):
    """
    Shift raw softmin by its theoretical lower bound:

        R_plus = R + K(T-1) tau log(9) >= 0.

    The additive constant does not change minimizers for fixed lambda_t.
    """
    X = structured_to_softmin(Y)

    raw = softmin(
        X, K=A.K, J=A.J, tau=tau,
        pad_mode="zero", normalize=False,
        return_gradient=False,
    )

    T = Y.shape[-1]
    corrected = raw + A.K*(T-1)*tau*np.log(9.0)

    return float(raw), float(max(corrected, 0.0))


def trajectory_metrics(Y, M):
    traj = dominant_trajectory(Y, M)

    pos = sum(a == b for a, b in zip(
        traj["physical_positions"], TRUE_PHYSICAL_POSITIONS))

    scale = sum(a == b for a, b in zip(
        traj["scale_indices"], TRUE_SCALE_INDICES))

    joint = sum(
        p == pt and s == st
        for p, s, pt, st in zip(
            traj["physical_positions"], traj["scale_indices"],
            TRUE_PHYSICAL_POSITIONS, TRUE_SCALE_INDICES)
    )

    T = len(TRUE_SCALE_INDICES)

    return traj, pos/T, scale/T, joint/T


def normalize_log(values):
    x = np.log10(np.maximum(np.asarray(values, float), 1e-15))
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi == lo else (x-lo)/(hi-lo)


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


def analyze_lcurve(rows):
    # lambda_t=0 is retained as a baseline but excluded from
    # log-coordinate curvature analysis.
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

    finite = [r for r in curve if r["curvature"] is not None]
    selected = max(finite, key=lambda r: r["curvature"]) if finite else None

    return selected, curve


def plot_summary(rows, lcurve_selected, oracle):
    lam = np.array([r["lambda_t"] for r in rows])
    positive = lam > 0

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].semilogx(
        lam[positive],
        [r["coefficient_error"] for r in rows if r["lambda_t"] > 0],
        "o-")
    axes[0, 0].set(
        xlabel="lambda_t", ylabel="Coefficient error",
        title="Oracle coefficient error")

    axes[0, 1].semilogx(
        lam[positive],
        [r["output_snr_db"] for r in rows if r["lambda_t"] > 0],
        "o-")
    axes[0, 1].set(
        xlabel="lambda_t", ylabel="Output SNR (dB)",
        title="Signal recovery")

    axes[1, 0].semilogx(
        lam[positive],
        [100*r["trajectory_accuracy"] for r in rows if r["lambda_t"] > 0],
        "o-")
    axes[1, 0].set(
        xlabel="lambda_t", ylabel="Joint accuracy (%)",
        title="Trajectory accuracy", ylim=(-5, 105))

    axes[1, 1].semilogx(
        lam[positive],
        [r["temporal_energy"] for r in rows if r["lambda_t"] > 0],
        "o-")
    axes[1, 1].set(
        xlabel="lambda_t", ylabel="Corrected temporal energy",
        title="Temporal coherence")

    for ax in axes.flat:
        ax.grid(True)

    for ax in axes.flat:
        ax.axvline(oracle["lambda_t"], color="k", ls="--")

        if lcurve_selected:
            ax.axvline(
                lcurve_selected["lambda_t"],
                color="r", ls=":")

    fig.tight_layout()
    fig.savefig(OUT / "lambda_t_summary.png", dpi=180)
    plt.close(fig)


def plot_lcurve(curve, selected):
    if not curve or selected is None:
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

    ax.plot(
        x[i], y[i], "r*", ms=16,
        label=f"Selected={selected['lambda_t']:.3g}")

    ax.set(
        xlabel="Normalized log data fidelity",
        ylabel="Normalized log temporal energy",
        title="Stage-2 temporal L-curve")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "lambda_t_lcurve.png", dpi=180)
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
    S_noisy, _, input_snr = add_noise_snr(
        S_clean, SNR_DB, rng)

    X_base = A.adjoint(S_noisy, structured=True)
    X_base /= np.max(np.abs(X_base))

    # Descending temporal continuation.
    sweep = np.sort(LAMBDA_T_VALUES)[::-1]
    X_start = X_base.copy()
    previous_lambda = None
    rows = []

    for lam in sweep:
        lam = float(lam)

        print(f"\n{'='*60}")
        print(f"lambda_s={LAMBDA_S:.3f}, lambda_t={lam:.5g}")
        print("="*60)

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

            save_cache(lam, Y, info)

        raw_t, temporal = temporal_energy(Y, A, TAU)
        S_hat = A.forward(Y)
        traj, pos_acc, scale_acc, joint_acc = trajectory_metrics(Y, M)

        row = {
            "lambda_t": lam,
            "warm_start_from_lambda": previous_lambda,
            "converged": bool(info["converged"]),
            "iterations": int(info["iterations"]),
            "final_rho": float(info["rho"]),
            "final_a": float(info["a"]),
            "data_fidelity": data_fidelity(Y, S_noisy, A),
            "softmin_raw": raw_t,
            "temporal_energy": temporal,
            "weighted_temporal": lam*raw_t,
            "coefficient_error": relative_error(Y, X_true),
            "signal_error": relative_error(S_hat, S_clean),
            "output_snr_db": reconstruction_snr_db(S_clean, S_hat),
            "significant_coefficients":
                int(np.count_nonzero(np.abs(Y) > 1e-3)),
            "position_accuracy": pos_acc,
            "scale_accuracy": scale_acc,
            "trajectory_accuracy": joint_acc,
            "trajectory": traj,
        }
        rows.append(row)

        print(
            f"D={row['data_fidelity']:.4e}  "
            f"T={temporal:.4e}  "
            f"Xerr={row['coefficient_error']:.4f}  "
            f"path={100*joint_acc:.0f}%  "
            f"iter={info['iterations']}  "
            f"conv={info['converged']}")

        if info["converged"]:
            X_start = Y.copy()
            previous_lambda = lam
        else:
            print("Unconverged solution not used as warm start.")

    rows.sort(key=lambda r: r["lambda_t"])

    oracle_pool = [r for r in rows if r["converged"]]
    oracle = min(
        oracle_pool,
        key=lambda r: r["coefficient_error"])

    lcurve_selected, curve = analyze_lcurve(rows)

    summary = {
        "input_snr_db": float(input_snr),
        "lambda_s": LAMBDA_S,
        "tau": TAU,
        "oracle_selected_lambda_t": oracle["lambda_t"],
        "oracle_selected_result": oracle,
        "lcurve_selected_lambda_t":
            None if lcurve_selected is None
            else lcurve_selected["lambda_t"],
        "lcurve_selected_result": lcurve_selected,
        "results": rows,
        "curve_results": curve,
    }

    with (OUT / "lambda_t_summary.json").open(
            "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_summary(rows, lcurve_selected, oracle)
    plot_lcurve(curve, lcurve_selected)

    print("\nStage-2 lambda_T selection")
    print("--------------------------")
    print(f"oracle lambda_T: {oracle['lambda_t']:.6g}")
    print(f"oracle X error:  {oracle['coefficient_error']:.6f}")

    if lcurve_selected:
        print(
            f"L-curve lambda_T: "
            f"{lcurve_selected['lambda_t']:.6g}")
        print(
            f"L-curve curvature:"
            f" {lcurve_selected['curvature']:.6f}")
    else:
        print("L-curve selection unavailable.")


if __name__ == "__main__":
    run()
