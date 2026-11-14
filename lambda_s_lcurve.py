import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gaussian_experiment import (
    SCALES, add_noise_snr, make_dictionary,
    make_true_coefficients, relative_error,
    reconstruction_snr_db,
)
from log_admm import solve_log_admm
from multiscale_convolution import MultiscaleConvolutionOperator


LAMBDAS = np.array([
    0.005, 0.0075, 0.010, 0.015, 0.020, 0.030,
    0.040, 0.050, 0.060, 0.075, 0.090, 0.105,
    0.120, 0.135, 0.150, 0.175, 0.200,
])

SNR_DB = 14.0
LAMBDA_T = 0.0
TAU = 1e-2
RHO = 0.10
SEED = 100
OUT = Path("results/lambda_s_lcurve")

CACHE = OUT / "cache"
CACHE_VERSION = "v2_warm_desc_snr14_lt0_tau1e-2_seed100"

def cache_path(lam):
    return CACHE / (
        f"{CACHE_VERSION}_lambda_{lam:.6f}.npz"
    )


def save_cache(lam, Y, info):
    np.savez_compressed(
        cache_path(lam),
        Y=Y,
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

    Y = z["Y"]

    info = {
        "converged": bool(z["converged"].item()),
        "iterations": int(z["iterations"].item()),
        "rho": float(z["rho"].item()),
        "a": float(z["a"].item()),
    }

    return Y, info

def penalty_unweighted(X, a):
    return float(np.sum(np.log1p(a*np.abs(X))) / a)


def data_fidelity(X, S, A):
    R = A.forward(X) - S
    return float(0.5*np.sum(R**2))


def moving_average_3(x):
    y = np.array(x, dtype=float, copy=True)
    if len(y) >= 3:
        y[1:-1] = (x[:-2] + x[1:-1] + x[2:]) / 3.0
    return y


def normalized_log_coordinates(rows):
    x = np.log10([r["data_fidelity"] for r in rows])
    y = np.log10([r["penalty_unweighted"] for r in rows])

    def norm(z):
        zmin, zmax = np.min(z), np.max(z)
        return np.zeros_like(z) if zmax == zmin else (z-zmin)/(zmax-zmin)

    return norm(x), norm(y)


def discrete_curvature(x, y):
    x, y = np.asarray(x), np.asarray(y)
    kappa = np.full(len(x), np.nan)

    for i in range(1, len(x)-1):
        xp = 0.5*(x[i+1] - x[i-1])
        yp = 0.5*(y[i+1] - y[i-1])
        xpp = x[i+1] - 2*x[i] + x[i-1]
        ypp = y[i+1] - 2*y[i] + y[i-1]
        denom = (xp*xp + yp*yp)**1.5
        if denom > 0:
            kappa[i] = abs(xp*ypp - yp*xpp) / denom

    return kappa


def plot_lcurve(rows, selected):
    x = np.array([r["lcurve_x_smoothed"] for r in rows])
    y = np.array([r["lcurve_y_smoothed"] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x, y, "o-")

    for xi, yi, r in zip(x, y, rows):
        ax.annotate(
            f"{r['lambda_s']:.3g}", (xi, yi),
            xytext=(4, 4), textcoords="offset points", fontsize=8)

    i = next(i for i, r in enumerate(rows)
             if r["lambda_s"] == selected["lambda_s"])

    ax.plot(
        x[i], y[i], "r*", markersize=16,
        label=f"Selected lambda_s={selected['lambda_s']:.3g}")

    ax.set(
        xlabel="Normalized log data fidelity",
        ylabel="Normalized log sparsity penalty",
        title="Stage-1 log-penalty L-curve",
    )
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "lcurve.png", dpi=180)
    plt.close(fig)


def plot_curvature(rows, selected):
    lam = np.array([r["lambda_s"] for r in rows])
    k = np.array([
        np.nan if r["curvature"] is None else r["curvature"]
        for r in rows
    ])

    fig, ax = plt.subplots()
    ax.semilogx(lam, k, "o-")
    ax.axvline(selected["lambda_s"], color="r", linestyle="--")
    ax.set(
        xlabel="lambda_s",
        ylabel="Discrete curvature",
        title="L-curve curvature",
    )
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUT / "lcurve_curvature.png", dpi=180)
    plt.close(fig)


def plot_posthoc(rows, selected):
    rows = [r for r in rows if r["converged"]]
    lam = np.array([r["lambda_s"] for r in rows])
    xe = np.array([r["coefficient_error"] for r in rows])
    snr = np.array([r["output_snr_db"] for r in rows])

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].semilogx(lam, xe, "o-")
    axes[0].axvline(selected["lambda_s"], color="r", linestyle="--")
    axes[0].set_ylabel("Oracle coefficient error")
    axes[0].grid(True)

    axes[1].semilogx(lam, snr, "o-")
    axes[1].axvline(selected["lambda_s"], color="r", linestyle="--")
    axes[1].set(xlabel="lambda_s", ylabel="Output SNR (dB)")
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(OUT / "posthoc_validation.png", dpi=180)
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

    X0 = A.adjoint(S_noisy, structured=True)
    X0 /= np.max(np.abs(X0))

    rows = []

    SWEEP_LAMBDAS = np.sort(LAMBDAS)[::-1]
    X_start = X0.copy()
    previous_lambda = None

    for lam in SWEEP_LAMBDAS:
        lam = float(lam)

        print(f"\n{'='*58}")
        print(f"lambda_s={lam:.5g}, lambda_t=0")
        print("="*58)

        cached = load_cache(lam)
        if cached is not None:
            Y, info = cached
            print(
                f"Loaded cached solution: "
                f"iter={info['iterations']}, "
                f"conv={info['converged']}"
            ) 
        else:    
            _, Y, _, info = solve_log_admm(
                X0=X_start.copy(),
                S=S_noisy,
                operator=A,
                lambda_s=lam,
                lambda2=LAMBDA_T,
                tau=TAU,
                rho=RHO,
                max_admm_iter=600,
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

            save_cache(lam, Y, info)

        # Continue from this solution only if it converged.
        if info["converged"]:
            X_start = Y.copy()
            previous_lambda = lam

        else:
            print("Warning: solution did not converge; "
                    "retaining previous warm start.")

        a = float(info["a"])
        d = data_fidelity(Y, S_noisy, A)
        penalty = penalty_unweighted(Y, a)

        p_fixed_1 = float(np.sum(np.log1p(1.0 * np.abs(Y))))
        p_fixed_10 = float(np.sum(np.log1p(10.0 * np.abs(Y))))
        l1_norm = float(np.sum(np.abs(Y)))
        active_count = int(np.count_nonzero(np.abs(Y) > 1e-3))

        Xerr = relative_error(Y, X_true)
        Shat = A.forward(Y)
        Serr = relative_error(Shat, S_clean)
        out_snr = reconstruction_snr_db(S_clean, Shat)

        row = {
            "lambda_s": lam,
            "converged": bool(info["converged"]),
            "iterations": int(info["iterations"]),
            "final_rho": float(info["rho"]),
            "final_a": a,
            "data_fidelity": d,
            "penalty_unweighted": penalty,
            "weighted_penalty": lam * penalty,
            "coefficient_error": float(Xerr),
            "signal_error": float(Serr),
            "output_snr_db": float(out_snr),
            "penalty_fixed_a1": p_fixed_1,
            "penalty_fixed_a10": p_fixed_10,
            "l1_norm": l1_norm,
            "active_count": active_count,
            "significant_coefficients": active_count,
            "warm_start": True,
            "warm_start_from_lambda": previous_lambda,
            "sweep_direction": "descending_lambda"
        }
        rows.append(row)

        print(
            f"D={d:.5e}  P={penalty:.5e}  "
            f"Xerr={Xerr:.5f}  "
            f"nz={row['significant_coefficients']:3d}  "
            f"iter={info['iterations']:3d}  "
            f"conv={info['converged']}"
        )

    # Use all valid converged solutions.
    curve = [
        r for r in rows
        if (
            r["converged"]
            and np.isfinite(r["data_fidelity"])
            and np.isfinite(r["penalty_unweighted"])
            and r["data_fidelity"] > 0
            and r["penalty_unweighted"] > 0
        )
    ]
    curve.sort(key=lambda r: r["lambda_s"])

    if len(curve) < 3:
        raise RuntimeError(
            "Need at least three valid converged points "
            "for L-curve curvature estimation."
        )

    print("\nConverged L-curve points")
    print("------------------------")
    print("lambda_s       data fidelity       penalty")

    for r in curve:
        print(
            f"{r['lambda_s']:8.4f}    "
            f"{r['data_fidelity']:14.6e}    "
            f"{r['penalty_unweighted']:14.6e}"
        )

    # Log coordinates, normalization and smoothing.
    x, y = normalized_log_coordinates(curve)
    xs = moving_average_3(x)
    ys = moving_average_3(y)

    curvature = discrete_curvature(xs, ys)
    finite = np.where(np.isfinite(curvature))[0]

    if len(finite) == 0:
        raise RuntimeError("No finite L-curve curvature values.")

    selected_index = finite[np.argmax(curvature[finite])]
    selected = curve[selected_index]

    for i, r in enumerate(curve):
        r["lcurve_x"] = float(x[i])
        r["lcurve_y"] = float(y[i])
        r["lcurve_x_smoothed"] = float(xs[i])
        r["lcurve_y_smoothed"] = float(ys[i])
        r["curvature"] = (
            None if not np.isfinite(curvature[i])
            else float(curvature[i])
        )

    summary = {
        "input_snr_db": float(input_snr),
        "lambda_t": LAMBDA_T,
        "tau": TAU,
        "selection_method": (
            "Valid converged solutions; normalized log L-curve; "
            "length-3 moving-average smoothing; "
            "maximum discrete curvature"
        ),
        "selected_lambda_s": float(selected["lambda_s"]),
        "selected_result": selected,
        "all_results": rows,
        "curve_results": curve,
    }

    with (OUT / "lcurve_summary.json").open(
            "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_lcurve(curve, selected)
    plot_curvature(curve, selected)
    plot_posthoc(rows, selected)

    print("\nL-curve selection")
    print("-----------------")
    print(f"selected lambda_s: {selected['lambda_s']:.6g}")
    print(f"data fidelity:      {selected['data_fidelity']:.6e}")
    print(f"unweighted penalty: {selected['penalty_unweighted']:.6e}")
    print(f"curvature:          {selected['curvature']:.6e}")
    print(f"oracle X error:     {selected['coefficient_error']:.6f}")


if __name__ == "__main__":
    run()
