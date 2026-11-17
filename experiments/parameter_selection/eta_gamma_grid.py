import argparse
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


ETA_VALUES = np.array([
    .075, .10, .125, .15, .175, .20, .225, .25, .30
])

GAMMA_VALUES = np.concatenate((
    [0.0],
    np.arange(.30, 1.0001, .05),
))
TAU, RHO, SEED = 1e-2, .10, 100
MAXITER, X_MAXITER = 600, 50
ACTIVE_THRESHOLD = 1e-3
CACHE_VERSION = "v1_eta_gamma_dense_030_100"

DELTA_X_MAX = .10
JACCARD_MIN = .90
JACCARD_NEIGHBOR_MIN = .80


def cache_path(cache, eta, gamma):
    return cache / (
        f"{CACHE_VERSION}_eta_{eta:.6f}_gamma_{gamma:.6f}.npz"
    )


def save_cache(cache, eta, gamma, Y, info):
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path(cache, eta, gamma),
        Y=Y,
        converged=np.array(info["converged"]),
        iterations=np.array(info["iterations"]),
        rho=np.array(info["rho"]),
        a=np.array(info["a"]),
    )


def load_cache(cache, eta, gamma):
    path = cache_path(cache, eta, gamma)
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


def fixed_log(Y, a=10.0):
    return float(np.sum(np.log1p(a*np.abs(Y))))


def temporal_energy(Y, A):
    X = structured_to_softmin(Y)

    raw = softmin(
        X, K=A.K, J=A.J, tau=TAU,
        pad_mode="zero", normalize=False,
        return_gradient=False,
    )

    correction = A.K*(Y.shape[-1]-1)*TAU*np.log(9.0)
    return float(raw), float(max(raw + correction, 0.0))


def support_jaccard(A, B):
    a = np.abs(A) > ACTIVE_THRESHOLD
    b = np.abs(B) > ACTIVE_THRESHOLD
    union = np.count_nonzero(a | b)

    return (
        1.0 if union == 0
        else float(np.count_nonzero(a & b)/union)
    )


def relative_change(A, B):
    return float(
        np.linalg.norm(A-B)
        / max(np.linalg.norm(B), 1e-12)
    )


def trajectory_metrics(Y, M):
    tr = dominant_trajectory(Y, M)

    pos = sum(a == b for a, b in zip(
        tr["physical_positions"], TRUE_PHYSICAL_POSITIONS
    ))

    scale = sum(a == b for a, b in zip(
        tr["scale_indices"], TRUE_SCALE_INDICES
    ))

    joint = sum(
        p == pt and s == st
        for p, s, pt, st in zip(
            tr["physical_positions"], tr["scale_indices"],
            TRUE_PHYSICAL_POSITIONS, TRUE_SCALE_INDICES,
        )
    )

    T = len(TRUE_SCALE_INDICES)

    return tr, pos/T, scale/T, joint/T


def solve_grid(S, A, X0, target, cache):
    rows = []
    previous_gamma_start = X0.copy()

    for gi, gamma in enumerate(GAMMA_VALUES):
        gamma = float(gamma)

        # Descending eta continuation inside each gamma row.
        Xstart = previous_gamma_start.copy()
        row_first = None

        print(f"\n{'='*64}")
        print(f"gamma={gamma:.3f}")
        print("="*64)

        for eta in np.sort(ETA_VALUES)[::-1]:
            eta = float(eta)
            ls, lt = eta, gamma*eta

            cached = load_cache(cache, eta, gamma)

            if cached:
                Y, info = cached
                print(
                    f"eta={eta:.3f} cache "
                    f"iter={info['iterations']} conv={info['converged']}"
                )
            else:
                _, Y, _, info = solve_log_admm(
                    X0=Xstart.copy(), S=S, operator=A,
                    lambda_s=ls, lambda2=lt,
                    tau=TAU, rho=RHO,
                    max_admm_iter=MAXITER, x_maxiter=X_MAXITER,
                    abs_tol=1e-4, rel_tol=1e-3,
                    adaptive_rho=True, rho_mu=10, rho_tau=2,
                    a_min=1e-4, continuation_iters=100,
                    a_fraction=.95, verbose=False,
                )
                save_cache(cache, eta, gamma, Y, info)

            raw, temporal = temporal_energy(Y, A)
            data = data_fidelity(Y, S, A)

            rows.append({
                "eta": eta,
                "gamma": gamma,
                "lambda_s": ls,
                "lambda_t": lt,
                "converged": bool(info["converged"]),
                "iterations": int(info["iterations"]),
                "rho": float(info["rho"]),
                "a": float(info["a"]),
                "data": data,
                "discrepancy_ratio": data/target,
                "log_discrepancy": abs(np.log(data/target)),
                "sparsity": fixed_log(Y),
                "softmin_raw": raw,
                "temporal": temporal,
                "active": int(
                    np.count_nonzero(np.abs(Y) > ACTIVE_THRESHOLD)
                ),
                "_Y": Y,
            })

            r = rows[-1]

            print(
                f"eta={eta:.3f} ls={ls:.3f} lt={lt:.4f} "
                f"D/Dn={r['discrepancy_ratio']:.3f} "
                f"nz={r['active']:2d} conv={r['converged']}"
            )

            if info["converged"]:
                Xstart = Y.copy()
                if row_first is None:
                    row_first = Y.copy()

        if row_first is not None:
            previous_gamma_start = row_first

    return rows


def select_eta_by_gamma(rows):
    selected = []

    for gamma in GAMMA_VALUES:
        candidates = [
            r for r in rows
            if r["converged"]
            and np.isclose(r["gamma"], gamma)
        ]

        if not candidates:
            continue

        best = min(candidates, key=lambda r: r["log_discrepancy"])
        best["selected_for_gamma"] = True
        selected.append(best)

    selected.sort(key=lambda r: r["gamma"])
    return selected


def add_gamma_stability(selected):
    for i, r in enumerate(selected):
        neighbors = []

        if i > 0:
            neighbors.append(selected[i-1])

        if i < len(selected)-1:
            neighbors.append(selected[i+1])

        if not neighbors:
            r["gamma_jaccard"] = None
            r["gamma_delta_x"] = None
            continue

        jaccard = [
            support_jaccard(r["_Y"], q["_Y"])
            for q in neighbors
        ]

        delta = [
            relative_change(r["_Y"], q["_Y"])
            for q in neighbors
        ]

        r["gamma_jaccard"] = float(np.mean(jaccard))
        r["gamma_jaccard_min"] = float(np.min(jaccard))
        r["gamma_delta_x"] = float(np.mean(delta))
        r["gamma_delta_x_max"] = float(np.max(delta))


def choose_gamma(selected):
    stable = [
        r for r in selected
        if (
            r["gamma"] > 0
            and r["gamma_jaccard"] is not None
            and r["gamma_jaccard"] >= JACCARD_MIN
            and r["gamma_jaccard_min"] is not None
            and r["gamma_jaccard_min"] >= JACCARD_NEIGHBOR_MIN
            and r["gamma_delta_x"] is not None
            and r["gamma_delta_x"] <= DELTA_X_MAX
            and r["gamma_delta_x_max"] is not None
            and r["gamma_delta_x_max"] <= DELTA_X_MAX
        )
    ]

    if not stable:
        raise RuntimeError(
            "No stable positive-gamma noise-matched solutions."
        )

    stable_lookup = {
        round(r["gamma"], 10): r
        for r in stable
    }

    positive_gamma = [
        float(g)
        for g in GAMMA_VALUES
        if g > 0
    ]

    runs = []
    current = []

    for gamma in positive_gamma:
        key = round(gamma, 10)

        if key in stable_lookup:
            current.append(stable_lookup[key])
        else:
            if current:
                runs.append(current)
                current = []

    if current:
        runs.append(current)

    if not runs:
        raise RuntimeError(
            "No contiguous stable positive-gamma run."
        )

    # Prefer the longest stable plateau.
    # Break equal-length ties by the best average noise match.
    runs.sort(
        key=lambda run: (
            -len(run),
            np.mean([
                r["log_discrepancy"]
                for r in run
            ]),
        )
    )

    plateau = runs[0]

    # Choose the center of the plateau.
    chosen = plateau[len(plateau) // 2]

    return chosen, plateau, stable


def add_truth_metrics(rows, Xtrue, Sclean, A, M):
    for r in rows:
        Y = r["_Y"]
        Shat = A.forward(Y)
        tr, pa, sa, ja = trajectory_metrics(Y, M)

        r["coefficient_error"] = relative_error(Y, Xtrue)
        r["signal_error"] = relative_error(Shat, Sclean)
        r["output_snr_db"] = reconstruction_snr_db(Sclean, Shat)
        r["position_accuracy"] = pa
        r["scale_accuracy"] = sa
        r["trajectory_accuracy"] = ja
        r["trajectory"] = tr


def clean(row):
    return {k: v for k, v in row.items() if k != "_Y"}


def plot_gamma_path(selected, chosen, oracle, out):
    gamma = np.array([r["gamma"] for r in selected])
    eta = np.array([r["eta"] for r in selected])
    disc = np.array([r["discrepancy_ratio"] for r in selected])
    jaccard = np.array([
        np.nan if r["gamma_jaccard"] is None
        else r["gamma_jaccard"]
        for r in selected
    ])
    delta = np.array([
        np.nan if r["gamma_delta_x"] is None
        else r["gamma_delta_x"]
        for r in selected
    ])
    xerr = np.array([r["coefficient_error"] for r in selected])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(gamma, eta, "o-")
    axes[0, 0].set(
        xlabel="gamma", ylabel="eta*(gamma)",
        title="Noise-matched regularization strength",
    )

    axes[0, 1].plot(gamma, disc, "o-")
    axes[0, 1].axhline(1.0, color="k", ls="--")
    axes[0, 1].set(
        xlabel="gamma", ylabel="D/Dnoise",
        title="Discrepancy match",
    )

    axes[1, 0].plot(gamma, jaccard, "o-", label="Jaccard")
    axes[1, 0].plot(gamma, delta, "s-", label="Delta X")
    axes[1, 0].axhline(JACCARD_MIN, color="k", ls="--")
    axes[1, 0].axhline(DELTA_X_MAX, color="gray", ls=":")
    axes[1, 0].set(
        xlabel="gamma", ylabel="Stability",
        title="Gamma-path stability",
    )
    axes[1, 0].legend()

    axes[1, 1].plot(gamma, xerr, "o-")
    axes[1, 1].set(
        xlabel="gamma", ylabel="Coefficient error",
        title="Post-hoc oracle validation",
    )

    for ax in axes.flat:
        ax.grid(True)
        ax.axvline(chosen["gamma"], color="r", ls="--")

    axes[1, 1].axvline(oracle["gamma"], color="k", ls=":")

    fig.tight_layout()
    fig.savefig(out / "eta_gamma_path.png", dpi=180)
    plt.close(fig)


def run(snr_db):
    out = Path(f"results/eta_gamma_snr{snr_db:g}")
    cache = out / "cache"

    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    M, N, T = 41, 96, 7

    D = make_dictionary(M, 4.0)

    A = MultiscaleConvolutionOperator(
        D=D, scales=SCALES, center=(M+1)//2,
        N=N, normalize=True,
    )

    Xtrue, _ = make_true_coefficients(
        L=A.L, M=M, J=len(SCALES), T=T,
    )

    Sclean = A.forward(Xtrue)
    Snoisy, noise, achieved_snr = add_noise_snr(
        Sclean, snr_db, rng,
    )

    target = float(.5*np.sum(noise*noise))

    X0 = A.adjoint(Snoisy, structured=True)
    X0 /= np.max(np.abs(X0))

    rows = solve_grid(Snoisy, A, X0, target, cache)

    selected_by_gamma = select_eta_by_gamma(rows)
    add_gamma_stability(selected_by_gamma)

    chosen, plateau, stable = choose_gamma(selected_by_gamma)

    # Synthetic truth is used only after selection.
    add_truth_metrics(rows, Xtrue, Sclean, A, M)

    oracle = min(
        [r for r in rows if r["converged"]],
        key=lambda r: r["coefficient_error"],
    )

    summary = {
        "requested_snr_db": float(snr_db),
        "achieved_snr_db": float(achieved_snr),
        "discrepancy_target": target,
        "eta_grid": [float(x) for x in ETA_VALUES],
        "gamma_grid": [float(x) for x in GAMMA_VALUES],
        "selection_rule": {
            "type": "noise_matched_eta_stable_gamma_plateau",
            "jaccard_mean_min": JACCARD_MIN,
            "jaccard_neighbor_min": JACCARD_NEIGHBOR_MIN,
            "delta_x_mean_max": DELTA_X_MAX,
            "delta_x_neighbor_max": DELTA_X_MAX,
            "plateau_choice": "center of longest stable positive-gamma run",    
        },
        "selected": clean(chosen),
        "stable_plateau": [clean(r) for r in plateau],
        "stable_gamma_solutions": [clean(r) for r in stable],
        "selected_by_gamma": [
            clean(r) for r in selected_by_gamma
        ],
        "oracle": clean(oracle),
        "results": [clean(r) for r in rows],
    }

    with (out / "eta_gamma_summary.json").open(
        "w", encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    plot_gamma_path(
        selected_by_gamma, chosen, oracle, out,
    )

    print()
    print("Eta-gamma selection")
    print("-------------------")
    print(f"selected gamma:    {chosen['gamma']:.3f}")
    print(f"selected eta:      {chosen['eta']:.3f}")
    print(f"selected lambda_s: {chosen['lambda_s']:.3f}")
    print(f"selected lambda_t: {chosen['lambda_t']:.4f}")
    print(f"discrepancy ratio: {chosen['discrepancy_ratio']:.4f}")
    print(
        f"gamma Jaccard:     "
        f"{chosen['gamma_jaccard']:.3f}"
    )
    print(
        f"gamma Delta X:     "
        f"{chosen['gamma_delta_x']:.4f}"
    )
    print(f"selected X error:  {chosen['coefficient_error']:.5f}")
    print()
    print(
        f"oracle eta/gamma:  "
        f"{oracle['eta']:.3f}/{oracle['gamma']:.3f}"
    )
    print(
        f"oracle lambdas:    "
        f"{oracle['lambda_s']:.3f}/"
        f"{oracle['lambda_t']:.4f}"
    )
    print(f"oracle X error:    {oracle['coefficient_error']:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr-db", type=float, default=14.0)
    args = parser.parse_args()
    run(args.snr_db)
    