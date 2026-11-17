import json
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np

from gaussian_experiment import (
    SCALES, TRUE_PHYSICAL_POSITIONS, TRUE_SCALE_INDICES,
    add_noise_snr, dominant_trajectory, make_dictionary,
    make_true_coefficients, reconstruction_snr_db, relative_error,
)
from joint_reanalyze import OUT
from lambda_s_lcurve import CACHE
from log_admm import solve_log_admm
from multiscale_convolution import MultiscaleConvolutionOperator
from softmin import softmin
from x_update import structured_to_softmin


LS_VALUES = np.array([.10, .125, .15, .175, .20, .225, .25])
LT_VALUES = np.array([0, .005, .01, .02, .03, .05, .07, .10, .15])
LOG_DISCREPANCY_TOL = .01
JACCARD_MEAN_MIN = .90
JACCARD_NEIGHBOR_MIN = .80

TAU, RHO, SEED = 1e-2, .10, 100
MAXITER, X_MAXITER = 600, 50
DISC_LO, DISC_HI = .90, 1.10
FIXED_A = 10.

CACHE_VERSION = "v1_joint_ls010_025_lt000_015"


def cache_path(cache, ls, lt):
    return cache / (
        f"{CACHE_VERSION}_ls_{ls:.6f}_lt_{lt:.6f}.npz"
    )


def save_cache(cache, ls, lt, Y, info):
    np.savez_compressed(
        cache_path(cache, ls, lt), Y=Y,
        converged=np.array(info["converged"]),
        iterations=np.array(info["iterations"]),
        rho=np.array(info["rho"]),
        a=np.array(info["a"]))


def load_cache(cache, ls, lt):
    p = cache_path(cache, ls, lt)

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


def fixed_log(Y):
    return float(np.sum(np.log1p(FIXED_A*np.abs(Y))))


def temporal_energy(Y, A):
    X = structured_to_softmin(Y)
    raw = softmin(
        X, K=A.K, J=A.J, tau=TAU,
        pad_mode="zero", normalize=False,
        return_gradient=False)

    c = A.K*(Y.shape[-1]-1)*TAU*np.log(9.)
    return float(raw), float(max(raw+c, 0.))


def trajectory_accuracy(Y, M):
    tr = dominant_trajectory(Y, M)

    joint = sum(
        p == pt and s == st
        for p, s, pt, st in zip(
            tr["physical_positions"], tr["scale_indices"],
            TRUE_PHYSICAL_POSITIONS, TRUE_SCALE_INDICES))

    return tr, joint/len(TRUE_SCALE_INDICES)


def pareto_filter(rows):
    out = []

    for i, a in enumerate(rows):
        dominated = False

        for j, b in enumerate(rows):
            if i == j:
                continue

            better = (
                b["sparsity"] <= a["sparsity"]
                and b["temporal"] <= a["temporal"]
            )

            strict = (
                b["sparsity"] < a["sparsity"]
                or b["temporal"] < a["temporal"]
            )

            if better and strict:
                dominated = True
                break

        if not dominated:
            out.append(a)

    return out


def normalized_ideal_choice(rows):
    s = np.array([r["sparsity"] for r in rows])
    t = np.array([r["temporal"] for r in rows])

    def norm(x):
        lo, hi = x.min(), x.max()
        return np.zeros_like(x) if hi == lo else (x-lo)/(hi-lo)

    sn, tn = norm(s), norm(t)
    dist = np.sqrt(sn*sn + tn*tn)

    for r, a, b, d in zip(rows, sn, tn, dist):
        r["normalized_sparsity"] = float(a)
        r["normalized_temporal"] = float(b)
        r["ideal_distance"] = float(d)

    return rows[int(np.argmin(dist))]


def neighbor_jaccard(Y1, Y2, threshold=1e-3):
    a = np.abs(Y1) > threshold
    b = np.abs(Y2) > threshold
    union = np.count_nonzero(a | b)

    if union == 0:
        return 1.

    return float(np.count_nonzero(a & b)/union)

def add_grid_jaccard(rows, ls_values, lt_values):
    ls = [float(x) for x in ls_values]
    lt = [float(x) for x in lt_values]

    lookup = {
        (round(r["lambda_s"], 10), round(r["lambda_t"], 10)): r
        for r in rows
    }

    for r in rows:
        i = int(np.argmin(np.abs(np.asarray(ls) - r["lambda_s"])))
        j = int(np.argmin(np.abs(np.asarray(lt) - r["lambda_t"])))

        neighbors = []

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ii, jj = i + di, j + dj

            if not (0 <= ii < len(ls) and 0 <= jj < len(lt)):
                continue

            q = lookup.get((round(ls[ii], 10), round(lt[jj], 10)))

            if q is not None and q["converged"]:
                neighbors.append(q)

        if not neighbors:
            r["neighbor_jaccard"] = None
            r["neighbor_jaccard_min"] = None
            r["neighbor_count"] = 0
            continue

        scores = [
            neighbor_jaccard(r["_Y"], q["_Y"])
            for q in neighbors
        ]

        r["neighbor_jaccard"] = float(np.mean(scores))
        r["neighbor_jaccard_min"] = float(np.min(scores))
        r["neighbor_count"] = len(scores)

def select_known_noise(stable):
    for r in stable:
        r["log_discrepancy"] = abs(
            np.log(r["discrepancy_ratio"])
        )

    best = min(r["log_discrepancy"] for r in stable)

    noise_matched = [
        r for r in stable
        if r["log_discrepancy"]
        <= best + LOG_DISCREPANCY_TOL
    ]

    min_active = min(r["active"] for r in noise_matched)

    sparse = [
        r for r in noise_matched
        if r["active"] == min_active
    ]

    selected = min(
        sparse,
        key=lambda r: r["temporal"]
    )

    return selected, noise_matched, sparse

def run(snr_db):
    OUT = Path(f"results/joint_grid_snr{snr_db:g}")
    CACHE = OUT / "cache"

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    M, N, T = 41, 96, 7

    D = make_dictionary(M, 4.)
    A = MultiscaleConvolutionOperator(
        D=D, scales=SCALES, center=(M+1)//2,
        N=N, normalize=True)

    Xtrue, _ = make_true_coefficients(
        L=A.L, M=M, J=len(SCALES), T=T)

    Sclean = A.forward(Xtrue)
    Snoisy, noise, achieved_snr = add_noise_snr(
        Sclean, snr_db, rng)

    target = float(.5*np.sum(noise*noise))

    Xbase = A.adjoint(Snoisy, structured=True)
    Xbase /= np.max(np.abs(Xbase))

    rows = []
    previous_row_start = Xbase.copy()

    # Serpentine traversal: neighboring solves remain nearby.
    for row_idx, ls in enumerate(LS_VALUES[::-1]):
        ls = float(ls)

        lt_order = (
            LT_VALUES[::-1]
            if row_idx % 2 == 0
            else LT_VALUES
        )

        Xstart = previous_row_start.copy()
        first_converged = None

        for lt in lt_order:
            lt = float(lt)

            print(f"\nls={ls:.4f}, lt={lt:.4f}")

            cached = load_cache(CACHE,ls, lt)

            if cached:
                Y, info = cached
                print(
                    f"cache: iter={info['iterations']} "
                    f"conv={info['converged']}")
            else:
                _, Y, _, info = solve_log_admm(
                    X0=Xstart.copy(), S=Snoisy, operator=A,
                    lambda_s=ls, lambda2=lt,
                    tau=TAU, rho=RHO,
                    max_admm_iter=MAXITER, x_maxiter=X_MAXITER,
                    abs_tol=1e-4, rel_tol=1e-3,
                    adaptive_rho=True, rho_mu=10, rho_tau=2,
                    a_min=1e-4, continuation_iters=100,
                    a_fraction=.95, verbose=False)

                save_cache(CACHE, ls, lt, Y, info)

            raw_t, temp = temporal_energy(Y, A)
            Shat = A.forward(Y)
            tr, traj_acc = trajectory_accuracy(Y, M)

            row = {
                "lambda_s": ls,
                "lambda_t": lt,
                "converged": bool(info["converged"]),
                "iterations": int(info["iterations"]),
                "rho": float(info["rho"]),
                "a": float(info["a"]),
                "data": data_fidelity(Y, Snoisy, A),
                "discrepancy_ratio":
                    data_fidelity(Y, Snoisy, A)/target,
                "sparsity": fixed_log(Y),
                "softmin_raw": raw_t,
                "temporal": temp,
                "active":
                    int(np.count_nonzero(np.abs(Y) > 1e-3)),
                "coefficient_error": relative_error(Y, Xtrue),
                "signal_error": relative_error(Shat, Sclean),
                "output_snr_db":
                    reconstruction_snr_db(Sclean, Shat),
                "trajectory_accuracy": traj_acc,
                "trajectory": tr,
                "_Y": Y,
            }

            rows.append(row)

            print(
                f"D/Dn={row['discrepancy_ratio']:.3f} "
                f"Rs={row['sparsity']:.3f} "
                f"Rt={row['temporal']:.3f} "
                f"nz={row['active']} "
                f"Xerr={row['coefficient_error']:.4f} "
                f"conv={row['converged']}")

            if info["converged"]:
                Xstart = Y.copy()
                if first_converged is None:
                    first_converged = Y.copy()

        if first_converged is not None:
            previous_row_start = first_converged

    # Explicit four-neighbor grid stability.
    add_grid_jaccard(rows, LS_VALUES, LT_VALUES)
    
    # -------------------------------------------------
    # Converged solutions
    # -------------------------------------------------
    converged = [r for r in rows if r["converged"]]

        # -------------------------------------------------
    # Stable feasible solutions
    # -------------------------------------------------
    stable = [
        r for r in converged
        if (
            r["neighbor_count"] >= 2
            and r["neighbor_jaccard"] is not None
            and r["neighbor_jaccard"] >= JACCARD_MEAN_MIN
            and r["neighbor_jaccard_min"] is not None
            and r["neighbor_jaccard_min"] >= JACCARD_NEIGHBOR_MIN
        )
    ]

    if not stable:
        raise RuntimeError("No stable converged grid points.")

    selected, noise_matched, sparse = select_known_noise(stable)

    # -------------------------------------------------
    # Synthetic oracle, diagnostic only
    # -------------------------------------------------

    oracle = min(
        converged,
        key=lambda r: r["coefficient_error"])

    print("\nJoint selection")
    print("---------------")
    print(
        f"selected: ls={selected['lambda_s']:.4f}, "
        f"lt={selected['lambda_t']:.4f}")
    print(
        f"discrepancy ratio={selected['discrepancy_ratio']:.3f}, "
        f"Jmean={selected['neighbor_jaccard']:.3f}, "
        f"Jmin={selected['neighbor_jaccard_min']:.3f}, "
        f"neighbors={selected['neighbor_count']}"
    )
    
    print(
        f"oracle:   ls={oracle['lambda_s']:.4f}, "
        f"lt={oracle['lambda_t']:.4f}")
    print(
        f"selected Xerr={selected['coefficient_error']:.5f}, "
        f"oracle Xerr={oracle['coefficient_error']:.5f}")

    summary = {
        "snr_db": float(achieved_snr),
        "discrepancy_target": target,
        "discrepancy_band": [DISC_LO, DISC_HI],
        "lambda_s_grid": [float(x) for x in LS_VALUES],
        "lambda_t_grid": [float(x) for x in LT_VALUES],
        "selected": {
            k: v for k, v in selected.items()
            if k != "_Y"
        },
        "oracle": {
            k: v for k, v in oracle.items()
            if k != "_Y"
        },
        "results": [
            {k: v for k, v in r.items() if k != "_Y"}
            for r in rows
        ],
        "selection_rule": {
            "type": "known_noise_joint",
            "log_discrepancy_tolerance": LOG_DISCREPANCY_TOL,
            "jaccard_mean_min": JACCARD_MEAN_MIN,
            "jaccard_neighbor_min": JACCARD_NEIGHBOR_MIN,
            "priority": [
                "converged",
                "grid-neighbor support stability",
                "noise discrepancy",
                "minimum active support",
                "minimum temporal energy",
            ],
        },
        "requested_snr_db": float(snr_db),
        "achieved_snr_db": float(achieved_snr),
    }

    with (OUT / "joint_summary.json").open(
            "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_heatmaps(rows, selected, oracle)


def plot_heatmaps(rows, selected, oracle):
    metrics = [
        ("coefficient_error", "Coefficient error"),
        ("discrepancy_ratio", "Discrepancy ratio"),
        ("active", "Active coefficients"),
        ("trajectory_accuracy", "Trajectory accuracy"),
    ]

    ls = np.array(LS_VALUES)
    lt = np.array(LT_VALUES)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for ax, (key, title) in zip(axes.flat, metrics):
        Z = np.full((len(ls), len(lt)), np.nan)

        for r in rows:
            i = int(np.argmin(np.abs(ls-r["lambda_s"])))
            j = int(np.argmin(np.abs(lt-r["lambda_t"])))
            Z[i, j] = r[key]

        im = ax.imshow(
            Z, origin="lower", aspect="auto",
            extent=[lt.min(), lt.max(), ls.min(), ls.max()])

        ax.scatter(
            selected["lambda_t"], selected["lambda_s"],
            marker="*", s=180, c="red", label="Selected")

        ax.scatter(
            oracle["lambda_t"], oracle["lambda_s"],
            marker="x", s=100, c="white", label="Oracle")

        ax.set(
            xlabel="lambda_t",
            ylabel="lambda_s",
            title=title)

        fig.colorbar(im, ax=ax)

    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "joint_heatmaps.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr-db", type=float, default=14.0)
    args = parser.parse_args()
    run(args.snr_db)

