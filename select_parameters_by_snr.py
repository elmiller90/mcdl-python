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

CACHE_VERSION = "v3_robust_corner_tau1e-2_seed100"
LS_VALUES = np.array([
    .02, .03, .04, .05, .075, .10, .125, .15, .175,
    .20, .25, .30, .35, .40, .50, .60, .70, .80, .90
])
LT_VALUES = np.array([
    0,
    .005, .01, .02, .03, .04, .05, .06, .07, .08,
    .10, .15, .20, .25, .30, .35, .40, .45, .50
])

STAGE1_SMOOTH_WINDOW = 5
CURVATURE_RADIUS = 1
MIN_LOCAL_DATA_SPAN = .01

TAU, RHO, SEED = 1e-2, .10, 100
MAXITER, X_MAXITER = 600, 50
FIXED_A = 10.0

# Require a reasonably oriented path before trusting curvature.
MIN_RANK_CORR = .5
MIN_COMPLEXITY_CORR = .5
MIN_DATA_CORR = 0.0
MIN_DATA_SPAN = .02

def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")

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

    # Shift by softmin lower bound so the plotted measure is nonnegative.
    c = A.K*(Y.shape[-1]-1)*TAU*np.log(9.0)
    return float(raw), float(max(raw + c, 0.0))


def cache_path(root, stage, lam, lambda_s=None):
    if stage == "s":
        name = f"{CACHE_VERSION}_s_{lam:.6f}.npz"
    else:
        name = (
            f"{CACHE_VERSION}_t_"
            f"ls_{lambda_s:.6f}_"
            f"lt_{lam:.6f}.npz"
        )

    return root / "cache" / name

def save_cache(path, Y, info):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, Y=Y,
        converged=np.array(info["converged"]),
        iterations=np.array(info["iterations"]),
        rho=np.array(info["rho"]),
        a=np.array(info["a"]))


def load_cache(path):
    if not path.exists():
        return None

    z = np.load(path)

    return z["Y"], {
        "converged": bool(z["converged"].item()),
        "iterations": int(z["iterations"].item()),
        "rho": float(z["rho"].item()),
        "a": float(z["a"].item()),
    }

def near_endpoint(value, grid):
    grid = np.sort(np.asarray(grid))
    return value in grid[:2] or value in grid[-2:]

def trajectory_metrics(Y, M):
    tr = dominant_trajectory(Y, M)

    pos = sum(a == b for a, b in zip(
        tr["physical_positions"], TRUE_PHYSICAL_POSITIONS))

    scale = sum(a == b for a, b in zip(
        tr["scale_indices"], TRUE_SCALE_INDICES))

    joint = sum(
        p == pt and s == st
        for p, s, pt, st in zip(
            tr["physical_positions"], tr["scale_indices"],
            TRUE_PHYSICAL_POSITIONS, TRUE_SCALE_INDICES))

    T = len(TRUE_SCALE_INDICES)
    return tr, pos/T, scale/T, joint/T


def solve_path(values, X0, S, A, stage, root, lambda_s=None):
    rows, Xstart, previous = [], X0.copy(), None

    for param in np.sort(values)[::-1]:
        param = float(param)

        ls = param if stage == "s" else lambda_s
        lt = 0.0 if stage == "s" else param

        print(f"\nStage {stage.upper()}: lambda_s={ls:.5g}, lambda_t={lt:.5g}")

        path = cache_path(root, stage, param, lambda_s=ls)
        cached = load_cache(path)

        if cached:
            Y, info = cached
            print(f"cache: iter={info['iterations']} conv={info['converged']}")
        else:
            _, Y, _, info = solve_log_admm(
                X0=Xstart.copy(), S=S, operator=A,
                lambda_s=ls, lambda2=lt, tau=TAU, rho=RHO,
                max_admm_iter=MAXITER, x_maxiter=X_MAXITER,
                abs_tol=1e-4, rel_tol=1e-3,
                adaptive_rho=True, rho_mu=10, rho_tau=2,
                a_min=1e-4, continuation_iters=100,
                a_fraction=.95, verbose=False)

            # Cache immediately after expensive solve.
            save_cache(path, Y, info)

        raw_t, temp = temporal_energy(Y, A)

        rows.append({
            "parameter": param,
            "lambda_s": ls,
            "lambda_t": lt,
            "warm_start_from": previous,
            "converged": bool(info["converged"]),
            "iterations": int(info["iterations"]),
            "rho": float(info["rho"]),
            "a": float(info["a"]),
            "data": data_fidelity(Y, S, A),
            "fixed_log": fixed_log(Y),
            "softmin_raw": raw_t,
            "temporal": temp,
            "active": int(np.count_nonzero(np.abs(Y) > 1e-3)),
            "_Y": Y,
        })

        r = rows[-1]
        print(
            f"D={r['data']:.5e}  "
            f"Rs={r['fixed_log']:.5e}  "
            f"Rt={r['temporal']:.5e}  "
            f"nz={r['active']:3d}  "
            f"iter={r['iterations']:3d}  "
            f"conv={r['converged']}"
        )

        if info["converged"]:
            Xstart = Y.copy()
            previous = param

    rows.sort(key=lambda r: r["parameter"])
    return rows


def ranks(v):
    v = np.asarray(v)
    order = np.argsort(v)
    r = np.empty(len(v), float)
    r[order] = np.arange(len(v))
    return r


def spearman(x, y):
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def normalize_log(v):
    x = np.log10(np.maximum(np.asarray(v, float), 1e-15))
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

def smooth5(x):
    x = np.asarray(x, dtype=float)
    y = x.copy()

    if len(x) >= 5:
        for i in range(2, len(x)-2):
            y[i] = np.mean(x[i-2:i+3])

    return y


def curvature_persistence(k):
    """
    Three-point average curvature score.
    Suppresses isolated one-point curvature spikes.
    """
    k = np.asarray(k, dtype=float)
    score = np.full(len(k), np.nan)

    for i in range(1, len(k)-1):
        q = k[i-1:i+2]
        if np.all(np.isfinite(q)):
            score[i] = np.mean(q)

    return score


def local_data_span(rows, i, radius=1):
    """
    Relative range of data fidelity around candidate i.
    """
    lo = max(0, i-radius)
    hi = min(len(rows), i+radius+1)

    d = np.array(
        [r["data"] for r in rows[lo:hi]],
        dtype=float,
    )

    if len(d) < 2 or d.min() <= 0:
        return 0.0

    return float((d.max() - d.min()) / d.min())


def contiguous_segments(rows, original_grid):
    grid = {
        round(float(v), 10): i
        for i, v in enumerate(np.sort(original_grid))
    }

    rows = sorted(rows, key=lambda r: r["parameter"])
    result, current, previous = [], [], None

    for row in rows:
        i = grid[round(row["parameter"], 10)]

        if previous is None or i == previous + 1:
            current.append(row)
        else:
            if current:
                result.append(current)
            current = [row]

        previous = i

    if current:
        result.append(current)

    return result


def lcurve_select(rows, complexity_key, grid, exclude_zero=False):
    valid = [
        r for r in rows
        if r["converged"]
        and r["data"] > 0
        and r[complexity_key] > 0
        and (not exclude_zero or r["parameter"] > 0)
    ]

    param = [r["parameter"] for r in valid]
    data = [r["data"] for r in valid]
    comp = [r[complexity_key] for r in valid]

    rho_data = spearman(param, data)
    rho_comp = spearman(param, comp)

    data_span = (max(data) - min(data)) / min(data)

    orientation_valid = (
        rho_data > MIN_DATA_CORR
        and rho_comp < -MIN_COMPLEXITY_CORR
        and data_span > MIN_DATA_SPAN
    )

    print(
        f"L-curve orientation: "
        f"rho(lambda,data)={rho_data:.3f}, "
        f"rho(lambda,complexity)={rho_comp:.3f}, "
        f"valid={orientation_valid}"
    )

    points = []

    # Never take derivatives across a missing convergence-grid region.
    for seg_id, seg in enumerate(contiguous_segments(valid, grid)):
        if len(seg) < 3:
            continue

        x = normalize_log([r["data"] for r in seg])
        y = normalize_log([r[complexity_key] for r in seg])
        xs, ys = smooth3(x), smooth3(y)
        k = curvature(xs, ys)

        for i, row in enumerate(seg):
            row["lcurve_segment"] = seg_id
            row["lcurve_x"] = float(x[i])
            row["lcurve_y"] = float(y[i])
            row["lcurve_x_smoothed"] = float(xs[i])
            row["lcurve_y_smoothed"] = float(ys[i])
            row["curvature"] = (
                None if not np.isfinite(k[i])
                else float(k[i]))
            points.append(row)

    candidates = [r for r in points if r["curvature"] is not None]

    # Important: still report best corner even when orientation fails,
    # but label it diagnostic rather than accepted.
    corner = (
        max(candidates, key=lambda r: r["curvature"])
        if candidates else None
    )

    return {
        "orientation_valid": orientation_valid,
        "rho_lambda_data": rho_data,
        "rho_lambda_complexity": rho_comp,
        "relative_data_span": data_span,
        "corner": corner,
        "corner_accepted": corner if orientation_valid else None,
        "points": points,
    }

def lcurve_select_stage1(rows, complexity_key, grid):
    valid = [
        r for r in rows
        if (
            r["converged"]
            and r["data"] > 0
            and r[complexity_key] > 0
        )
    ]

    valid.sort(key=lambda r: r["parameter"])

    param = [r["parameter"] for r in valid]
    data = [r["data"] for r in valid]
    comp = [r[complexity_key] for r in valid]

    rho_data = spearman(param, data)
    rho_comp = spearman(param, comp)

    orientation_valid = (
        rho_data > 0
        and rho_comp < -.5
    )

    print(
        f"Stage-1 robust orientation: "
        f"rho(lambda,data)={rho_data:.3f}, "
        f"rho(lambda,complexity)={rho_comp:.3f}, "
        f"valid={orientation_valid}"
    )

    points = []

    for seg_id, seg in enumerate(
        contiguous_segments(valid, grid)
    ):
        if len(seg) < 5:
            continue

        x = normalize_log(
            [r["data"] for r in seg]
        )

        y = normalize_log(
            [r[complexity_key] for r in seg]
        )

        # More aggressive macroscopic smoothing.
        xs = smooth5(x)
        ys = smooth5(y)

        k = curvature(xs, ys)
        score = curvature_persistence(k)

        for i, row in enumerate(seg):
            span = local_data_span(
                seg,
                i,
                radius=CURVATURE_RADIUS,
            )

            row["lcurve_segment"] = seg_id
            row["lcurve_x"] = float(x[i])
            row["lcurve_y"] = float(y[i])
            row["lcurve_x_smoothed"] = float(xs[i])
            row["lcurve_y_smoothed"] = float(ys[i])

            row["curvature"] = (
                None
                if not np.isfinite(k[i])
                else float(k[i])
            )

            row["curvature_score"] = (
                None
                if not np.isfinite(score[i])
                else float(score[i])
            )

            row["local_data_span"] = float(span)

            row["corner_candidate"] = bool(
                row["curvature_score"] is not None
                and span >= MIN_LOCAL_DATA_SPAN
            )

            points.append(row)

    candidates = [
        r for r in points
        if r["corner_candidate"]
    ]

    corner = (
        max(
            candidates,
            key=lambda r: r["curvature_score"],
        )
        if candidates else None
    )

    return {
        "orientation_valid": orientation_valid,
        "rho_lambda_data": rho_data,
        "rho_lambda_complexity": rho_comp,
        "corner": corner,
        "corner_accepted":
            corner if orientation_valid else None,
        "points": points,
    }

def discrepancy(noise):
    return float(.5*np.sum(noise*noise))


def add_posthoc_truth(rows, Xtrue, Sclean, A, M):
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


def remove_arrays(rows):
    return [
        {k: v for k, v in r.items() if k != "_Y"}
        for r in rows
    ]

def clean_row(row):
    if row is None:
        return None
    return {k: v for k, v in row.items() if k != "_Y"}


def clean_lcurve(result):
    return {
        "orientation_valid": bool(result["orientation_valid"]),
        "rho_lambda_data": float(result["rho_lambda_data"]),
        "rho_lambda_complexity": float(result["rho_lambda_complexity"]),
        "corner": clean_row(result["corner"]),
        "corner_accepted": clean_row(result["corner_accepted"]),
    }


def plot_lcurve(result, title, ylabel, path):
    points = result["points"]

    if not points:
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    for seg in sorted(set(r["lcurve_segment"] for r in points)):
        q = [r for r in points if r["lcurve_segment"] == seg]

        ax.plot(
            [r["lcurve_x_smoothed"] for r in q],
            [r["lcurve_y_smoothed"] for r in q],
            "o-")

        for r in q:
            ax.annotate(
                f"{r['parameter']:.3g}",
                (r["lcurve_x_smoothed"], r["lcurve_y_smoothed"]),
                xytext=(4, 4), textcoords="offset points", fontsize=8)

    corner = result["corner"]

    if corner:
        color = "r" if result["orientation_valid"] else "orange"
        label = (
            f"Accepted={corner['parameter']:.3g}"
            if result["orientation_valid"]
            else f"Diagnostic corner={corner['parameter']:.3g}"
        )

        ax.plot(
            corner["lcurve_x_smoothed"],
            corner["lcurve_y_smoothed"],
            "*", color=color, ms=16, label=label)

    ax.set(
        xlabel="Normalized log data fidelity",
        ylabel=ylabel,
        title=title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

def plot_stage1_corner_scores(result, path):
    rows = [
        r for r in result["points"]
        if r.get("curvature_score") is not None
    ]

    if not rows:
        return

    lam = np.array([r["parameter"] for r in rows])
    raw = np.array([r["curvature"] for r in rows])
    score = np.array([r["curvature_score"] for r in rows])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogx(
        lam, raw, "o-",
        label="Pointwise curvature"
    )

    ax.semilogx(
        lam, score, "s-",
        label="Persistent curvature"
    )

    eligible = [r for r in rows if r["corner_candidate"]]

    if eligible:
        ax.scatter(
            [r["parameter"] for r in eligible],
            [r["curvature_score"] for r in eligible],
            marker="*",
            s=120,
            label="Eligible corners",
        )

    corner = result["corner_accepted"]

    if corner is not None:
        ax.axvline(
            corner["parameter"],
            color="r",
            linestyle="--",
            label=f"Selected={corner['parameter']:.3g}",
        )

    ax.set(
        xlabel="lambda_s",
        ylabel="Curvature",
        title="Robust Stage-1 corner detection",
    )

    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

def run(snr_db):
    root = Path(f"results/parameter_selection/snr_{snr_db:g}")
    root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    M, N, T = 41, 96, 7

    D = make_dictionary(M, 4.0)
    A = MultiscaleConvolutionOperator(
        D=D, scales=SCALES, center=(M+1)//2,
        N=N, normalize=True)

    Xtrue, _ = make_true_coefficients(
        L=A.L, M=M, J=len(SCALES), T=T)

    Sclean = A.forward(Xtrue)
    Snoisy, noise, achieved_snr = add_noise_snr(
        Sclean, snr_db, rng)

    target = discrepancy(noise)

    X0 = A.adjoint(Snoisy, structured=True)
    X0 /= np.max(np.abs(X0))

    print(f"\nSNR={snr_db:g} dB")
    print(f"Discrepancy diagnostic target = {target:.6e}")
    print(
        f"lambda_s grid: {LS_VALUES[0]:.2f} "
        f"to {LS_VALUES[-1]:.2f} "
        f"({len(LS_VALUES)} points)"
    )
    print(
        f"lambda_t grid: {LT_VALUES[1]:.3f} "
        f"to {LT_VALUES[-1]:.3f} "
        f"({len(LT_VALUES)-1} positive points + zero baseline)"
    )

    # ---------------------------------------------------------
    # Stage 1: lambda_T=0, select lambda_S by fixed-a log L-curve
    # ---------------------------------------------------------

    stage1 = solve_path(
        LS_VALUES, X0, Snoisy, A,
        stage="s", root=root)

    stage1_lcurve = lcurve_select_stage1(
        stage1, "fixed_log", LS_VALUES)

    ls_row = stage1_lcurve["corner_accepted"]

    print()
    print("Stage-1 robust corner")
    print("---------------------")
    print(
        f"lambda_s:          "
        f"{ls_row['lambda_s']:.6g}"
    )
    print(
        f"raw curvature:     "
        f"{ls_row['curvature']:.6g}"
    )
    print(
        f"persistent score:  "
        f"{ls_row['curvature_score']:.6g}"
    )
    print(
        f"local data span:   "
        f"{100*ls_row['local_data_span']:.2f}%"
    )

    if ls_row is None:
        raise RuntimeError(
            "Stage-1 L-curve orientation is invalid. "
            "Inspect the saved path before selecting lambda_s."
        )

    selected_ls = ls_row["lambda_s"]

    print(f"\nStage-1 selected lambda_s={selected_ls:.6g}")
    if near_endpoint(selected_ls, LS_VALUES):
        print(
            "WARNING: lambda_s corner is near grid boundary; "
            "selection range may not bracket the L-curve knee."
        )

    # ---------------------------------------------------------
    # Stage 2: fixed lambda_S, select lambda_T by temporal L-curve
    # ---------------------------------------------------------

    stage2 = solve_path(
        LT_VALUES, ls_row["_Y"], Snoisy, A,
        stage="t", root=root, lambda_s=selected_ls)

    stage2_lcurve = lcurve_select(
        stage2, "temporal", LT_VALUES, exclude_zero=True)

    stage2_valid = [
        r for r in stage2
        if r["converged"] and r["lambda_t"] > 0
    ]

    rho_lt_sparsity = spearman(
        [r["lambda_t"] for r in stage2_valid],
        [r["fixed_log"] for r in stage2_valid],
    )
    print(
        f"Stage-2 sparsity coupling: "
        f"rho(lambda_t, Rs)="
        f"{rho_lt_sparsity:.3f}"
    )

    lt_row = stage2_lcurve["corner_accepted"]

    if lt_row is None:
        raise RuntimeError(
            "Stage-2 L-curve orientation is invalid. "
            "Inspect the saved path before selecting lambda_t."
        )

    selected_lt = lt_row["lambda_t"]

    print(f"Stage-2 selected lambda_t={selected_lt:.6g}")

    # ---------------------------------------------------------
    # Post-hoc synthetic validation only
    # ---------------------------------------------------------

    add_posthoc_truth(stage1, Xtrue, Sclean, A, M)
    add_posthoc_truth(stage2, Xtrue, Sclean, A, M)

    final = lt_row
    Yfinal = final["_Y"]

    # Record discrepancy ratios without using them for selection.
    for rows in [stage1, stage2]:
        for r in rows:
            r["discrepancy_ratio"] = r["data"]/target

    summary = {
        "requested_snr_db": float(snr_db),
        "achieved_snr_db": float(achieved_snr),
        "discrepancy_target_diagnostic": target,
        "selected_lambda_s": selected_ls,
        "selected_lambda_t": selected_lt,
        "stage1_lcurve": clean_lcurve(stage1_lcurve),
        "stage2_lcurve": clean_lcurve(stage2_lcurve),
        "final_coefficient_error": relative_error(Yfinal, Xtrue),
        "final_signal_error": relative_error(A.forward(Yfinal), Sclean),
        "final_output_snr_db":
            reconstruction_snr_db(Sclean, A.forward(Yfinal)),
        "final_discrepancy_ratio": final["data"]/target,
        "stage1": remove_arrays(stage1),
        "stage2": remove_arrays(stage2),
        "cache_version": CACHE_VERSION,
        "lambda_s_grid": [float(x) for x in LS_VALUES],
        "lambda_t_grid": [float(x) for x in LT_VALUES],
        "stage2_rho_lambda_sparsity": float(rho_lt_sparsity),
    }

    with (root / "selection_summary.json").open(
            "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_default)

    plot_lcurve(
        stage1_lcurve,
        "Stage-1 sparsity L-curve",
        "Normalized log fixed-a sparsity",
        root / "stage1_lcurve.png")
    
    plot_stage1_corner_scores(
        stage1_lcurve,
        root / "stage1_corner_scores.png",
    )

    plot_lcurve(
        stage2_lcurve,
        "Stage-2 temporal L-curve",
        "Normalized log temporal energy",
        root / "stage2_lcurve.png")

    print("\nAutomatic L-curve parameter selection")
    print("-------------------------------------")
    print(f"lambda_s:          {selected_ls:.6g}")
    print(f"lambda_t:          {selected_lt:.6g}")
    print(
        f"stage1 orientation: "
        f"{stage1_lcurve['rho_lambda_data']:.3f}, "
        f"{stage1_lcurve['rho_lambda_complexity']:.3f}")
    print(
        f"stage2 orientation: "
        f"{stage2_lcurve['rho_lambda_data']:.3f}, "
        f"{stage2_lcurve['rho_lambda_complexity']:.3f}")
    print(f"discrepancy ratio: {summary['final_discrepancy_ratio']:.3f}")
    print(f"final X error:     {summary['final_coefficient_error']:.6f}")
    print(f"final S error:     {summary['final_signal_error']:.6f}")
    print(f"output SNR:        {summary['final_output_snr_db']:.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr-db", type=float, default=14.0)
    args = parser.parse_args()
    run(args.snr_db)
