"""Hierarchical adaptive selector for one finite-track noise realization.

Run from the project root:
    python -m experiments.parameter_selection.adaptive_barycentric_selector

If saved in the project root instead:
    python .\adaptive_barycentric_selector.py

Selection never uses Xtrue. Truth is used only in the final post-hoc section.
"""
from email import parser
import json
import time
import argparse

from pathlib import Path

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
from log_admm import log_penalty
from multiscale_convolution import MultiscaleConvolutionOperator

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SNR_DB = 14.0
SEED = 100
M, N, T = 41, 96, 12
ACTIVE_FRAMES = np.arange(2, 10)
TRUE_POSITIONS = np.array([28, 29, 30, 31, 32, 33, 34, 35])
TRUE_SCALES = np.array([0, 1, 2, 3, 3, 2, 1, 0])

LAMBDA_S_GRID = np.array([0.075, 0.10, 0.125, 0.15, 0.175, 0.20])
LAMBDA_T_GRID = np.array([0.0, 3e-6, 1e-5, 3e-5, 1e-4])
BETA_GRID = np.array([0.0, 0.05, 0.075, 0.0875, 0.10, 0.125, 0.15])

DISC_LO, DISC_HI = 0.95, 1.05
N_LAMBDA_S_RETAIN = 3
N_TEMPORAL_RETAIN = 2

GATE_MASS = 0.05
MASS_FLOOR = 1e-10
SHIFT_VELOCITY = 1.0
TAU = 1e-2
RHO = 0.10
MAX_ADMM_ITER = 600
X_MAXITER = 50
COEFF_THRESHOLD = 1e-3
ACTIVITY_THRESHOLD = 0.05

OUT = None
CACHE = None

# -----------------------------------------------------------------------------
# Problem and diagnostics
# -----------------------------------------------------------------------------
def make_truth(A):
    X = np.zeros((A.L, A.K, A.J, T), dtype=float)
    for t, pos, j in zip(ACTIVE_FRAMES, TRUE_POSITIONS, TRUE_SCALES):
        X[int(pos + M - 1), 0, int(j), int(t)] = 1.0
    return X


def temporal_callback(lt, beta, shifts, scales):
    if lt == 0:
        return None
    return make_barycentric_transition_regularizer(
        shift_coordinates=shifts,
        scale_coordinates=scales,
        gate_mass=GATE_MASS,
        mass_floor=MASS_FLOOR,
        shift_velocity=SHIFT_VELOCITY,
        spread_weight=beta,
    )


def dominant_structure(Y):
    masses = np.sum(Y, axis=(0, 1, 2))
    activity = masses > ACTIVITY_THRESHOLD
    positions, scales = [], []
    for t in range(T):
        if not activity[t]:
            positions.append(None)
            scales.append(None)
            continue
        frame = np.abs(Y[:, :, :, t])
        n, _, j = np.unravel_index(np.argmax(frame), frame.shape)
        positions.append(int(n - (M - 1)))
        scales.append(int(j))
    return masses, activity.astype(int).tolist(), positions, scales


def structure_signature(row):
    return (
        tuple(row["activity"]),
        tuple(row["positions"]),
        tuple(row["scales"]),
        int(row["support"]),
    )


def candidate_metrics(Y, info, ls, lt, beta, A, Snoisy, Sclean, Xtrue,
                      sigma, noise, shifts, scales, runtime, stage):
    Shat = A.forward(Y)
    residual = Shat - Snoisy
    data = float(0.5 * np.sum(residual * residual))
    target_expected = float(0.5 * Snoisy.size * sigma**2)
    target_realized = float(0.5 * np.sum(noise * noise))
    support = int(np.count_nonzero(np.abs(Y) > COEFF_THRESHOLD))
    masses, activity, positions, recovered_scales = dominant_structure(Y)

    if lt == 0:
        motion = spread = temporal = 0.0
    else:
        temporal, _, td = barycentric_transition_value_grad(
            Y, shift_coordinates=shifts, scale_coordinates=scales,
            gate_mass=GATE_MASS, mass_floor=MASS_FLOOR,
            shift_velocity=SHIFT_VELOCITY, spread_weight=beta,
        )
        motion = float(td["motion_value"])
        spread = float(td["spread_value"])

    a = float(info["a"])
    raw_log = float(log_penalty(Y, 1.0, a))
    sparsity = float(log_penalty(Y, ls, a))

    # Post-hoc fields are stored but never used by selectors below.
    pos_active = [positions[int(t)] for t in ACTIVE_FRAMES]
    scale_active = [recovered_scales[int(t)] for t in ACTIVE_FRAMES]
    posthoc = {
        "coefficient_error": float(relative_error(Y, Xtrue)),
        "output_snr_db": float(reconstruction_snr_db(Sclean, Shat)),
        "position_accuracy": float(np.mean(np.array(pos_active) == TRUE_POSITIONS)),
        "scale_accuracy": float(np.mean(np.array(scale_active) == TRUE_SCALES)),
        "joint_accuracy": float(np.mean(
            (np.array(pos_active) == TRUE_POSITIONS)
            & (np.array(scale_active) == TRUE_SCALES)
        )),
    }

    return {
        "stage": stage,
        "lambda_s": float(ls),
        "lambda_t": float(lt),
        "beta_v": float(beta),
        "converged": bool(info["converged"]),
        "iterations": int(info["iterations"]),
        "rho": float(info["rho"]),
        "a": a,
        "data": data,
        "noise_target_expected": target_expected,
        "noise_target_realized": target_realized,
        "discrepancy_expected": data / target_expected,
        "discrepancy_realized": data / target_realized,
        "support": support,
        "activity": activity,
        "positions": positions,
        "scales": recovered_scales,
        "mass_min": float(np.min(masses)),
        "mass_max": float(np.max(masses)),
        "log_penalty_raw": raw_log,
        "weighted_log_penalty": sparsity,
        "motion": motion,
        "spread": spread,
        "temporal_raw": float(temporal),
        "weighted_temporal": float(lt * temporal),
        "objective_observed": float(data + sparsity + lt * temporal),
        "runtime": float(runtime),
        "posthoc": posthoc,
    }

# -----------------------------------------------------------------------------
# Cache and solve
# -----------------------------------------------------------------------------
def cache_name(stage, ls, lt, beta):
    return CACHE / (
        f"{stage}_ls_{ls:.6f}_lt_{lt:.8f}_beta_{beta:.4f}.npz"
    )


def save_cache(path, X, Y, U, info):
    path.parent.mkdir(parents=True, exist_ok=True)
    s = info["state"]
    np.savez_compressed(
        path, X=X, Y=Y, U=U,
        state_X=s["X"], state_Y=s["Y"], state_U=s["U"],
        state_rho=np.array(s["rho"]), state_a=np.array(s["a"]),
        state_iteration=np.array(s["iteration"]),
        converged=np.array(info["converged"]),
        iterations=np.array(info["iterations"]),
        rho=np.array(info["rho"]), a=np.array(info["a"]),
    )


def load_cache(path):
    if not path.exists():
        return None
    z = np.load(path)
    state = {
        "X": z["state_X"], "Y": z["state_Y"], "U": z["state_U"],
        "rho": float(z["state_rho"]), "a": float(z["state_a"]),
        "iteration": int(z["state_iteration"]),
    }
    info = {
        "state": state,
        "converged": bool(z["converged"]),
        "iterations": int(z["iterations"]),
        "rho": float(z["rho"]), "a": float(z["a"]),
    }
    return z["X"], z["Y"], z["U"], info


def solve_candidate(stage, ls, lt, beta, state0, Xbase, A, Snoisy,
                    Sclean, Xtrue, sigma, noise, shifts, scales):
    path = cache_name(stage, ls, lt, beta)
    cached = load_cache(path)
    if cached is not None:
        X, Y, U, info = cached
        runtime = 0.0
    else:
        reg = temporal_callback(lt, beta, shifts, scales)
        start = time.perf_counter()
        X, Y, U, info = solve_log_admm_stateful(
            X0=Xbase if state0 is None else state0["X"],
            S=Snoisy, operator=A, lambda_s=ls, lambda2=lt, tau=TAU,
            rho=RHO, max_admm_iter=MAX_ADMM_ITER, x_maxiter=X_MAXITER,
            abs_tol=1e-4, rel_tol=1e-3, adaptive_rho=True,
            rho_mu=10, rho_tau=2, a_min=1e-4,
            continuation_iters=100, a_fraction=.95, verbose=False,
            temporal_regularizer=reg, state0=state0,
            continue_iteration=False,
        )
        runtime = time.perf_counter() - start
        save_cache(path, X, Y, U, info)

    row = candidate_metrics(
        Y, info, ls, lt, beta, A, Snoisy, Sclean, Xtrue,
        sigma, noise, shifts, scales, runtime, stage,
    )
    return row, info["state"]

# -----------------------------------------------------------------------------
# Selection utilities: observable fields only
# -----------------------------------------------------------------------------
def discrepancy_error(row):
    return abs(np.log(row["discrepancy_expected"]))


def feasible(rows):
    return [
        r for r in rows
        if r["converged"] and DISC_LO <= r["discrepancy_expected"] <= DISC_HI
    ]


def rank_stage_a(rows):
    pool = feasible(rows) or [r for r in rows if r["converged"]]
    # Expected-discrepancy closeness, then lower support, then interior lambda_s.
    center = float(np.median(LAMBDA_S_GRID))
    return sorted(pool, key=lambda r: (
        discrepancy_error(r), r["support"], abs(r["lambda_s"] - center)
    ))


def discrete_curvature_selector(rows, x_name, y_name, parameter_name):
    rows = sorted(rows, key=lambda r: r[parameter_name])
    if len(rows) < 3:
        return min(rows, key=discrepancy_error)
    pts = np.array([
        [np.log(max(r[x_name], 1e-300)), np.log(max(r[y_name], 1e-300))]
        for r in rows
    ])
    curvatures = np.full(len(rows), -np.inf)
    for i in range(1, len(rows) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        ab, bc, ca = np.linalg.norm(b-a), np.linalg.norm(c-b), np.linalg.norm(a-c)
        u= b - a; v = c - a
        area2 = np.abs(u[0]*v[1] - u[1]*v[0])
        denom = ab * bc * ca
        curvatures[i] = 2.0 * area2 / denom if denom > 0 else 0.0
    return rows[int(np.argmax(curvatures))]


def longest_stable_run(rows, parameter_name):
    rows = sorted(feasible(rows), key=lambda r: r[parameter_name])
    if not rows:
        return []
    runs, current = [], [rows[0]]
    for row in rows[1:]:
        if structure_signature(row) == structure_signature(current[-1]):
            current.append(row)
        else:
            runs.append(current)
            current = [row]
    runs.append(current)
    runs.sort(key=lambda run: (-len(run), np.mean([discrepancy_error(r) for r in run])))
    return runs[0]


def center_of_run(run, parameter_name):
    if not run:
        return None
    values = np.array([r[parameter_name] for r in run], dtype=float)
    center = 0.5 * (values.min() + values.max())
    return min(run, key=lambda r: (
        abs(r[parameter_name] - center), discrepancy_error(r)
    ))


def public_row(row):
    return {k: v for k, v in row.items() if k != "posthoc"}

# -----------------------------------------------------------------------------
# Main hierarchy
# -----------------------------------------------------------------------------
def main():
    global SNR_DB, SEED, OUT, CACHE

    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=float, default=SNR_DB)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    SNR_DB = args.snr
    SEED = args.seed
    OUT = args.out or Path(f"results/adaptive_selector_snr{SNR_DB:g}_seed{SEED}")
    CACHE = OUT / "cache"

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    D = make_dictionary(M, 4.0)
    A = MultiscaleConvolutionOperator(
        D=D, scales=SCALES, center=(M + 1) // 2, N=N, normalize=True
    )
    Xtrue = make_truth(A)
    Sclean = A.forward(Xtrue)
    Snoisy, noise, achieved_snr, sigma = add_noise_snr(
        Sclean, SNR_DB, rng, return_sigma=True
    )
    scales = physical_scale_values(SCALES)
    shifts = np.arange(A.L, dtype=float)
    Xbase = np.maximum(A.adjoint(Snoisy, structured=True), 0.0)
    if Xbase.max() > 0:
        Xbase /= Xbase.max()

    all_rows = []
    stage_a = []
    print("\nStage A: sparsity sweep")
    print("-----------------------")
    for ls in LAMBDA_S_GRID:
        row, _ = solve_candidate(
            "A", float(ls), 0.0, 0.0, None, Xbase, A, Snoisy,
            Sclean, Xtrue, sigma, noise, shifts, scales,
        )
        stage_a.append(row); all_rows.append(row)
        print(
            f"ls={ls:.3f} conv={row['converged']} support={row['support']:3d} "
            f"D/Dexp={row['discrepancy_expected']:.3f} "
            f"Rlog={row['log_penalty_raw']:.3e}"
        )

    ranked_a = rank_stage_a(stage_a)
    retained_ls = []
    for row in ranked_a:
        if row["lambda_s"] not in retained_ls:
            retained_ls.append(row["lambda_s"])
        if len(retained_ls) == N_LAMBDA_S_RETAIN:
            break
    lcurve_a = discrete_curvature_selector(
        stage_a, "data", "log_penalty_raw", "lambda_s"
    )

    print("\nRetained lambda_s:", retained_ls)
    print("Stage-A L-curve lambda_s:", lcurve_a["lambda_s"])

    stage_b_by_ls, retained_pairs = {}, []
    print("\nStage B: temporal sweep")
    print("-----------------------")
    for ls in retained_ls:
        rows, state = [], None
        for lt in LAMBDA_T_GRID:
            row, state = solve_candidate(
                "B", ls, float(lt), 0.0, state, Xbase, A, Snoisy,
                Sclean, Xtrue, sigma, noise, shifts, scales,
            )
            rows.append(row); all_rows.append(row)
            print(
                f"ls={ls:.3f} lt={lt:.1e} support={row['support']:3d} "
                f"D/Dexp={row['discrepancy_expected']:.3f} "
                f"motion={row['motion']:.3e}"
            )
        stage_b_by_ls[str(ls)] = rows
        run = longest_stable_run(rows, "lambda_t")
        center = center_of_run(run, "lambda_t")
        if center is None:
            center = min(rows, key=discrepancy_error)
        retained_pairs.append(center)

    retained_pairs = sorted(retained_pairs, key=lambda r: (
        -len(longest_stable_run(stage_b_by_ls[str(r['lambda_s'])], "lambda_t")),
        discrepancy_error(r), r["support"]
    ))[:N_TEMPORAL_RETAIN]

    stage_c_groups = []
    print("\nStage C: spread sweep")
    print("---------------------")
    for pair in retained_pairs:
        ls, lt = pair["lambda_s"], pair["lambda_t"]
        rows, state = [], None
        for beta in BETA_GRID:
            row, state = solve_candidate(
                "C", ls, lt, float(beta), state, Xbase, A, Snoisy,
                Sclean, Xtrue, sigma, noise, shifts, scales,
            )
            rows.append(row); all_rows.append(row)
            print(
                f"ls={ls:.3f} lt={lt:.1e} beta={beta:.4f} "
                f"support={row['support']:3d} D/Dexp={row['discrepancy_expected']:.3f} "
                f"spread={row['spread']:.3e}"
            )
        stage_c_groups.append(rows)

    # Stability selection: longest feasible identical-structure beta run across groups.
    stable_candidates = []
    for rows in stage_c_groups:
        run = longest_stable_run(rows, "beta_v")
        center = center_of_run(run, "beta_v")
        if center is not None:
            stable_candidates.append((len(run), center, run))
    if stable_candidates:
        stable_candidates.sort(key=lambda x: (-x[0], discrepancy_error(x[1]), x[1]["support"]))
        _, stability_selected, selected_run = stable_candidates[0]
    else:
        stability_selected = min(all_rows, key=discrepancy_error)
        selected_run = [stability_selected]

    # Nested L-curve selection: Stage-A corner, nearest tested B group, then spread curve.
    lcurve_ls = min(retained_ls, key=lambda x: abs(x - lcurve_a["lambda_s"]))
    b_rows = stage_b_by_ls[str(lcurve_ls)]
    positive_b = [r for r in b_rows if r["lambda_t"] > 0]
    lcurve_b = discrete_curvature_selector(
        positive_b or b_rows, "data", "motion", "lambda_t"
    )
    c_rows = min(
        stage_c_groups,
        key=lambda rows: abs(rows[0]["lambda_s"] - lcurve_b["lambda_s"])
        + 1e4 * abs(rows[0]["lambda_t"] - lcurve_b["lambda_t"]),
    )
    positive_c = [r for r in c_rows if r["beta_v"] > 0 and r["spread"] > 0]
    lcurve_selected = discrete_curvature_selector(
        positive_c or c_rows, "data", "spread", "beta_v"
    )

    oracle = min(all_rows, key=lambda r: r["posthoc"]["coefficient_error"])

    summary = {
        "experiment": {
            "requested_snr_db": SNR_DB,
            "achieved_snr_db": float(achieved_snr),
            "sigma": float(sigma),
            "seed": SEED,
            "discrepancy_band": [DISC_LO, DISC_HI],
        },
        "grids": {
            "lambda_s": LAMBDA_S_GRID.tolist(),
            "lambda_t": LAMBDA_T_GRID.tolist(),
            "beta_v": BETA_GRID.tolist(),
        },
        "selection_uses_truth": False,
        "retained_lambda_s": retained_ls,
        "retained_temporal_pairs": [public_row(r) for r in retained_pairs],
        "stability_selection": public_row(stability_selected),
        "stability_region": [public_row(r) for r in selected_run],
        "lcurve_selection": public_row(lcurve_selected),
        "posthoc_validation": {
            "stability_selection": stability_selected["posthoc"],
            "lcurve_selection": lcurve_selected["posthoc"],
            "oracle_parameters": {
                "lambda_s": oracle["lambda_s"],
                "lambda_t": oracle["lambda_t"],
                "beta_v": oracle["beta_v"],
            },
            "oracle_metrics": oracle["posthoc"],
        },
        "candidates": [public_row(r) for r in all_rows],
    }

    with (OUT / "adaptive_selector_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nAdaptive selections")
    print("-------------------")
    for name, row in [
        ("stability", stability_selected),
        ("L-curve", lcurve_selected),
        ("oracle", oracle),
    ]:
        p = row["posthoc"]
        print(
            f"{name:10s}: ls={row['lambda_s']:.3f} lt={row['lambda_t']:.1e} "
            f"beta={row['beta_v']:.4f} support={row['support']:3d} "
            f"D/Dexp={row['discrepancy_expected']:.3f} "
            f"Xerr={p['coefficient_error']:.4f} joint={100*p['joint_accuracy']:.1f}%"
        )
    print("\nSaved:", OUT / "adaptive_selector_summary.json")


if __name__ == "__main__":
    main()
