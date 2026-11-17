import json
import time
from pathlib import Path

import numpy as np

from gaussian_experiment import (
    SCALES, TRUE_PHYSICAL_POSITIONS, TRUE_SCALE_INDICES,
    add_noise_snr, dominant_trajectory, make_dictionary,
    make_true_coefficients, reconstruction_snr_db, relative_error,
)
from barycentric_transition import (
    barycentric_transition_value_grad,
    make_barycentric_transition_regularizer,
)
from local_graph import physical_scale_values
from log_admm import solve_log_admm
from multiscale_convolution import MultiscaleConvolutionOperator


LS_VALUES = np.array([
    .075, .10, .125, .15, .175, .20
])

LT_VALUES = np.array([
    0, .0005, .001, .0015, .002, .003, .005
])

SNR_DB = 14.0
SEED = 100

GATE_MASS = .05
MASS_FLOOR = 1e-10
SHIFT_VELOCITY = 1.0

TAU = 1e-2
RHO = .10
MAX_ADMM_ITER = 600
X_MAXITER = 50

OUT = Path("results/barycentric_joint_snr14")
CACHE = OUT / "cache"
CACHE_VERSION = "v1_barycentric_joint"


def cache_path(ls, lt):
    return CACHE / (
        f"{CACHE_VERSION}_"
        f"ls_{ls:.6f}_lt_{lt:.6f}.npz"
    )


def save_cache(ls, lt, Y, info):
    CACHE.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        cache_path(ls, lt),
        Y=Y,
        converged=np.array(info["converged"]),
        iterations=np.array(info["iterations"]),
        rho=np.array(info["rho"]),
        a=np.array(info["a"]),
    )


def load_cache(ls, lt):
    p = cache_path(ls, lt)

    if not p.exists():
        return None

    z = np.load(p)

    return z["Y"], {
        "converged": bool(z["converged"].item()),
        "iterations": int(z["iterations"].item()),
        "rho": float(z["rho"].item()),
        "a": float(z["a"].item()),
    }


def trajectory_metrics(Y, M):
    tr = dominant_trajectory(Y, M)
    T = len(TRUE_SCALE_INDICES)

    pa = sum(
        a == b
        for a, b in zip(
            tr["physical_positions"],
            TRUE_PHYSICAL_POSITIONS,
        )
    ) / T

    sa = sum(
        a == b
        for a, b in zip(
            tr["scale_indices"],
            TRUE_SCALE_INDICES,
        )
    ) / T

    ja = sum(
        p == pt and s == st
        for p, s, pt, st in zip(
            tr["physical_positions"],
            tr["scale_indices"],
            TRUE_PHYSICAL_POSITIONS,
            TRUE_SCALE_INDICES,
        )
    ) / T

    return tr, pa, sa, ja


def clean(r):
    return {
        k: v
        for k, v in r.items()
        if not k.startswith("_")
    }


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    M, N, T = 41, 96, 7

    D = make_dictionary(M, 4.0)

    A = MultiscaleConvolutionOperator(
        D=D,
        scales=SCALES,
        center=(M + 1) // 2,
        N=N,
        normalize=True,
    )

    Xtrue, _ = make_true_coefficients(
        L=A.L,
        M=M,
        J=len(SCALES),
        T=T,
    )

    Sclean = A.forward(Xtrue)

    Snoisy, noise, achieved_snr = add_noise_snr(
        Sclean, SNR_DB, rng
    )

    noise_target = float(
        .5 * np.sum(noise * noise)
    )

    transport_scales = physical_scale_values(
        SCALES
    )

    shift_coordinates = np.arange(
        A.L,
        dtype=np.float64,
    )

    temporal_regularizer = (
        make_barycentric_transition_regularizer(
            shift_coordinates=shift_coordinates,
            scale_coordinates=transport_scales,
            gate_mass=GATE_MASS,
            mass_floor=MASS_FLOOR,
            shift_velocity=SHIFT_VELOCITY,
        )
    )

    Xbase = A.adjoint(
        Snoisy,
        structured=True,
    )

    Xbase = np.maximum(Xbase, 0.0)

    if Xbase.max() > 0:
        Xbase /= Xbase.max()

    rows = []

    print()
    print("Barycentric joint grid")
    print("----------------------")
    print(f"SNR: {achieved_snr:.2f} dB")
    print(
        f"grid: {len(LS_VALUES)} x "
        f"{len(LT_VALUES)} = "
        f"{len(LS_VALUES)*len(LT_VALUES)}"
    )

    # Each point starts from the same Xbase.
    # This avoids traversal-dependent warm-start effects.
    for ls in LS_VALUES:
        for lt in LT_VALUES:
            ls = float(ls)
            lt = float(lt)

            print()
            print(
                f"ls={ls:.4f}, lt={lt:.4g}"
            )

            cached = load_cache(ls, lt)

            if cached is not None:
                Y, info = cached
                elapsed = 0.0
                print("cache")
            else:
                start = time.perf_counter()

                _, Y, _, info = solve_log_admm(
                    X0=Xbase.copy(),
                    S=Snoisy,
                    operator=A,
                    lambda_s=ls,
                    lambda2=lt,
                    tau=TAU,
                    rho=RHO,
                    max_admm_iter=MAX_ADMM_ITER,
                    x_maxiter=X_MAXITER,
                    abs_tol=1e-4,
                    rel_tol=1e-3,
                    adaptive_rho=True,
                    rho_mu=10,
                    rho_tau=2,
                    a_min=1e-4,
                    continuation_iters=100,
                    a_fraction=.95,
                    verbose=False,
                    temporal_regularizer=
                        temporal_regularizer,
                )

                elapsed = (
                    time.perf_counter() - start
                )

                save_cache(
                    ls, lt, Y, info
                )

            Shat = A.forward(Y)
            residual = Shat - Snoisy

            data = float(
                .5
                * np.sum(residual * residual)
            )

            rt, _, td = (
                barycentric_transition_value_grad(
                    Y,
                    shift_coordinates=
                        shift_coordinates,
                    scale_coordinates=
                        transport_scales,
                    gate_mass=GATE_MASS,
                    mass_floor=MASS_FLOOR,
                    shift_velocity=
                        SHIFT_VELOCITY,
                )
            )

            tr, pa, sa, ja = (
                trajectory_metrics(Y, M)
            )

            row = {
                "lambda_s": ls,
                "lambda_t": lt,
                "converged":
                    bool(info["converged"]),
                "iterations":
                    int(info["iterations"]),
                "rho": float(info["rho"]),
                "a": float(info["a"]),
                "data": data,
                "discrepancy_ratio":
                    data / noise_target,
                "active": int(
                    np.count_nonzero(
                        np.abs(Y) > 1e-3
                    )
                ),
                "temporal": float(rt),
                "weighted_temporal":
                    float(lt * rt),
                "coefficient_error":
                    float(
                        relative_error(
                            Y, Xtrue
                        )
                    ),
                "signal_error":
                    float(
                        relative_error(
                            Shat, Sclean
                        )
                    ),
                "output_snr_db":
                    float(
                        reconstruction_snr_db(
                            Sclean, Shat
                        )
                    ),
                "position_accuracy":
                    float(pa),
                "scale_accuracy":
                    float(sa),
                "trajectory_accuracy":
                    float(ja),
                "trajectory": tr,
                "masses":
                    td["masses"][0].tolist(),
                "position_centroids":
                    td[
                        "position_centroids"
                    ][0].tolist(),
                "log_scale_centroids":
                    td[
                        "log_scale_centroids"
                    ][0].tolist(),
                "runtime": float(elapsed),
            }

            rows.append(row)

            print(
                f"conv={row['converged']} "
                f"iter={row['iterations']} "
                f"nz={row['active']} "
                f"D/Dn="
                f"{row['discrepancy_ratio']:.3f} "
                f"Xerr="
                f"{row['coefficient_error']:.4f} "
                f"SNR="
                f"{row['output_snr_db']:.2f} "
                f"joint="
                f"{100*row['trajectory_accuracy']:.1f}%"
            )

    converged = [
        r for r in rows
        if r["converged"]
    ]

    # Post-hoc synthetic oracle only.
    oracle = min(
        converged,
        key=lambda r:
            r["coefficient_error"],
    )

    # Known-noise feasible set.
    feasible = [
        r for r in converged
        if .90
        <= r["discrepancy_ratio"]
        <= 1.10
    ]

    # Minimum support in feasible region.
    min_active = min(
        r["active"]
        for r in feasible
    )

    sparse_feasible = [
        r for r in feasible
        if r["active"] == min_active
    ]

    summary = {
        "snr_db":
            float(achieved_snr),
        "noise_target":
            noise_target,
        "lambda_s_grid":
            [float(x) for x in LS_VALUES],
        "lambda_t_grid":
            [float(x) for x in LT_VALUES],
        "discrepancy_band":
            [.90, 1.10],
        "minimum_feasible_support":
            int(min_active),
        "oracle":
            clean(oracle),
        "sparse_feasible":
            [clean(r) for r in sparse_feasible],
        "results":
            [clean(r) for r in rows],
    }

    with (
        OUT / "barycentric_joint_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("Joint-grid summary")
    print("------------------")
    print(
        f"converged: "
        f"{len(converged)}/{len(rows)}"
    )
    print(
        f"feasible D/Dn .90-1.10: "
        f"{len(feasible)}"
    )
    print(
        f"minimum feasible support: "
        f"{min_active}"
    )

    print()
    print("Synthetic oracle")
    print("----------------")
    print(
        f"ls={oracle['lambda_s']:.4f}, "
        f"lt={oracle['lambda_t']:.4g}"
    )
    print(
        f"nz={oracle['active']}, "
        f"D/Dn="
        f"{oracle['discrepancy_ratio']:.4f}, "
        f"Xerr="
        f"{oracle['coefficient_error']:.5f}, "
        f"SNR="
        f"{oracle['output_snr_db']:.2f}"
    )

    print()
    print("Sparse feasible solutions")
    print("-------------------------")
    print(
        "lambda_s lambda_t nz "
        "D/Dn  Xerr  SNR  joint"
    )

    for r in sparse_feasible:
        print(
            f"{r['lambda_s']:8.3f} "
            f"{r['lambda_t']:8.4g} "
            f"{r['active']:2d} "
            f"{r['discrepancy_ratio']:5.3f} "
            f"{r['coefficient_error']:6.4f} "
            f"{r['output_snr_db']:5.2f} "
            f"{100*r['trajectory_accuracy']:5.1f}%"
        )


if __name__ == "__main__":
    run()

    