import time
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


LAMBDA_T_VALUES = np.array([
    0.0,
    1e-4,
    3e-4,
    5e-4,
    7.5e-4,
    1e-3,
    1.5e-3,
    2e-3,
    3e-3,
    1e-2,
])

SNR_DB = 14.0
SEED = 100
LAMBDA_S = 0.125

GATE_MASS = 0.05
MASS_FLOOR = 1e-10
SHIFT_VELOCITY = 1.0

TAU = 1e-2
RHO = 0.10
MAX_ADMM_ITER = 600
X_MAXITER = 50


def trajectory_metrics(Y, M):
    tr = dominant_trajectory(Y, M)
    T = len(TRUE_SCALE_INDICES)

    pa = sum(
        a == b for a, b in zip(
            tr["physical_positions"],
            TRUE_PHYSICAL_POSITIONS,
        )
    ) / T

    sa = sum(
        a == b for a, b in zip(
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


def significant_coefficients(Y, threshold=1e-3):
    idx = np.argwhere(np.abs(Y) > threshold)

    return [
        {
            "t": int(t),
            "n": int(n),
            "k": int(k),
            "j": int(j),
            "amp": float(Y[n, k, j, t]),
        }
        for n, k, j, t in idx
    ]


def main():
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
        Sclean,
        SNR_DB,
        rng,
    )

    noise_target = 0.5 * np.sum(noise * noise)

    scales = physical_scale_values(SCALES)

    shift_coordinates = np.arange(
        A.L,
        dtype=np.float64,
    )

    temporal_regularizer = (
        make_barycentric_transition_regularizer(
            shift_coordinates=shift_coordinates,
            scale_coordinates=scales,
            gate_mass=GATE_MASS,
            mass_floor=MASS_FLOOR,
            shift_velocity=SHIFT_VELOCITY,
        )
    )

    X0 = A.adjoint(
        Snoisy,
        structured=True,
    )

    X0 = np.maximum(X0, 0.0)

    if X0.max() > 0:
        X0 /= X0.max()

    rows = []

    print()
    print("Barycentric lambda_t sweep")
    print("--------------------------")
    print(f"SNR:       {achieved_snr:.2f} dB")
    print(f"lambda_s:  {LAMBDA_S}")
    print(f"gate mass: {GATE_MASS}")
    print()

    for lt in LAMBDA_T_VALUES:
        start = time.perf_counter()

        _, Y, _, info = solve_log_admm(
            X0=X0.copy(),
            S=Snoisy,
            operator=A,
            lambda_s=LAMBDA_S,
            lambda2=float(lt),
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
            temporal_regularizer=temporal_regularizer,
        )

        elapsed = time.perf_counter() - start

        Shat = A.forward(Y)
        residual = Shat - Snoisy

        data = 0.5 * np.sum(residual * residual)

        xerr = relative_error(Y, Xtrue)
        serr = relative_error(Shat, Sclean)
        out_snr = reconstruction_snr_db(Sclean, Shat)

        tr, pa, sa, ja = trajectory_metrics(Y, M)

        sig = significant_coefficients(Y)
        nz = len(sig)

        rt, _, td = barycentric_transition_value_grad(
            Y,
            shift_coordinates=shift_coordinates,
            scale_coordinates=scales,
            gate_mass=GATE_MASS,
            mass_floor=MASS_FLOOR,
            shift_velocity=SHIFT_VELOCITY,
        )

        row = {
            "lambda_t": float(lt),
            "converged": bool(info["converged"]),
            "iterations": int(info["iterations"]),
            "nz": nz,
            "discrepancy": float(data / noise_target),
            "xerr": float(xerr),
            "serr": float(serr),
            "snr": float(out_snr),
            "pos": float(pa),
            "scale": float(sa),
            "joint": float(ja),
            "rt": float(rt),
            "weighted_rt": float(lt * rt),
            "runtime": float(elapsed),
            "trajectory": tr,
            "significant": sig,
            "masses": td["masses"][0].copy(),
            "position_centroids":
                td["position_centroids"][0].copy(),
            "log_scale_centroids":
                td["log_scale_centroids"][0].copy(),
        }

        rows.append(row)

        print(
            f"lt={lt:8.1e} "
            f"conv={str(row['converged']):5s} "
            f"iter={row['iterations']:3d} "
            f"nz={row['nz']:2d} "
            f"D/Dn={row['discrepancy']:.3f} "
            f"Xerr={row['xerr']:.4f} "
            f"SNR={row['snr']:5.2f} "
            f"joint={100*row['joint']:5.1f}% "
            f"Rt={row['rt']:7.3f} "
            f"time={row['runtime']:5.1f}s"
        )

    print()
    print("Summary")
    print("-------")
    print(
        "lambda_t  conv iter  nz  D/Dn   "
        "Xerr    SNRout  joint    Rt"
    )

    for r in rows:
        print(
            f"{r['lambda_t']:8.1e} "
            f"{str(r['converged']):5s} "
            f"{r['iterations']:4d} "
            f"{r['nz']:3d} "
            f"{r['discrepancy']:6.3f} "
            f"{r['xerr']:7.4f} "
            f"{r['snr']:7.2f} "
            f"{100*r['joint']:6.1f}% "
            f"{r['rt']:7.3f}"
        )

    print()
    print("Dominant trajectories")
    print("---------------------")

    for r in rows:
        tr = r["trajectory"]
        print(
            f"lt={r['lambda_t']:8.1e} "
            f"pos={tr['physical_positions']} "
            f"scale={tr['scale_indices']}"
        )

    print()
    print("Significant-coefficient details")
    print("-------------------------------")

    for r in rows:
        print()
        print(
            f"lambda_t={r['lambda_t']:.1e}, "
            f"nz={r['nz']}"
        )

        for c in r["significant"]:
            print(
                f"  t={c['t']} "
                f"n={c['n']} "
                f"j={c['j']} "
                f"amp={c['amp']:.6f}"
            )


if __name__ == "__main__":
    main()
    