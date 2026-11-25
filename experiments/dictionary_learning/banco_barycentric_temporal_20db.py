import argparse
import json
from pathlib import Path

import numpy as np

from banco_barycentric_coefficient_step import banco_barycentric_coefficient_step
from banco_coefficient_step import banco_coefficient_step
from banco_gaussian_ramp import (
    BANCO_SCALES,
    make_banco_coefficients,
    make_banco_dictionary,
)
from banco_rho_update import banco_rho_update
from banco_sparse_penalty import banco_sparse_penalty
from barycentric_transition_vectorized import (
    barycentric_transition_value_grad_vectorized,
    make_barycentric_transition_regularizer_vectorized,
)
from dictionary_alpha import compute_dictionary_alpha
from dictionary_cg import solve_dictionary_cg
from dictionary_forward_operator import DictionaryForwardOperator
from dictionary_initialization import make_banco_boxes
from dictionary_projection import project_dictionary_nonnegative
from gaussian_experiment import add_noise_snr
from multiscale_convolution import MultiscaleConvolutionOperator


def cosine(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.vdot(a, b).real / max(den, 1e-15))


def similarities(G, Gtrue):
    C = np.array([
        [cosine(G[:, i], Gtrue[:, j]) for j in range(2)]
        for i in range(2)
    ])
    if C[0, 0] + C[1, 1] >= C[0, 1] + C[1, 0]:
        return C, "direct", C[0, 0], C[1, 1]
    return C, "swapped", C[1, 0], C[0, 1]


def make_operator(G, scales, center, N):
    return MultiscaleConvolutionOperator(
        D=G, scales=scales, center=center, N=N, normalize=True
    )


def dictionary_step(
    G, H, Gprev, Y, S, rho, scales, center, N, M, relax=1.8
):
    alpha, _ = compute_dictionary_alpha(G, scales, center, N)
    op = DictionaryForwardOperator(
        X=Y, alpha=alpha, scales=scales, center=center, N=N, M=M
    )
    D, info = solve_dictionary_cg(
        op=op, S=S, G=G, H=H, rho=rho, D0=G,
        rtol=1e-10, maxiter=500,
    )
    if info != 0:
        raise RuntimeError(f"dictionary CG failed: {info}")

    Gold = G.copy()
    Dr = relax * D + (1.0 - relax) * G
    G = project_dictionary_nonnegative(Dr + H)
    rd = float(np.linalg.norm(D - G))
    sd = float(rho * np.linalg.norm(Gprev - G))
    H = H + Dr - G
    rho, H = banco_rho_update(
        rho, rd, sd, H,
        ratio=10.0, factor=2.0, dual_floor_boost=False,
    )
    return G, H, G.copy(), rho, rd, sd, float(np.linalg.norm(G - Gold))


def run_prefit(
    S, Sclean, Gtrue, shape, lambda_s, nit, scales, center, N, M
):
    G = make_banco_boxes(M, Gtrue.shape[1])
    H = np.zeros_like(G)
    X = np.zeros(shape)
    Y = np.zeros(shape)
    U = np.zeros(shape)
    Yprev, Gprev = Y.copy(), G.copy()
    rho_x, rho_d = 1000.0, 500.0

    print("\nStage 1: Banco non-temporal prefit")
    print("----------------------------------")
    print(
        "iter cleanerr cosG   cosR   nz   rhoX    rhoD    "
        "rX      sX      rD      sD      dG"
    )

    for it in range(1, nit + 1):
        rho_x_step, rho_d_step = rho_x, rho_d
        if it > 1:
            G, H, Gprev, rho_d, rd, sd, dchange = dictionary_step(
                G, H, Gprev, Y, S, rho_d_step,
                scales, center, N, M,
            )
        else:
            rd = sd = dchange = 0.0

        A = make_operator(G, scales, center, N)
        X, Y, U, _ = banco_coefficient_step(
            X, Y, U, S, A,
            lambda_s=lambda_s, rho=rho_x_step,
            iteration=it, x_relax=1.8,
        )
        rx = float(np.linalg.norm(X - Y))
        sx = float(rho_x_step * np.linalg.norm(Yprev - Y))
        rho_x, U = banco_rho_update(
            rho_x_step, rx, sx, U,
            ratio=10.0, factor=2.0, dual_floor_boost=True,
        )
        Yprev = Y.copy()

        cleanerr = float(
            np.linalg.norm(A.forward(Y) - Sclean) / np.linalg.norm(Sclean)
        )
        _, _, cos_g, cos_r = similarities(G, Gtrue)
        nz = int(np.count_nonzero(np.abs(Y) > 1e-3))
        print(
            f"{it:3d} {cleanerr:8.2e} {cos_g:6.4f} {cos_r:6.4f} "
            f"{nz:4d} {rho_x:7.1e} {rho_d:7.1e} "
            f"{rx:7.1e} {sx:7.1e} {rd:7.1e} {sd:7.1e} "
            f"{dchange:7.1e}"
        )

    return {"G": G, "Y": Y}


def save_prefit(path, prefit, args, snr, sigma, dims, scales):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    M, N, T, K, center = dims
    np.savez_compressed(
        path,
        G_prefit=prefit["G"], Y_prefit=prefit["Y"],
        lambda_s=float(args.lambda_s),
        prefit_iterations=int(args.prefit_iters),
        achieved_snr=float(snr), sigma=float(sigma), seed=100,
        M=M, N=N, T=T, K=K, center=center,
        scales=np.asarray(scales, dtype=np.int64),
    )
    print(f"\nsaved prefit: {path}")


def load_prefit(path, args, dims, scales, shape):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"prefit checkpoint not found: {path}")

    M, N, T, K, center = dims
    with np.load(path, allow_pickle=False) as data:
        G = np.asarray(data["G_prefit"], dtype=float).copy()
        Y = np.asarray(data["Y_prefit"], dtype=float).copy()
        checks = {
            "lambda_s": np.isclose(
                float(data["lambda_s"]), args.lambda_s,
                rtol=0.0, atol=1e-15,
            ),
            "prefit_iterations": (
                int(data["prefit_iterations"]) == args.prefit_iters
            ),
            "dimensions": (
                int(data["M"]), int(data["N"]), int(data["T"]),
                int(data["K"]), int(data["center"]),
            ) == dims,
            "scales": np.array_equal(
                data["scales"], np.asarray(scales, dtype=np.int64)
            ),
            "G shape": G.shape == (M, K),
            "Y shape": Y.shape == shape,
        }

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(
            "incompatible prefit checkpoint: " + ", ".join(failed)
        )

    print(f"\nloaded prefit: {path}")
    print(f"prefit G shape: {G.shape}")
    print(f"prefit Y shape: {Y.shape}")
    return {"G": G, "Y": Y}


def temporal_diagnostics(X, shifts, scale_coords, beta_v):
    value, _, td = barycentric_transition_value_grad_vectorized(
        X,
        shift_coordinates=shifts,
        scale_coordinates=scale_coords,
        gate_mass=0.05,
        mass_floor=1e-10,
        shift_velocity=1.0,
        scale_velocity=None,
        spread_weight=beta_v,
    )
    motion = float(td["motion_value"])
    spread = float(td["spread_value"])
    weighted_spread = float(td["weighted_spread_value"])
    motion_atom = np.sum(td["weighted_pair_values"], axis=1)
    spread_atom = np.sum(
        td["frame_gates"] * td["frame_spread"], axis=1
    )
    weighted_spread_atom = beta_v * spread_atom

    if not np.isclose(motion_atom.sum(), motion, rtol=1e-10, atol=1e-10):
        raise RuntimeError("per-atom motion does not sum to total")
    if not np.isclose(
        weighted_spread_atom.sum(), weighted_spread,
        rtol=1e-10, atol=1e-10,
    ):
        raise RuntimeError("per-atom spread does not sum to total")

    return {
        "value": float(value),
        "motion": motion,
        "spread": spread,
        "weighted_spread": weighted_spread,
        "motion_atom": motion_atom,
        "spread_atom": spread_atom,
        "weighted_spread_atom": weighted_spread_atom,
    }


def run_temporal(
    lambda_t, S, Sclean, Xtrue, Gtrue, G0, Y0,
    lambda_s, beta_v, nit, scales, center, N, M,
    x_solver="lbfgsb", x_maxiter=400, x_gtol=1e-4,
    rho_floor=15.625,
):
    G, Y = G0.copy(), Y0.copy()
    X = Y.copy()
    U = np.zeros_like(Y)
    H = np.zeros_like(G)
    Yprev, Gprev = Y.copy(), G.copy()
    rho_x, rho_d = 1000.0, 500.0

    shifts = np.arange(Y.shape[0], dtype=float)
    scale_coords = np.array([c1 / c2 for c1, c2 in scales], dtype=float)
    temporal = make_barycentric_transition_regularizer_vectorized(
        shift_coordinates=shifts,
        scale_coordinates=scale_coords,
        gate_mass=0.05,
        mass_floor=1e-10,
        shift_velocity=1.0,
        scale_velocity=None,
        spread_weight=beta_v,
    )

    history = []
    Ghist = np.empty((nit + 1, M, G.shape[1]))
    Ghist[0] = G

    print(
        f"\nStage 2: barycentric MCDL, lambda_t={lambda_t:g}, "
        f"beta_v={beta_v:g}, solver={x_solver}"
    )
    print("-" * 80)
    print(
        "iter cleanerr coefferr cosG   cosR   nz   Jdata   Jsparse  "
        "Jt0      Jt1      inner ok pginf   dG"
    )

    for it in range(1, nit + 1):
        rho_x_step, rho_d_step = rho_x, rho_d
        if it > 1:
            G, H, Gprev, rho_d, rd, sd, dchange = dictionary_step(
                G, H, Gprev, Y, S, rho_d_step,
                scales, center, N, M,
            )
        else:
            rd = sd = dchange = 0.0

        A = make_operator(G, scales, center, N)
        penalty_iteration = 100 + it
        X, Y, U, opt = banco_barycentric_coefficient_step(
            X, Y, U, S, A,
            lambda_s=lambda_s,
            lambda_t=lambda_t,
            temporal_regularizer=temporal,
            rho=rho_x_step,
            iteration=penalty_iteration,
            x_relax=1.8,
            maxiter=x_maxiter,
            use_fourier_init=(it == 1),
            solver=x_solver,
            gtol=x_gtol,
        )

        rx = float(np.linalg.norm(X - Y))
        sx = float(rho_x_step * np.linalg.norm(Yprev - Y))
        rho_x, U = banco_rho_update(
            rho_x_step, rx, sx, U,
            ratio=10.0, factor=2.0, dual_floor_boost=True,
        )
        if rho_x < rho_floor:
            old_rho = rho_x
            rho_x = rho_floor
            U *= old_rho / rho_x

        Yprev = Y.copy()
        Ghist[it] = G
        Shat = A.forward(Y)
        cleanerr = float(
            np.linalg.norm(Shat - Sclean) / np.linalg.norm(Sclean)
        )
        coefferr = float(
            np.linalg.norm(Y - Xtrue) / np.linalg.norm(Xtrue)
        )
        C, match, cos_g, cos_r = similarities(G, Gtrue)
        nz = int(np.count_nonzero(np.abs(Y) > 1e-3))

        td = temporal_diagnostics(X, shifts, scale_coords, beta_v)
        Jdata = float(0.5 * np.sum((Shat - S) ** 2))
        Rsp = banco_sparse_penalty(
            Y,
            lambda_s=lambda_s,
            iteration=penalty_iteration,
            l1_iters=20,
        )
        Jsparse = float(lambda_s * Rsp)
        motion_cost_atom = lambda_t * td["motion_atom"]
        spread_cost_atom = lambda_t * td["weighted_spread_atom"]
        temporal_cost_atom = motion_cost_atom + spread_cost_atom
        Jmotion = float(motion_cost_atom.sum())
        Jspread = float(spread_cost_atom.sum())
        Jtemporal = Jmotion + Jspread
        Jtotal = Jdata + Jsparse + Jtemporal
        jt0, jt1 = temporal_cost_atom

        inner_success = bool(opt.success)
        inner_pginf = float(opt.projected_grad_inf)
        inner_nit = int(opt.nit)
        inner_nfev = int(opt.nfev)

        row = {
            "iteration": it,
            "lambda_t": float(lambda_t),
            "lambda_s": float(lambda_s),
            "beta_v": float(beta_v),
            "x_solver": x_solver,
            "clean_error": cleanerr,
            "coefficient_error": coefferr,
            "cosine_gaussian": float(cos_g),
            "cosine_ramp": float(cos_r),
            "matching": match,
            "significant_count": nz,
            "rho_x": float(rho_x),
            "rho_d": float(rho_d),
            "rx": rx,
            "sx": sx,
            "rd": rd,
            "sd": sd,
            "dictionary_change": dchange,
            "data_cost": Jdata,
            "sparsity_cost": Jsparse,
            "motion_cost": Jmotion,
            "spread_cost": Jspread,
            "temporal_cost": Jtemporal,
            "total_model_cost": Jtotal,
            "temporal_value": td["value"],
            "motion_value": td["motion"],
            "spread_value": td["spread"],
            "weighted_spread_value": td["weighted_spread"],
            "motion_by_atom": td["motion_atom"].tolist(),
            "spread_by_atom": td["spread_atom"].tolist(),
            "weighted_spread_by_atom": (
                td["weighted_spread_atom"].tolist()
            ),
            "motion_cost_by_atom": motion_cost_atom.tolist(),
            "spread_cost_by_atom": spread_cost_atom.tolist(),
            "temporal_cost_by_atom": temporal_cost_atom.tolist(),
            "inner_iterations": inner_nit,
            "inner_function_evals": inner_nfev,
            "inner_success": inner_success,
            "inner_status": int(opt.status),
            "inner_message": str(opt.message),
            "inner_projected_grad_inf": inner_pginf,
            "similarity_matrix": C.tolist(),
        }
        history.append(row)

        print(
            f"{it:3d} {cleanerr:8.2e} {coefferr:8.2e} "
            f"{cos_g:6.4f} {cos_r:6.4f} {nz:4d} "
            f"{Jdata:7.2e} {Jsparse:7.2e} "
            f"{jt0:7.2e} {jt1:7.2e} "
            f"{inner_nit:5d} {int(inner_success):2d} "
            f"{inner_pginf:7.1e} {dchange:7.1e}"
        )

    return {
        "lambda_t": float(lambda_t),
        "final": history[-1],
        "history": history,
        "G": G,
        "Y": Y,
        "X": X,
        "Ghist": Ghist,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--lambda-t-values", type=float, nargs="+", default=[0.0, 3e-5]
    )
    p.add_argument("--lambda-s", type=float, default=0.24)
    p.add_argument("--beta-v", type=float, default=0.10)
    p.add_argument("--prefit-iters", type=int, default=100)
    p.add_argument("--temporal-iters", type=int, default=50)
    p.add_argument("--x-solver", choices=["lbfgsb", "spg"], default="lbfgsb")
    p.add_argument("--x-maxiter", type=int, default=400)
    p.add_argument("--x-gtol", type=float, default=1e-4)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/dictionary_learning/banco_barycentric_temporal_20db"
        ),
    )
    p.add_argument("--prefit-file", type=Path, default=None)
    p.add_argument("--save-prefit-only", type=Path, default=None)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.prefit_file is not None and args.save_prefit_only is not None:
        raise ValueError(
            "--prefit-file and --save-prefit-only are mutually exclusive"
        )

    M, N, T, K, center = 41, 80, 80, 2, 21
    dims = (M, N, T, K, center)
    scales = BANCO_SCALES
    Gtrue = make_banco_dictionary(M)
    Atrue = make_operator(Gtrue, scales, center, N)
    Xtrue = make_banco_coefficients(T, M, N)
    Sclean = Atrue.forward(Xtrue)
    S, _, snr, sigma = add_noise_snr(
        Sclean, 20.0, np.random.default_rng(100), return_sigma=True
    )
    shape = (Atrue.L, K, len(scales), T)

    print("Banco-initialized barycentric temporal MCDL")
    print("-------------------------------------------")
    print(f"achieved SNR: {snr:.3f} dB")
    print(f"sigma:        {sigma:.6e}")
    print(f"lambda_s:     {args.lambda_s:g}")
    print(f"beta_v:       {args.beta_v:g}")
    print(f"x_solver:     {args.x_solver}")
    print(f"x_maxiter:    {args.x_maxiter}")
    print(f"x_gtol:       {args.x_gtol:.1e}")
    print(f"true nz:      {np.count_nonzero(Xtrue)}")
    print(f"lambda_t:     {args.lambda_t_values}")

    if args.prefit_file is not None:
        prefit = load_prefit(args.prefit_file, args, dims, scales, shape)
    else:
        prefit = run_prefit(
            S, Sclean, Gtrue, shape,
            args.lambda_s, args.prefit_iters,
            scales, center, N, M,
        )
        if args.save_prefit_only is not None:
            save_prefit(
                args.save_prefit_only, prefit, args,
                snr, sigma, dims, scales,
            )
            return

    results = [
        run_temporal(
            lambda_t=lt,
            S=S,
            Sclean=Sclean,
            Xtrue=Xtrue,
            Gtrue=Gtrue,
            G0=prefit["G"],
            Y0=prefit["Y"],
            lambda_s=args.lambda_s,
            beta_v=args.beta_v,
            nit=args.temporal_iters,
            scales=scales,
            center=center,
            N=N,
            M=M,
            x_solver=args.x_solver,
            x_maxiter=args.x_maxiter,
            x_gtol=args.x_gtol,
        )
        for lt in args.lambda_t_values
    ]

    summary = {
        "requested_snr_db": 20.0,
        "achieved_snr_db": float(snr),
        "sigma": float(sigma),
        "seed": 100,
        "lambda_s": float(args.lambda_s),
        "beta_v": float(args.beta_v),
        "prefit_iterations": args.prefit_iters,
        "temporal_iterations": args.temporal_iters,
        "x_solver": args.x_solver,
        "x_maxiter": args.x_maxiter,
        "x_gtol": args.x_gtol,
        "true_nonzero_count": int(np.count_nonzero(Xtrue)),
        "prefit_file": (
            str(args.prefit_file) if args.prefit_file is not None else None
        ),
        "candidates": [
            {
                "lambda_t": r["lambda_t"],
                **r["final"],
                "history": r["history"],
            }
            for r in results
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    for r in results:
        tag = f"{r['lambda_t']:.8g}".replace(".", "p")
        np.savez_compressed(
            args.output_dir / f"lambda_t_{tag}.npz",
            lambda_s=args.lambda_s,
            lambda_t=r["lambda_t"],
            beta_v=args.beta_v,
            x_solver=args.x_solver,
            x_maxiter=args.x_maxiter,
            x_gtol=args.x_gtol,
            achieved_snr=snr,
            sigma=sigma,
            Gtrue=Gtrue,
            Xtrue=Xtrue,
            Sclean=Sclean,
            S=S,
            G_prefit=prefit["G"],
            Y_prefit=prefit["Y"],
            G_final=r["G"],
            X_final=r["X"],
            Y_final=r["Y"],
            Ghist=r["Ghist"],
        )

    print("\nTemporal comparison")
    print("-------------------")
    print(
        "lambda_t Jdata   Jsparse Jm0     Js0     Jm1     Js1     "
        "Jtotal  cosG   cosR   nz  inner pginf"
    )
    for r in results:
        final = r["final"]
        jm0, jm1 = final["motion_cost_by_atom"]
        js0, js1 = final["spread_cost_by_atom"]
        print(
            f"{r['lambda_t']:8.1e} "
            f"{final['data_cost']:7.3f} "
            f"{final['sparsity_cost']:7.3f} "
            f"{jm0:7.3f} {js0:7.3f} "
            f"{jm1:7.3f} {js1:7.3f} "
            f"{final['total_model_cost']:7.3f} "
            f"{final['cosine_gaussian']:6.4f} "
            f"{final['cosine_ramp']:6.4f} "
            f"{final['significant_count']:4d} "
            f"{final['inner_iterations']:5d} "
            f"{final['inner_projected_grad_inf']:7.1e}"
        )

    print(f"\nsaved: {summary_path}")


if __name__ == "__main__":
    main()
