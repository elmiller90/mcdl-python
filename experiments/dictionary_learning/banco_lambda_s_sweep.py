import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from banco_coefficient_step import banco_coefficient_step
from banco_gaussian_ramp import (
    BANCO_SCALES,
    make_banco_coefficients,
    make_banco_dictionary,
)
from banco_rho_update import banco_rho_update
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


def similarity_matrix(G, Gtrue):
    return np.array([
        [cosine(G[:, i], Gtrue[:, j]) for j in range(2)]
        for i in range(2)
    ])


def best_match(C):
    direct = C[0, 0] + C[1, 1]
    swapped = C[0, 1] + C[1, 0]
    if direct >= swapped:
        return "direct", [0, 1], C[0, 0], C[1, 1]
    return "swapped", [1, 0], C[1, 0], C[0, 1]


def save_final_plot(path, G, Gtrue, match, lambda_s):
    order = [0, 1] if match == "direct" else [1, 0]
    x = np.arange(G.shape[0]) - (G.shape[0] - 1) / 2
    names = ("Gaussian", "Ramp")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for j in range(2):
        ax[j].plot(
            x, Gtrue[:, j], "k--", linewidth=2,
            label=f"True {names[j]}",
        )
        ax[j].plot(
            x, G[:, order[j]], linewidth=2,
            label=f"Learned atom {order[j]}",
        )
        ax[j].set_title(
            f"{names[j]} cosine = "
            f"{cosine(G[:, order[j]], Gtrue[:, j]):.4f}"
        )
        ax[j].set_xlabel("Base-atom coordinate")
        ax[j].grid(alpha=0.25)
        ax[j].legend()

    ax[0].set_ylabel("Amplitude")
    fig.suptitle(
        f"Banco prefit final atoms: lambda_s={lambda_s:.6g}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_surface_plot(path, Ghist, lambda_s):
    iterations = np.arange(Ghist.shape[0])
    coords = (
        np.arange(Ghist.shape[2])
        - (Ghist.shape[2] - 1) / 2
    )
    XX, YY = np.meshgrid(coords, iterations)

    fig = plt.figure(figsize=(13, 5))

    for k in range(2):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot_surface(
            XX, YY, Ghist[:, k, :],
            cmap="viridis",
            linewidth=0,
            antialiased=True,
        )
        ax.set_title(f"Learned atom {k}")
        ax.set_xlabel("Base-atom coordinate")
        ax.set_ylabel("Iteration")
        ax.set_zlabel("Amplitude")
        ax.view_init(elev=28, azim=-125)

    fig.suptitle(
        f"Dictionary evolution: lambda_s={lambda_s:.6g}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_candidate(
    lambda_s, S, Sclean, Gtrue, Xtrue,
    scales, center, N, M, T, out_dir,
):
    K = Gtrue.shape[1]

    G = make_banco_boxes(M, K)
    H = np.zeros_like(G)

    shape = (N + M - 1, K, len(scales), T)
    X = np.zeros(shape)
    Y = np.zeros(shape)
    U = np.zeros(shape)

    Yprev = Y.copy()
    Gprev = G.copy()

    rho_x, rho_d = 1000.0, 500.0
    x_relax = d_relax = 1.8
    maxiter = 100

    Ghist = np.empty((maxiter + 1, K, M))
    Ghist[0] = G.T

    history = []

    print()
    print(f"lambda_s={lambda_s:.8g}")
    print(
        "iter cleanerr cosG   cosR   nz   "
        "rhoX    rhoD    rX      sX      "
        "rD      sD      dG"
    )

    for it in range(1, maxiter + 1):
        if it > 1:
            alpha, _ = compute_dictionary_alpha(
                G, scales, center, N
            )

            B = DictionaryForwardOperator(
                X=Y,
                alpha=alpha,
                scales=scales,
                center=center,
                N=N,
                M=M,
            )

            D, cginfo = solve_dictionary_cg(
                op=B,
                S=S,
                G=G,
                H=H,
                rho=rho_d,
                D0=G,
                rtol=1e-10,
                maxiter=500,
            )

            if cginfo != 0:
                raise RuntimeError(
                    f"dictionary CG failed: {cginfo}"
                )

            Gold = G.copy()

            Dr = d_relax * D + (1.0 - d_relax) * G
            G = project_dictionary_nonnegative(Dr + H)

            rd = float(np.linalg.norm(D - G))
            sd = float(
                rho_d * np.linalg.norm(Gprev - G)
            )

            H = H + Dr - G

            rho_d, H = banco_rho_update(
                rho_d,
                rd,
                sd,
                H,
                ratio=10.0,
                factor=2.0,
                dual_floor_boost=False,
            )

            dchange = float(np.linalg.norm(G - Gold))
            Gprev = G.copy()

        else:
            rd = sd = dchange = 0.0

        A = MultiscaleConvolutionOperator(
            D=G,
            scales=scales,
            center=center,
            N=N,
            normalize=True,
        )

        X, Y, U, _ = banco_coefficient_step(
            X, Y, U, S, A,
            lambda_s=lambda_s,
            rho=rho_x,
            iteration=it,
            x_relax=x_relax,
        )

        rx = float(np.linalg.norm(X - Y))
        sx = float(
            rho_x * np.linalg.norm(Yprev - Y)
        )

        rho_x, U = banco_rho_update(
            rho_x,
            rx,
            sx,
            U,
            ratio=10.0,
            factor=2.0,
            dual_floor_boost=True,
        )

        Yprev = Y.copy()

        Shat = A.forward(Y)

        cleanerr = float(
            np.linalg.norm(Shat - Sclean)
            / np.linalg.norm(Sclean)
        )

        noisyerr = float(
            np.linalg.norm(Shat - S)
            / np.linalg.norm(S)
        )

        C = similarity_matrix(G, Gtrue)
        match, order, cos_g, cos_r = best_match(C)

        nz = int(
            np.count_nonzero(np.abs(Y) > 1e-3)
        )

        usage = [
            float(np.sum(np.abs(Y[:, k, :, :])))
            for k in range(K)
        ]

        row = {
            "iteration": it,
            "penalty_phase": (
                "l1" if it <= 20 else "log"
            ),
            "clean_error": cleanerr,
            "noisy_error": noisyerr,
            "cosine_gaussian": float(cos_g),
            "cosine_ramp": float(cos_r),
            "matching": match,
            "matching_order": order,
            "significant_count": nz,
            "atom_usage": usage,
            "rho_x": float(rho_x),
            "rho_d": float(rho_d),
            "rx": rx,
            "sx": sx,
            "rd": rd,
            "sd": sd,
            "dictionary_change": dchange,
            "similarity_matrix": C.tolist(),
        }

        history.append(row)
        Ghist[it] = G.T

        print(
            f"{it:3d} "
            f"{cleanerr:8.2e} "
            f"{cos_g:6.4f} "
            f"{cos_r:6.4f} "
            f"{nz:4d} "
            f"{rho_x:7.1e} "
            f"{rho_d:7.1e} "
            f"{rx:7.1e} "
            f"{sx:7.1e} "
            f"{rd:7.1e} "
            f"{sd:7.1e} "
            f"{dchange:7.1e}"
        )

    final = history[-1]

    prefix = (
        f"lambda_{lambda_s:.8g}"
        .replace(".", "p")
    )

    save_final_plot(
        out_dir / f"{prefix}_final_atoms.png",
        G,
        Gtrue,
        final["matching"],
        lambda_s,
    )

    save_surface_plot(
        out_dir / f"{prefix}_atom_evolution.png",
        Ghist,
        lambda_s,
    )

    result = {
        "lambda_s": float(lambda_s),
        "true_nonzero_count": int(
            np.count_nonzero(Xtrue)
        ),
        "final": final,
        "final_dictionary": G.tolist(),
        "true_dictionary": Gtrue.tolist(),
        "dictionary_history": Ghist.tolist(),
        "iteration_history": history,
        "files": {
            "final_atoms_plot":
                f"{prefix}_final_atoms.png",
            "atom_evolution_plot":
                f"{prefix}_atom_evolution.png",
        },
    }

    json_path = out_dir / f"{prefix}_result.json"
    tmp_path = json_path.with_suffix(".json.tmp")

    tmp_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(json_path)

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/dictionary_learning/"
            "banco_lambda_s_sweep"
        ),
    )

    parser.add_argument(
        "--lambda-values",
        type=float,
        nargs="*",
        default=np.logspace(
            np.log10(0.05),
            np.log10(0.5),
            10,
        ).tolist(),
    )

    args = parser.parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    M, N, T = 41, 80, 80
    center = 21
    scales = BANCO_SCALES

    Gtrue = make_banco_dictionary(M)

    Atrue = MultiscaleConvolutionOperator(
        D=Gtrue,
        scales=scales,
        center=center,
        N=N,
        normalize=True,
    )

    Xtrue = make_banco_coefficients(
        T, M, N
    )

    Sclean = Atrue.forward(Xtrue)

    S, _, achieved_snr, sigma = add_noise_snr(
        Sclean,
        20.0,
        np.random.default_rng(100),
        return_sigma=True,
    )

    print("Banco lambda_s prefit sweep")
    print("---------------------------")
    print(
        f"achieved SNR: "
        f"{achieved_snr:.3f} dB"
    )
    print(f"sigma:        {sigma:.6e}")
    print(
        f"true nz:      "
        f"{np.count_nonzero(Xtrue)}"
    )
    print(f"lambdas:      {args.lambda_values}")

    results = []

    for lambda_s in args.lambda_values:
        results.append(
            run_candidate(
                lambda_s,
                S,
                Sclean,
                Gtrue,
                Xtrue,
                scales,
                center,
                N,
                M,
                T,
                args.output_dir,
            )
        )

    summary = {
        "requested_snr_db": 20.0,
        "achieved_snr_db": float(
            achieved_snr
        ),
        "sigma": float(sigma),
        "seed": 100,
        "dimensions": {
            "M": M,
            "N": N,
            "T": T,
            "K": 2,
            "J": len(scales),
        },
        "scales": [
            list(s) for s in scales
        ],
        "true_nonzero_count": int(
            np.count_nonzero(Xtrue)
        ),
        "candidates": [
            {
                "lambda_s": r["lambda_s"],
                "clean_error":
                    r["final"]["clean_error"],
                "noisy_error":
                    r["final"]["noisy_error"],
                "cosine_gaussian":
                    r["final"]["cosine_gaussian"],
                "cosine_ramp":
                    r["final"]["cosine_ramp"],
                "significant_count":
                    r["final"]["significant_count"],
                "rho_x":
                    r["final"]["rho_x"],
                "rho_d":
                    r["final"]["rho_d"],
                "dictionary_change":
                    r["final"]["dictionary_change"],
                "files": r["files"],
            }
            for r in results
        ],
    }

    path = (
        args.output_dir
        / "sweep_summary.json"
    )

    tmp = path.with_suffix(".json.tmp")

    tmp.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    tmp.replace(path)

    print("\nSweep summary")
    print("-------------")
    print(
        "lambda     cleanerr  cosG    cosR    "
        "nz   rhoX    rhoD"
    )

    for r in results:
        f = r["final"]

        print(
            f"{r['lambda_s']:8.5f} "
            f"{f['clean_error']:9.3e} "
            f"{f['cosine_gaussian']:7.4f} "
            f"{f['cosine_ramp']:7.4f} "
            f"{f['significant_count']:4d} "
            f"{f['rho_x']:7.1e} "
            f"{f['rho_d']:7.1e}"
        )

    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()