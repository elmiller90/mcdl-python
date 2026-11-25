import argparse
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
    den = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.vdot(a, b).real/max(den, 1e-15))


def similarities(G, Gtrue):
    C = np.array([
        [cosine(G[:, i], Gtrue[:, j]) for j in range(2)]
        for i in range(2)
    ])
    if C[0, 0]+C[1, 1] >= C[0, 1]+C[1, 0]:
        return C, "direct", C[0, 0], C[1, 1]
    return C, "swapped", C[1, 0], C[0, 1]


def run_stage(
    name, S, Sclean, Gtrue, G0, Y0,
    lambda_s, maxiter, scales, center, N, M,
    iteration_offset=0,
):
    G = G0.copy()
    Y = Y0.copy()

    X = Y.copy()
    U = np.zeros_like(Y)
    H = np.zeros_like(G)

    Yprev = Y.copy()
    Gprev = G.copy()

    rho_x, rho_d = 1000.0, 500.0
    x_relax = d_relax = 1.8

    Ghist = np.empty((maxiter+1, G.shape[0], G.shape[1]))
    Ghist[0] = G

    print(f"\n{name}")
    print("-"*len(name))
    print(
        "iter cleanerr cosG   cosR   nz   "
        "rhoX    rhoD    rX      sX      "
        "rD      sD      dG"
    )

    for it in range(1, maxiter+1):
        rho_x_step, rho_d_step = rho_x, rho_d

        if it > 1:
            alpha, _ = compute_dictionary_alpha(
                G, scales, center, N
            )
            B = DictionaryForwardOperator(
                X=Y, alpha=alpha, scales=scales,
                center=center, N=N, M=M,
            )
            D, cginfo = solve_dictionary_cg(
                op=B, S=S, G=G, H=H,
                rho=rho_d_step, D0=G,
                rtol=1e-10, maxiter=500,
            )
            if cginfo != 0:
                raise RuntimeError(
                    f"{name}: dictionary CG failed: {cginfo}"
                )

            Gold = G.copy()
            Dr = d_relax*D+(1.0-d_relax)*G
            G = project_dictionary_nonnegative(Dr+H)

            rd = float(np.linalg.norm(D-G))
            sd = float(
                rho_d_step*np.linalg.norm(Gprev-G)
            )
            H = H+Dr-G

            rho_d, H = banco_rho_update(
                rho_d_step, rd, sd, H,
                ratio=10.0, factor=2.0,
                dual_floor_boost=False,
            )
            dchange = float(np.linalg.norm(G-Gold))
            Gprev = G.copy()
        else:
            rd = sd = dchange = 0.0

        A = MultiscaleConvolutionOperator(
            D=G, scales=scales,
            center=center, N=N, normalize=True,
        )
        
        penalty_iteration = iteration_offset + it
        X, Y, U, _ = banco_coefficient_step(
            X, Y, U, S, A,
            lambda_s=lambda_s,
            rho=rho_x_step,
            iteration=penalty_iteration,
            x_relax=x_relax,
        )

        rx = float(np.linalg.norm(X-Y))
        sx = float(
            rho_x_step*np.linalg.norm(Yprev-Y)
        )
        rho_x, U = banco_rho_update(
            rho_x_step, rx, sx, U,
            ratio=10.0, factor=2.0,
            dual_floor_boost=True,
        )
        Yprev = Y.copy()
        Ghist[it] = G

        Shat = A.forward(Y)
        cleanerr = float(
            np.linalg.norm(Shat-Sclean)
            / np.linalg.norm(Sclean)
        )
        _, _, cos_g, cos_r = similarities(G, Gtrue)
        nz = int(np.count_nonzero(np.abs(Y) > 1e-3))

        print(
            f"{it:3d} {cleanerr:8.2e} "
            f"{cos_g:6.4f} {cos_r:6.4f} "
            f"{nz:4d} {rho_x:7.1e} {rho_d:7.1e} "
            f"{rx:7.1e} {sx:7.1e} "
            f"{rd:7.1e} {sd:7.1e} "
            f"{dchange:7.1e}"
        )

    C, match, cos_g, cos_r = similarities(G, Gtrue)
    A = MultiscaleConvolutionOperator(
        D=G, scales=scales,
        center=center, N=N, normalize=True,
    )
    cleanerr = float(
        np.linalg.norm(A.forward(Y)-Sclean)
        / np.linalg.norm(Sclean)
    )

    result = {
        "G": G,
        "Y": Y,
        "X": X,
        "Ghist": Ghist,
        "rho_x": rho_x,
        "rho_d": rho_d,
        "cleanerr": cleanerr,
        "similarity": C,
        "matching": match,
        "cos_g": float(cos_g),
        "cos_r": float(cos_r),
        "nz": int(np.count_nonzero(np.abs(Y) > 1e-3)),
    }

    print(f"\n{name} final")
    print("-"*(len(name)+6))
    print(C)
    print(f"matching:         {match}")
    print(f"clean error:      {cleanerr:.8f}")
    print(f"Gaussian cosine: {cos_g:.8f}")
    print(f"ramp cosine:     {cos_r:.8f}")
    print(f"nz:              {result['nz']}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-s", type=float, default=0.3)
    parser.add_argument("--prefit-iters", type=int, default=100)
    parser.add_argument("--main-iters", type=int, default=100)
    parser.add_argument(
        "--output",
        default="banco_prefit_then_mcdl_20db.npz",
    )
    args = parser.parse_args()

    M, N, T, K = 41, 80, 80, 2
    center = 21
    scales = BANCO_SCALES

    Gtrue = make_banco_dictionary(M)
    Atrue = MultiscaleConvolutionOperator(
        D=Gtrue, scales=scales,
        center=center, N=N, normalize=True,
    )
    Xtrue = make_banco_coefficients(T, M, N)
    Sclean = Atrue.forward(Xtrue)

    S, _, snr, sigma = add_noise_snr(
        Sclean, 20.0, np.random.default_rng(100),
        return_sigma=True,
    )

    shape = (Atrue.L, K, len(scales), T)

    print("Banco prefit followed by non-temporal MCDL")
    print("-------------------------------------------")
    print(f"achieved SNR: {snr:.3f} dB")
    print(f"sigma:        {sigma:.6e}")
    print(f"lambda_s:     {args.lambda_s:g}")
    print(f"true nz:      {np.count_nonzero(Xtrue)}")

    prefit = run_stage(
        name="Stage 1: non-temporal flat-box prefit",
        S=S,
        Sclean=Sclean,
        Gtrue=Gtrue,
        G0=make_banco_boxes(M, K),
        Y0=np.zeros(shape),
        lambda_s=args.lambda_s,
        maxiter=args.prefit_iters,
        scales=scales,
        center=center,
        N=N,
        M=M,
        iteration_offset=0,
    )

    # Dan's wrapper carries D and Y from the prefit.
    # Start a fresh ADMM solve with reset duals and rhos.
    main_result = run_stage(
        name="Stage 2: initialized non-temporal MCDL",
        S=S,
        Sclean=Sclean,
        Gtrue=Gtrue,
        G0=prefit["G"],
        Y0=prefit["Y"],
        lambda_s=args.lambda_s,
        maxiter=args.main_iters,
        scales=scales,
        center=center,
        N=N,
        M=M,
        iteration_offset=args.prefit_iters,
    )

    np.savez_compressed(
        args.output,
        lambda_s=args.lambda_s,
        achieved_snr=snr,
        sigma=sigma,
        Gtrue=Gtrue,
        Xtrue=Xtrue,
        Sclean=Sclean,
        S=S,
        G_prefit=prefit["G"],
        Y_prefit=prefit["Y"],
        Ghist_prefit=prefit["Ghist"],
        G_final=main_result["G"],
        Y_final=main_result["Y"],
        Ghist_main=main_result["Ghist"],
        similarity_prefit=prefit["similarity"],
        similarity_final=main_result["similarity"],
    )

    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()