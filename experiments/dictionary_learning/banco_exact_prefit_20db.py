import numpy as np

from banco_coefficient_step import banco_coefficient_step
from banco_gaussian_ramp import (
    BANCO_SCALES,
    make_banco_coefficients,
    make_banco_dictionary,
)
from banco_rho_update import banco_rho_update
from banco_sparse_penalty import banco_sparse_penalty
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


def similarity_matrix(G, Gtrue):
    return np.array([
        [cosine(G[:, i], Gtrue[:, j]) for j in range(2)]
        for i in range(2)
    ])


def best_match(C):
    if C[0, 0]+C[1, 1] >= C[0, 1]+C[1, 0]:
        return "direct", C[0, 0], C[1, 1]
    return "swapped", C[1, 0], C[0, 1]


def main():
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

    G = make_banco_boxes(M, K)
    H = np.zeros_like(G)

    shape = (Atrue.L, K, len(scales), T)
    X = np.zeros(shape)
    Y = np.zeros(shape)
    U = np.zeros(shape)

    Yprev = Y.copy()
    Gprev = G.copy()

    lambda_s = 0.3
    rho_x, rho_d = 1000.0, 500.0
    x_relax = d_relax = 1.8
    maxiter = 100

    min_objective = np.inf
    Gmin = G.copy()
    Ymin = Y.copy()
    min_iter = 0

    best_ramp_cos = -np.inf
    Gbest_ramp = G.copy()
    best_ramp_iter = 0

    print("Exact Banco-style 20 dB prefit")
    print("------------------------------")
    print(f"achieved SNR: {snr:.2f} dB")
    print(f"sigma:        {sigma:.3e}")
    print(f"J:            {len(scales)}")
    print(f"lambda_s:     {lambda_s:g}")
    print(f"rho_x(0):     {rho_x:g}")
    print(f"rho_d(0):     {rho_d:g}")

    print("\nInitial similarity matrix:")
    print(similarity_matrix(G, Gtrue))

    print(
        "\niter cleanerr cosG   cosR   nz   "
        "rhoX    rhoD    rX      sX      "
        "rD      sD      dG      obj"
    )

    for it in range(1, maxiter+1):
        # Save penalties used for this iteration before adaptation.
        rho_x_step = rho_x
        rho_d_step = rho_d

        # Dictionary ADMM step, skipped at iteration 1.
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
                    f"dictionary CG failed: {cginfo}"
                )

            Gold = G.copy()
            Dr = d_relax*D+(1.0-d_relax)*G
            G = project_dictionary_nonnegative(Dr+H)

            rd = float(np.linalg.norm(D-G))
            sd = float(
                rho_d_step*np.linalg.norm(Gprev-G)
            )
            Jrho_d = float(
                0.5*rho_d_step*np.sum((Dr-G)**2)
            )

            H = H+Dr-G
            rho_d, H = banco_rho_update(
                rho_d_step,
                primal=rd,
                dual=sd,
                scaled_dual=H,
                ratio=10.0,
                factor=2.0,
                dual_floor_boost=False,
            )

            dchange = float(np.linalg.norm(G-Gold))
            Gprev = G.copy()
        else:
            rd = sd = dchange = Jrho_d = 0.0

        # Coefficient ADMM step.
        A = MultiscaleConvolutionOperator(
            D=G, scales=scales,
            center=center, N=N, normalize=True,
        )
        X, Y, U, _ = banco_coefficient_step(
            X, Y, U, S, A,
            lambda_s=lambda_s,
            rho=rho_x_step,
            iteration=it,
            x_relax=x_relax,
        )

        # Reconstruct the relaxed variable using Y from the
        # beginning of the current coefficient step.
        Xr = x_relax*X+(1.0-x_relax)*Yprev

        # Banco objective, before coefficient rho adaptation.
        Shat = A.forward(Y)
        Jdf = float(0.5*np.sum((Shat-S)**2))
        Rsp = banco_sparse_penalty(
            Y, lambda_s=lambda_s,
            iteration=it, l1_iters=20,
        )
        Jsp = float(lambda_s*Rsp)
        Jrho_x = float(
            0.5*rho_x_step*np.sum((Xr-Y)**2)
        )
        objective = Jdf+Jsp+Jrho_x+Jrho_d

        # Standard residuals used by Dan's code.
        rx = float(np.linalg.norm(X-Y))
        sx = float(
            rho_x_step*np.linalg.norm(Yprev-Y)
        )
        rho_x, U = banco_rho_update(
            rho_x_step,
            primal=rx,
            dual=sx,
            scaled_dual=U,
            ratio=10.0,
            factor=2.0,
            dual_floor_boost=True,
        )

        Yprev = Y.copy()

        cleanerr = float(
            np.linalg.norm(Shat-Sclean)
            / np.linalg.norm(Sclean)
        )
        C = similarity_matrix(G, Gtrue)
        match, cos_g, cos_r = best_match(C)

        if objective < min_objective:
            min_objective = objective
            Gmin = G.copy()
            Ymin = Y.copy()
            min_iter = it

        if cos_r > best_ramp_cos:
            best_ramp_cos = cos_r
            Gbest_ramp = G.copy()
            best_ramp_iter = it

        nz = int(np.count_nonzero(np.abs(Y) > 1e-3))

        print(
            f"{it:3d} {cleanerr:8.2e} "
            f"{cos_g:6.4f} {cos_r:6.4f} "
            f"{nz:4d} {rho_x:7.1e} {rho_d:7.1e} "
            f"{rx:7.1e} {sx:7.1e} "
            f"{rd:7.1e} {sd:7.1e} "
            f"{dchange:7.1e} {objective:9.2e}"
        )

    C = similarity_matrix(G, Gtrue)
    match, cos_g, cos_r = best_match(C)

    Cmin = similarity_matrix(Gmin, Gtrue)
    match_min, cg_min, cr_min = best_match(Cmin)

    Cbest = similarity_matrix(Gbest_ramp, Gtrue)
    match_best, cg_best, cr_best = best_match(Cbest)

    alpha, norm_vals = compute_dictionary_alpha(
        G, scales, center, N
    )

    print("\nFinal similarity matrix")
    print("-----------------------")
    print(C)
    print(f"\nbest matching:   {match}")
    print(f"Gaussian cosine: {cos_g:.8f}")
    print(f"ramp cosine:     {cos_r:.8f}")

    print("\nMinimum-objective iterate")
    print("-------------------------")
    print(f"iteration:        {min_iter}")
    print(f"objective:        {min_objective:.6e}")
    print(f"matching:         {match_min}")
    print(f"Gaussian cosine: {cg_min:.8f}")
    print(f"ramp cosine:     {cr_min:.8f}")

    print("\nBest-ramp iterate (diagnostic only)")
    print("-----------------------------------")
    print(f"iteration:        {best_ramp_iter}")
    print(f"matching:         {match_best}")
    print(f"Gaussian cosine: {cg_best:.8f}")
    print(f"ramp cosine:     {cr_best:.8f}")

    print("\nFinal ADMM parameters")
    print("---------------------")
    print(f"rho_x: {rho_x:.6g}")
    print(f"rho_d: {rho_d:.6g}")

    print("\nPrefit retained state")
    print("---------------------")
    print("G shape:", G.shape)
    print("Y shape:", Y.shape)
    print("Ymin shape:", Ymin.shape)
    print("alpha shape:", alpha.shape)
    print("NormVals shape:", norm_vals.shape)


if __name__ == "__main__":
    main()