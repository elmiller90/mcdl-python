import numpy as np

from dictionary_alpha import compute_dictionary_alpha
from dictionary_cg import solve_dictionary_cg
from dictionary_forward_operator import DictionaryForwardOperator
from dictionary_initialization import make_banco_boxes
from dictionary_projection import project_dictionary_nonnegative
from gaussian_experiment import add_noise_snr
from log_admm_stateful_compat import solve_log_admm_stateful
from multiscale_convolution import MultiscaleConvolutionOperator
from experiments.dictionary_learning.alternating_two_atom_log_noisy import (
    relerr, cosine, similarity_matrix, make_dictionary,
)


def best_match(C):
    diag = C[0, 0] + C[1, 1]
    swap = C[0, 1] + C[1, 0]

    if diag >= swap:
        return "direct", C[0, 0], C[1, 1]

    return "swapped", C[0, 1], C[1, 0]

def make_coefficients(L, M, T):
    X = np.zeros((L, 2, 3, T), dtype=float)

    p0 = np.rint(np.linspace(20, 38, T)).astype(int)
    p1 = np.rint(np.linspace(65, 47, T)).astype(int)

    cycle = np.array([0, 0, 1, 1, 2, 2, 1, 1])
    j0 = np.resize(cycle, T)
    j1 = np.resize(cycle[::-1], T)

    for t in range(T):
        X[(M-1)+p0[t], 0, j0[t], t] = 1.0
        X[(M-1)+p1[t], 1, j1[t], t] = 1.0

    return X


def main():
    M, N, K, T, center = 41, 96, 2, 40, 21
    scales = [(1, 2), (3, 4), (1, 1)]

    Gtrue = make_dictionary(M)
    Atrue = MultiscaleConvolutionOperator(
        D=Gtrue, scales=scales, center=center,
        N=N, normalize=True,
    )

    Xtrue = make_coefficients(Atrue.L, M, T)
    Sclean = Atrue.forward(Xtrue)
    S, _, achieved_snr, sigma = add_noise_snr(
        Sclean, 20.0, np.random.default_rng(100),
        return_sigma=True,
    )

    # MATLAB-compatible flat initialization: widths 21 and 41.
    G = make_banco_boxes(M, K)
    H = np.zeros_like(G)

    # MATLAB prefit starts coefficient-side variables at zero.
    shape = (Atrue.L, K, len(scales), T)
    Xstart = np.zeros(shape, dtype=float)
    Y = np.zeros(shape, dtype=float)
    state = None

    lambda_s = 0.1
    rho_x, rho_d = 0.1, 1.0
    x_relax, d_relax = 1.8, 1.8
    outer_maxiter = 100

    print("Banco-style coupled 20 dB prefit")
    print("--------------------------------")
    print(f"achieved SNR: {achieved_snr:.2f} dB")
    print(f"sigma:        {sigma:.3e}")
    print("\nInitial similarity matrix:")
    print(similarity_matrix(G, Gtrue))
    print("iter   cleanerr   cosG      cosR      match    dG       nz    use0    use1")


    for outer in range(1, outer_maxiter + 1):
        # MATLAB order: D -> G -> alpha -> H.
        # The first main iteration skips the D update.
        if outer > 1:
            alpha, _ = compute_dictionary_alpha(
                G, scales, center, N
            )
            B = DictionaryForwardOperator(
                X=Y, alpha=alpha, scales=scales,
                center=center, N=N, M=M,
            )
            D, cginfo = solve_dictionary_cg(
                op=B, S=S, G=G, H=H, rho=rho_d,
                D0=G, rtol=1e-10, maxiter=500,
            )
            if cginfo != 0:
                raise RuntimeError(
                    f"Dictionary CG failed: {cginfo}"
                )
            Dr = d_relax*D + (1.0-d_relax)*G
            Gnew = project_dictionary_nonnegative(Dr + H)
            H = H + Dr - Gnew
            dchange = np.linalg.norm(Gnew - G)
            G = Gnew
        else:
            dchange = 0.0

        # MATLAB order continues: one X -> Y -> U update.
        A = MultiscaleConvolutionOperator(
            D=G, scales=scales, center=center,
            N=N, normalize=True,
        )
        _, Y, _, info = solve_log_admm_stateful(
            X0=Xstart, S=S, operator=A,
            lambda_s=lambda_s, lambda2=0.0, tau=1e-2,
            rho=rho_x, max_admm_iter=1, x_maxiter=50, 
            abs_tol=1e-4, rel_tol=1e-3, adaptive_rho=True,
            rho_mu=10.0, rho_tau=2.0, a_min=1e-4,
            continuation_iters=100, a_fraction=0.95,
            verbose=False, temporal_regularizer=None,
            state0=state, continue_iteration=True,
            x_relax=x_relax,
        )

        state = info["state"]
        Xstart = Y.copy()
        
        if outer == 1 or outer % 1 == 0:
            
            usage0 = np.sum(np.abs(Y[:, 0, :, :]))
            usage1 = np.sum(np.abs(Y[:, 1, :, :]))
            
            Aeval = MultiscaleConvolutionOperator(
                D=G, scales=scales, center=center,
                N=N, normalize=True,
            )
            cleanerr = relerr(Aeval.forward(Y), Sclean)
            C = similarity_matrix(G, Gtrue)
            match, cos_g, cos_r = best_match(C)
            nz = np.count_nonzero(np.abs(Y) > 1e-3)

            print(
                f"{outer:4d} {cleanerr:10.3e} "
                f"{cos_g:9.6f} {cos_r:9.6f} "
                f"{match:7s} {dchange:9.3e} {nz:4d}"
                f"{usage0:7.2f} {usage1:7.2f}"
            )

    C = similarity_matrix(G, Gtrue)
    match, cos_g, cos_r = best_match(C)
    alpha, norm_vals = compute_dictionary_alpha(
        G, scales, center, N
    )

    print("\nFinal similarity matrix")
    print("-----------------------")
    print(C)
    print(f"\nbest matching:   {match}")
    print(f"Gaussian cosine: {cos_g:.8f}")
    print(f"ramp cosine:     {cos_r:.8f}")

    print("\nPrefit retained state")
    print("---------------------")
    print("G shape:", G.shape)
    print("alpha shape:", alpha.shape)
    print("NormVals shape:", norm_vals.shape)

if __name__ == "__main__":
    main()