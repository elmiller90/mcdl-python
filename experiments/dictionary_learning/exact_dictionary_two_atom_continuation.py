import numpy as np

from dictionary_alpha import compute_dictionary_alpha
from dictionary_cg import solve_dictionary_cg
from dictionary_forward_operator import DictionaryForwardOperator
from dictionary_projection import project_dictionary_nonnegative
from log_admm_stateful_compat import solve_log_admm_stateful
from multiscale_convolution import MultiscaleConvolutionOperator


def relerr(a, b):
    return np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-15)


def cosine(a, b):
    return np.vdot(a, b).real / (
        np.linalg.norm(a)*np.linalg.norm(b)
    )


def similarity_matrix(G, Gtrue):
    return np.array([
        [cosine(G[:, i], Gtrue[:, j])
         for j in range(Gtrue.shape[1])]
        for i in range(G.shape[1])
    ])


def make_dictionary(M):
    x = np.arange(M) - (M-1)/2
    g1 = np.exp(-0.5*(x/4.0)**2)
    g2 = (
        0.75*np.exp(-0.5*((x+6.0)/2.5)**2)
        + 0.35*np.exp(-0.5*((x-7.0)/3.0)**2)
    )
    return project_dictionary_nonnegative(
        np.column_stack((g1, g2))
    )


def make_coefficients(L, M):
    X = np.zeros((L, 2, 3, 6))

    p1 = [25, 26, 27, 28, 29, 30]
    j1 = [0, 1, 2, 2, 1, 0]

    p2 = [55, 54, 53, 52, 51, 50]
    j2 = [2, 1, 0, 0, 1, 2]

    for t in range(6):
        X[(M-1)+p1[t], 0, j1[t], t] = 1.0
        X[(M-1)+p2[t], 1, j2[t], t] = 1.0

    return X


def main():
    rng = np.random.default_rng(40)

    M, N, center = 41, 96, 21
    scales = [(1, 2), (1, 1), (2, 1)]

    Gtrue = make_dictionary(M)

    Atrue = MultiscaleConvolutionOperator(
        D=Gtrue, scales=scales, center=center,
        N=N, normalize=True,
    )

    Xtrue = make_coefficients(Atrue.L, M)
    S = Atrue.forward(Xtrue)

    # Perturbed but labelled initial atoms.
    G = Gtrue.copy()


    lambda_s = 0.03
    rho_x, rho_d = 0.1, 1.0
    outer_maxiter = 25

    A = MultiscaleConvolutionOperator(
        D=G, scales=scales, center=center,
        N=N, normalize=True,
    )

    Xstart = A.adjoint(S, structured=True)
    xmax = np.max(np.abs(Xstart))
    if xmax > 0:
        Xstart /= xmax

    state = None

    print("Alternating log-X/D two-atom recovery")
    print("-------------------------------------")
    print("iter   recon      Xerr      cos1      cos2      dG       nz")

    for outer in range(1, outer_maxiter+1):
        A = MultiscaleConvolutionOperator(
            D=G, scales=scales, center=center,
            N=N, normalize=True,
        )

        _, Y, _, info = solve_log_admm_stateful(
            X0=Xstart, S=S, operator=A,
            lambda_s=lambda_s, lambda2=0.0, tau=1e-2,
            rho=rho_x, max_admm_iter=200, x_maxiter=50,
            abs_tol=1e-4, rel_tol=1e-3, adaptive_rho=True,
            rho_mu=10.0, rho_tau=2.0, a_min=1e-4,
            continuation_iters=100, a_fraction=0.95,
            verbose=False, temporal_regularizer=None,
            state0=state, continue_iteration=False,
        )

        state = info["state"]

        dchange = 0.0
 
        Aeval = MultiscaleConvolutionOperator(
            D=G, scales=scales, center=center,
            N=N, normalize=True,
        )

        recon = relerr(Aeval.forward(Y), S)
        xerr = relerr(Y, Xtrue)
        C = similarity_matrix(G, Gtrue)
        nz = np.count_nonzero(np.abs(Y) > 1e-3)

        print(
            f"{outer:4d} {recon:9.3e} {xerr:9.3e} "
            f"{C[0,0]:9.6f} {C[1,1]:9.6f} "
            f"{dchange:9.3e} {nz:4d}"
        )

        Xstart = Y.copy()

    C = similarity_matrix(G, Gtrue)

    print("\nFinal similarity matrix")
    print("-----------------------")
    print(C)

    print(
        f"\nfinal reconstruction error: "
        f"{relerr(Aeval.forward(Y), S):.3e}"
    )
    print(
        f"final coefficient error:    "
        f"{relerr(Y, Xtrue):.3e}"
    )
    print(
        f"significant coefficients:   "
        f"{np.count_nonzero(np.abs(Y) > 1e-3)}"
    )

    print("\nSignificant coefficients:")
    for n, k, j, t in np.argwhere(np.abs(Y) > 1e-3):
        print(
            f"t={t} atom={k} pos={n-(M-1)} "
            f"scale={j} amp={Y[n,k,j,t]:.6f}"
        )


if __name__ == "__main__":
    main()