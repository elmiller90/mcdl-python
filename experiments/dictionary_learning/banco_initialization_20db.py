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


def make_ramp(M, R=31):
    if M % 2 == 0 or R % 2 == 0:
        raise ValueError("M and R must be odd")
    if R > M:
        raise ValueError("R cannot exceed M")

    d = np.zeros(M, dtype=float)
    start = (M-R)//2
    d[start:start+R] = np.arange(R, dtype=float)
    d /= np.linalg.norm(d)
    return d


def make_dictionary(M):
    x = np.arange(M, dtype=float)
    c = (M-1)/2

    gaussian = np.exp(
        -0.5*((x-c)/8.0)**2
    )
    ramp = make_ramp(M, R=31)

    return project_dictionary_nonnegative(
        np.column_stack((gaussian, ramp))
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

def make_banco_boxes(M, K):
    D = np.zeros((M, K), dtype=float)

    for k in range(1, K+1):
        width = int(np.floor(M*k/(K+1)))

        # Force odd support for exact centering.
        if width % 2 == 0:
            width -= 1

        start = (M-width)//2
        D[start:start+width, k-1] = 1.0

    return project_dictionary_nonnegative(D)

def main():
    rng = np.random.default_rng(40)

    M, N, center = 41, 96, 21
    scales = [(1, 2), (3, 4), (1, 1)]

    Gtrue = make_dictionary(M)

    print(
        "true atom cosine:",
        f"{cosine(Gtrue[:, 0], Gtrue[:, 1]):.6f}"
    )

    Atrue = MultiscaleConvolutionOperator(
        D=Gtrue,
        scales=scales,
        center=center,
        N=N,
        normalize=True,
    )

    Xtrue = make_coefficients(Atrue.L, M)
    S = Atrue.forward(Xtrue)

    # Perturbed but labelled initialization.
    G = Gtrue.copy()
    G += 0.10*rng.random(G.shape)
    G = project_dictionary_nonnegative(G)
    H = np.zeros_like(G)

    lambda_s = 0.03
    rho_x, rho_d = 0.1, 1.0
    outer_maxiter = 25

    A = MultiscaleConvolutionOperator(
        D=G,
        scales=scales,
        center=center,
        N=N,
        normalize=True,
    )

    Xstart = A.adjoint(
        S,
        structured=True,
    )

    xmax = np.max(np.abs(Xstart))
    if xmax > 0:
        Xstart /= xmax

    state = None

    print()
    print("Noiseless Gaussian-ramp alternating recovery")
    print("--------------------------------------------")
    print(
        "iter   recon      Xerr      "
        "cos1      cos2      dG       nz"
    )

    for outer in range(1, outer_maxiter+1):

        # One stateful coefficient solve.
        A = MultiscaleConvolutionOperator(
            D=G,
            scales=scales,
            center=center,
            N=N,
            normalize=True,
        )

        _, Y, _, info = solve_log_admm_stateful(
            X0=Xstart,
            S=S,
            operator=A,
            lambda_s=lambda_s,
            lambda2=0.0,
            tau=1e-2,
            rho=rho_x,
            max_admm_iter=200,
            x_maxiter=50,
            abs_tol=1e-4,
            rel_tol=1e-3,
            adaptive_rho=True,
            rho_mu=10.0,
            rho_tau=2.0,
            a_min=1e-4,
            continuation_iters=100,
            a_fraction=0.95,
            verbose=False,
            temporal_regularizer=None,
            state0=state,
            continue_iteration=False,
        )

        state = info["state"]
        Xstart = Y.copy()

        # One dictionary update.
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
                f"Dictionary CG failed: {cginfo}"
            )

        Gnew = project_dictionary_nonnegative(
            D+H
        )

        H = H+D-Gnew
        dchange = np.linalg.norm(Gnew-G)
        G = Gnew

        # Diagnostics.
        Aeval = MultiscaleConvolutionOperator(
            D=G,
            scales=scales,
            center=center,
            N=N,
            normalize=True,
        )

        Shat = Aeval.forward(Y)
        recon = relerr(Shat, S)
        xerr = relerr(Y, Xtrue)
        C = similarity_matrix(G, Gtrue)

        nz = np.count_nonzero(
            np.abs(Y) > 1e-3
        )

        print(
            f"{outer:4d} "
            f"{recon:9.3e} "
            f"{xerr:9.3e} "
            f"{C[0,0]:9.6f} "
            f"{C[1,1]:9.6f} "
            f"{dchange:9.3e} "
            f"{nz:4d}"
        )

    C = similarity_matrix(G, Gtrue)
    Shat = Aeval.forward(Y)

    print("\nFinal similarity matrix")
    print("-----------------------")
    print(C)

    print(
        f"\nfinal reconstruction error: "
        f"{relerr(Shat, S):.3e}"
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
    for n, k, j, t in np.argwhere(
        np.abs(Y) > 1e-3
    ):
        print(
            f"t={t} atom={k} "
            f"pos={n-(M-1)} "
            f"scale={j} "
            f"amp={Y[n,k,j,t]:.6f}"
        )


if __name__ == "__main__":
    main()