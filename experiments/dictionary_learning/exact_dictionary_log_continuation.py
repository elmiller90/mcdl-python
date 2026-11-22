import numpy as np

from log_admm_stateful_compat import solve_log_admm_stateful
from multiscale_convolution import MultiscaleConvolutionOperator


def relerr(a, b):
    return np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-15)


def make_dictionary(M):
    x = np.arange(M, dtype=float)
    c = (M-1)/2
    d = np.exp(-0.5*((x-c)/4.0)**2)
    return (d/np.linalg.norm(d))[:, None]


def make_coefficients(L, M):
    X = np.zeros((L, 1, 3, 5))
    for t, (p, j) in enumerate(zip(
        [36, 37, 38, 39, 40], [0, 1, 2, 1, 0]
    )):
        X[(M-1)+p, 0, j, t] = 1.0
    return X


def main():
    M, N, center = 41, 96, 21
    scales = [(1, 2), (1, 1), (2, 1)]

    G = make_dictionary(M)
    A = MultiscaleConvolutionOperator(
        D=G, scales=scales, center=center,
        N=N, normalize=True,
    )

    Xtrue = make_coefficients(A.L, M)
    S = A.forward(Xtrue)

    Xstart = A.adjoint(S, structured=True)
    Xstart /= np.max(np.abs(Xstart))

    state = None

    print("Exact-D repeated log continuation")
    print("---------------------------------")
    print("pass   recon      Xerr       nz")

    for outer in range(1, 21):
        _, Y, _, info = solve_log_admm_stateful(
            X0=Xstart, S=S, operator=A,
            lambda_s=0.03, lambda2=0.0, tau=1e-2,
            rho=0.1, max_admm_iter=200, x_maxiter=50,
            abs_tol=1e-4, rel_tol=1e-3, adaptive_rho=True,
            rho_mu=10.0, rho_tau=2.0, a_min=1e-4,
            continuation_iters=100, a_fraction=0.95,
            verbose=False, temporal_regularizer=None,
            state0=state, continue_iteration=False,
        )

        state = info["state"]
        Xstart = Y.copy()

        recon = relerr(A.forward(Y), S)
        xerr = relerr(Y, Xtrue)
        nz = np.count_nonzero(np.abs(Y) > 1e-3)

        print(
            f"{outer:4d} {recon:9.3e} "
            f"{xerr:9.3e} {nz:4d}"
        )

    print("\nSignificant coefficients:")
    for n, k, j, t in np.argwhere(np.abs(Y) > 1e-3):
        print(
            f"t={t} pos={n-(M-1)} scale={j} "
            f"amp={Y[n,k,j,t]:.6f}"
        )


if __name__ == "__main__":
    main()