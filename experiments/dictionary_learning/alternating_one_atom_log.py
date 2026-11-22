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


def make_dictionary(M, width):
    x = np.arange(M, dtype=float)
    c = (M-1)/2
    d = np.exp(-0.5*((x-c)/width)**2)
    return (d/np.linalg.norm(d))[:, None]


def make_coefficients(L, M):
    X = np.zeros((L, 1, 3, 5))
    for t, (p, j) in enumerate(zip(
        [36, 37, 38, 39, 40],
        [0, 1, 2, 1, 0],
    )):
        X[(M-1)+p, 0, j, t] = 1.0
    return X


def dominant_trajectory(X, M):
    pos, scales, amps = [], [], []
    for t in range(X.shape[3]):
        f = np.abs(X[:, 0, :, t])
        n, j = np.unravel_index(np.argmax(f), f.shape)
        pos.append(int(n-(M-1)))
        scales.append(int(j))
        amps.append(float(X[n, 0, j, t]))
    return pos, scales, amps


def main():
    rng = np.random.default_rng(30)

    M, N, center = 41, 96, 21
    scales = [(1, 2), (1, 1), (2, 1)]

    Gtrue = make_dictionary(M, 4.0)
    Atrue = MultiscaleConvolutionOperator(
        D=Gtrue, scales=scales, center=center,
        N=N, normalize=True,
    )

    Xtrue = make_coefficients(Atrue.L, M)
    S = Atrue.forward(Xtrue)

    # Perturbed initial dictionary.
    G = make_dictionary(M, 6.0)
    G += 0.03*rng.random(G.shape)
    G = project_dictionary_nonnegative(G)
    H = np.zeros_like(G)

    lambda_s = 0.03
    rho_x, rho_d = 0.1, 1.0
    outer_maxiter = 20

    A = MultiscaleConvolutionOperator(
        D=G, scales=scales, center=center,
        N=N, normalize=True,
    )

    Xstart = A.adjoint(S, structured=True)
    xmax = np.max(np.abs(Xstart))
    if xmax > 0:
        Xstart /= xmax

    state = None

    print("Alternating log-X/D one-atom recovery")
    print("-------------------------------------")
    print("iter   recon      Xerr       Dcos       Dchg      nz")

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

        Gnew = project_dictionary_nonnegative(D + H)
        H = H + D - Gnew

        dchange = np.linalg.norm(Gnew-G)
        G = Gnew

        Aeval = MultiscaleConvolutionOperator(
            D=G, scales=scales, center=center,
            N=N, normalize=True,
        )

        recon = relerr(Aeval.forward(Y), S)
        xerr = relerr(Y, Xtrue)
        dcos = cosine(G[:, 0], Gtrue[:, 0])
        nz = np.count_nonzero(np.abs(Y) > 1e-3)

        print(
            f"{outer:4d} {recon:9.3e} {xerr:9.3e} "
            f"{dcos:10.8f} {dchange:9.3e} {nz:4d}"
        )

        Xstart = Y.copy()

    pos, scale_ids, amps = dominant_trajectory(Y, M)

    print("\nFinal trajectory")
    print("----------------")
    print("positions:", pos)
    print("scales:   ", scale_ids)
    print("amplitudes:", [round(a, 6) for a in amps])

    print("\nTruth")
    print("-----")
    print("positions:", [36, 37, 38, 39, 40])
    print("scales:   ", [0, 1, 2, 1, 0])

    print(
        f"\nfinal dictionary cosine: "
        f"{cosine(G[:, 0], Gtrue[:, 0]):.10f}"
    )
    print(
        f"final coefficient error: "
        f"{relerr(Y, Xtrue):.3e}"
    )

    print("\nSignificant coefficients:")
    for n, k, j, t in np.argwhere(np.abs(Y) > 1e-3):
        print(
            f"t={t} pos={n-(M-1)} scale={j} "
            f"amp={Y[n,k,j,t]:.6f}"
        )
    assert relerr(Y, Xtrue) < 1e-3
    assert cosine(G[:, 0], Gtrue[:, 0]) > 0.999999
    assert np.count_nonzero(np.abs(Y) > 1e-3) == 5
    assert pos == [36, 37, 38, 39, 40]
    assert scale_ids == [0, 1, 2, 1, 0]

if __name__ == "__main__":
    main()