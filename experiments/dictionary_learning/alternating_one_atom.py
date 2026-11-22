import numpy as np

from dictionary_alpha import compute_dictionary_alpha
from dictionary_cg import solve_dictionary_cg
from dictionary_forward_operator import DictionaryForwardOperator
from dictionary_projection import project_dictionary_nonnegative
from multiscale_convolution import MultiscaleConvolutionOperator
from sparse_admm import solve_l1_admm


def relerr(a, b):
    return np.linalg.norm(a - b) / max(
        np.linalg.norm(b), 1e-15
    )


def cosine(a, b):
    return np.vdot(a, b).real / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def make_dictionary(M, width):
    x = np.arange(M, dtype=float)
    c = (M - 1) / 2.0
    d = np.exp(
        -0.5 * ((x - c) / width) ** 2
    )
    d /= np.linalg.norm(d)
    return d[:, None]


def make_coefficients(L, M):
    X = np.zeros((L, 1, 3, 5))

    positions = [36, 37, 38, 39, 40]
    scales = [0, 1, 2, 1, 0]

    for t, (p, j) in enumerate(
        zip(positions, scales)
    ):
        n = (M - 1) + p
        X[n, 0, j, t] = 1.0

    return X


def dominant_trajectory(X, M):
    positions = []
    scales = []

    for t in range(X.shape[3]):
        frame = np.abs(X[:, 0, :, t])
        n, j = np.unravel_index(
            np.argmax(frame), frame.shape
        )
        positions.append(int(n - (M - 1)))
        scales.append(int(j))

    return positions, scales


def main():
    rng = np.random.default_rng(30)

    M = 41
    N = 96
    center = 21
    scales = [(1, 2), (1, 1), (2, 1)]

    # True problem.
    Gtrue = make_dictionary(M, width=4.0)

    true_op = MultiscaleConvolutionOperator(
        D=Gtrue,
        scales=scales,
        center=center,
        N=N,
        normalize=True,
    )

    L = true_op.L
    Xtrue = make_coefficients(L, M)
    S = true_op.forward(Xtrue)
    
    ###
    X0_trueD = true_op.adjoint(
        S,
        structured=True,
    )
    lambda1 = 0.003
    rho_x = 0.1

    xmax = np.max(np.abs(X0_trueD))
    if xmax > 0:
        X0_trueD /= xmax

    _, Y_trueD, _, info_trueD = solve_l1_admm(
        X0=X0_trueD,
        S=S,
        operator=true_op,
        lambda1=lambda1,
        lambda2=0.0,
        tau=1e-2,
        rho=rho_x,
        max_admm_iter=200,
        x_maxiter=50,
        abs_tol=1e-4,
        rel_tol=1e-3,
        nonnegative=True,
        adaptive_rho=True,
        rho_mu=10.0,
        rho_tau=2.0,
        verbose=False,
    )

    print()
    print("Exact-dictionary coefficient control")
    print("------------------------------------")
    print(
        f"coefficient error: "
        f"{relerr(Y_trueD, Xtrue):.3e}"
    )
    print(
        f"significant coefficients: "
        f"{np.count_nonzero(np.abs(Y_trueD) > 1e-3)}"
    )
    print(
        f"reconstruction error: "
        f"{relerr(true_op.forward(Y_trueD), S):.3e}"
    )
    ###
    # Deliberately incorrect feasible dictionary.
    G = make_dictionary(M, width=6.0)

    # Small asymmetric perturbation so initialization is
    # not merely another member of the Gaussian family.
    G += 0.03 * rng.random(G.shape)
    G = project_dictionary_nonnegative(G)

    H = np.zeros_like(G)

    # Coefficient parameters.
    lambda1 = 0.003
    rho_x = 0.1

    # Dictionary ADMM parameter.
    rho_d = 1.0

    outer_maxiter = 20

    # Initial coefficient estimate from current dictionary.
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

    print("Alternating X/D one-atom recovery")
    print("----------------------------------")
    print(
        "iter  recon       Xerr        "
        "Dcos        Dchg        nz"
    )

    for outer in range(1, outer_maxiter + 1):
        # -----------------------------------------
        # X update with current projected dictionary.
        # -----------------------------------------
        A = MultiscaleConvolutionOperator(
            D=G,
            scales=scales,
            center=center,
            N=N,
            normalize=True,
        )

        Xsmooth, Y, U, xinfo = solve_l1_admm(
            X0=Xstart,
            S=S,
            operator=A,
            lambda1=lambda1,
            lambda2=0.0,
            tau=1e-2,
            rho=rho_x,
            max_admm_iter=200,
            x_maxiter=50,
            abs_tol=1e-4,
            rel_tol=1e-3,
            nonnegative=True,
            adaptive_rho=True,
            rho_mu=10.0,
            rho_tau=2.0,
            verbose=False,
        )

        # MATLAB dictionary update uses sparse Y.
        Xdict = Y

        # -----------------------------------------
        # D update with Y and alpha fixed.
        # -----------------------------------------
        alpha, _ = compute_dictionary_alpha(
            G,
            scales,
            center,
            N,
        )

        B = DictionaryForwardOperator(
            X=Xdict,
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
            D + H
        )

        H = H + D - Gnew

        dchange = np.linalg.norm(Gnew - G)
        G = Gnew

        # -----------------------------------------
        # Diagnostics using current G and sparse Y.
        # -----------------------------------------
        Aeval = MultiscaleConvolutionOperator(
            D=G,
            scales=scales,
            center=center,
            N=N,
            normalize=True,
        )

        recon = relerr(
            Aeval.forward(Y),
            S,
        )

        xerr = relerr(Y, Xtrue)

        dcos = cosine(
            G[:, 0],
            Gtrue[:, 0],
        )

        nz = np.count_nonzero(
            np.abs(Y) > 1e-3
        )

        print(
            f"{outer:4d} "
            f"{recon:10.3e} "
            f"{xerr:10.3e} "
            f"{dcos:11.8f} "
            f"{dchange:10.3e} "
            f"{nz:4d}"
        )

        # Warm-start next outer X solve from sparse Y.
        Xstart = Y.copy()

    positions, scale_ids = dominant_trajectory(
        Y, M
    )

    print()
    print("Final trajectory")
    print("----------------")
    print("positions:", positions)
    print("scales:   ", scale_ids)

    print()
    print("Truth")
    print("-----")
    print("positions:", [36, 37, 38, 39, 40])
    print("scales:   ", [0, 1, 2, 1, 0])

    print()
    print(
        f"final dictionary cosine: "
        f"{cosine(G[:, 0], Gtrue[:, 0]):.10f}"
    )
    print(
        f"final coefficient error: "
        f"{relerr(Y, Xtrue):.3e}"
    )


if __name__ == "__main__":
    main()